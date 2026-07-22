"""Signature-tracing MDP terms for mjlab: a command term that assigns each
env a recorded signature path (and places the car's pencil tip at its start),
plus the reward and termination functions of the accuracy-gated objective
from rl/signature_env.py, vectorized in torch over the batched sim.

Faithful to rl/signature_env.py Run-3 semantics:
  - pure-pursuit style observation [dx_local, dy_local, dist_to_final,
    at_end], same OBS scaling (0.01 / 0.01 / 0.1 / 1);
  - windowed nearest-point follower (search_window ahead), local-window
    tracking error (+- local_err_window points), accuracy-gated arc-length
    progress with err_gate_mm, off-path termination, finish detection;
  - initial pose randomization (xy disk + yaw) around the path start.

Deliberate difference: the action space is the two wheel EFFORTS directly
(JointEffortActionCfg), not (v, omega) through a PI wheel-speed loop - the
mjlab policy learns motor control end-to-end.

Timing note (see ManagerBasedRlEnv.step): the command term's buffers are
updated AFTER rewards/terminations each step, so reward and termination
terms read values that are one control step (20 ms) stale. This is a fixed
property of the manager call order and is benign for RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Same feature scaling as rl/signature_env.py OBS_SCALE.
_OBS_SCALE = (0.01, 0.01, 0.1, 1.0)


class SignaturePathCommand(CommandTerm):
  """Per-env signature path + pure-pursuit lookahead features.

  The 4-dim command is the scaled observation [dx_local, dy_local,
  dist_to_final, at_end]. Reward/termination terms read the extra buffers
  (err_mm, progress_mm, gate, finished) via
  env.command_manager.get_term("signature_path").
  """

  cfg: "SignaturePathCommandCfg"

  def __init__(self, cfg: "SignaturePathCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    if not cfg.paths:
      raise ValueError("SignaturePathCommandCfg.paths is empty")
    max_len = max(len(p) for p in cfg.paths)
    paths = np.stack(
      [np.pad(p, ((0, max_len - len(p)), (0, 0)), mode="edge") for p in cfg.paths]
    )
    self._paths = torch.tensor(paths, dtype=torch.float32, device=self.device)
    self._path_len = torch.tensor(
      [len(p) for p in cfg.paths], dtype=torch.long, device=self.device
    )

    E = self.num_envs
    self._env_path = torch.zeros(E, dtype=torch.long, device=self.device)
    self._idx = torch.zeros(E, dtype=torch.long, device=self.device)
    self._prev_idx = torch.zeros(E, dtype=torch.long, device=self.device)
    self.err_mm = torch.zeros(E, device=self.device)
    self.progress_mm = torch.zeros(E, device=self.device)
    self.gate = torch.zeros(E, device=self.device)
    self.dist_final = torch.zeros(E, device=self.device)
    self.at_end = torch.zeros(E, dtype=torch.bool, device=self.device)
    self.finished = torch.zeros(E, dtype=torch.bool, device=self.device)
    self._command = torch.zeros(E, 4, device=self.device)

    self.metrics["tracking_err_mm"] = torch.zeros(E, device=self.device)
    self._debug_vis_ok = True

  @property
  def command(self) -> torch.Tensor:
    return self._command

  # -- helpers ---------------------------------------------------------------

  def _tip_and_yaw(self):
    """Pencil-tip xy in each env's LOCAL (paper) frame, and chassis yaw.
    The tip is a rigid offset behind the chassis, so it is computed from the
    root pose instead of a site lookup."""
    pos_w = self.robot.data.root_link_pos_w
    quat = self.robot.data.root_link_quat_w  # (w, x, y, z)
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    tip_x = pos_w[:, 0] + c * self.cfg.tip_offset_x
    tip_y = pos_w[:, 1] + s * self.cfg.tip_offset_x
    origins = self._env.scene.env_origins
    tip = torch.stack([tip_x - origins[:, 0], tip_y - origins[:, 1]], dim=-1)
    return tip, yaw, c, s

  def _gather_window(self, start: torch.Tensor, length: int, lo: int = 0):
    """Path points at indices start+lo .. start+lo+length-1, clamped to each
    env's valid range. Returns (points (E, L, 2), indices (E, L))."""
    offs = torch.arange(lo, lo + length, device=self.device)
    max_i = (self._path_len[self._env_path] - 1).unsqueeze(1)
    idx = torch.clamp(start.unsqueeze(1) + offs, min=0)
    idx = torch.minimum(idx, max_i)
    pts = self._paths[self._env_path.unsqueeze(1).expand_as(idx), idx]
    return pts, idx

  # -- CommandTerm API ---------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    dev = self.device
    pid = torch.randint(len(self._path_len), (n,), device=dev)
    self._env_path[env_ids] = pid
    plen = self._path_len[pid]

    p0 = self._paths[pid, 0]
    head = max(2, int(0.01 / self.cfg.path_spacing))  # ~10mm ahead for heading
    p1 = self._paths[pid, torch.clamp(torch.full_like(plen, head), max=plen - 1)]
    yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
    yaw = yaw + (torch.rand(n, device=dev) * 2 - 1) * self.cfg.init_yaw_noise

    ang = torch.rand(n, device=dev) * 2 * math.pi
    r = self.cfg.init_xy_noise * torch.sqrt(torch.rand(n, device=dev))
    tip = p0 + torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=-1)

    c, s = torch.cos(yaw), torch.sin(yaw)
    chassis = tip - self.cfg.tip_offset_x * torch.stack([c, s], dim=-1)
    origins = self._env.scene.env_origins[env_ids]
    pos_w = torch.stack(
      [
        chassis[:, 0] + origins[:, 0],
        chassis[:, 1] + origins[:, 1],
        torch.full((n,), 0.02, device=dev),
      ],
      dim=-1,
    )
    quat = torch.stack(
      [
        torch.cos(yaw / 2),
        torch.zeros(n, device=dev),
        torch.zeros(n, device=dev),
        torch.sin(yaw / 2),
      ],
      dim=-1,
    )
    self.robot.write_root_link_pose_to_sim(
      torch.cat([pos_w, quat], dim=-1), env_ids=env_ids
    )
    self.robot.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=dev), env_ids=env_ids
    )
    n_joints = len(self.robot.data.joint_pos[0])
    self.robot.write_joint_state_to_sim(
      torch.zeros(n, n_joints, device=dev),
      torch.zeros(n, n_joints, device=dev),
      env_ids=env_ids,
    )

    self._idx[env_ids] = 0
    self._prev_idx[env_ids] = 0
    self.err_mm[env_ids] = 0.0
    self.progress_mm[env_ids] = 0.0
    self.gate[env_ids] = 0.0
    self.at_end[env_ids] = False
    self.finished[env_ids] = False

  def _update_command(self) -> None:
    cfg = self.cfg
    tip, yaw, c, s = self._tip_and_yaw()
    plen = self._path_len[self._env_path]
    last = self._paths[self._env_path, plen - 1]

    # Advance the follower to the nearest point within the search window.
    pts, _ = self._gather_window(self._idx, cfg.search_window)
    d = torch.linalg.norm(pts - tip.unsqueeze(1), dim=-1)
    self._idx = torch.minimum(self._idx + d.argmin(dim=1), plen - 1)

    # Lookahead target: first windowed point at least `lookahead` away.
    pts2, _ = self._gather_window(self._idx, cfg.lookahead_window)
    d2 = torch.linalg.norm(pts2 - tip.unsqueeze(1), dim=-1)
    beyond = d2 >= cfg.lookahead
    has_target = beyond.any(dim=1)
    first = beyond.int().argmax(dim=1)
    target = pts2[torch.arange(self.num_envs, device=self.device), first]
    target = torch.where(has_target.unsqueeze(1), target, last)
    self.at_end = ~has_target

    self.dist_final = torch.linalg.norm(last - tip, dim=-1)

    # Local-window tracking error (same rationale as rl/signature_env.py:
    # measure against the tip's OWN stretch of path, not a nearby fold).
    w = cfg.local_err_window
    pts3, _ = self._gather_window(self._idx, 2 * w + 1, lo=-w)
    d3 = torch.linalg.norm(pts3 - tip.unsqueeze(1), dim=-1)
    self.err_mm = d3.min(dim=1).values * 1000.0

    self.progress_mm = (self._idx - self._prev_idx).float() * cfg.path_spacing * 1000.0
    self._prev_idx = self._idx.clone()
    self.gate = torch.exp(-((self.err_mm / cfg.err_gate_mm) ** 2))
    self.finished = (self.dist_final < cfg.finish_tol) & self.at_end

    dvec = target - tip
    dx_local = c * dvec[:, 0] + s * dvec[:, 1]
    dy_local = -s * dvec[:, 0] + c * dvec[:, 1]
    self._command = torch.stack(
      [
        dx_local / _OBS_SCALE[0],
        dy_local / _OBS_SCALE[1],
        self.dist_final / _OBS_SCALE[2],
        self.at_end.float(),
      ],
      dim=-1,
    )

  def _update_metrics(self) -> None:
    self.metrics["tracking_err_mm"][:] = self.err_mm

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw the selected envs' target paths (blue dots) and current lookahead
    targets (red)."""
    if not self._debug_vis_ok:
      return
    try:
      env_indices = visualizer.get_env_indices(self.num_envs)
      if not env_indices:
        return
      origins = self._env.scene.env_origins.cpu().numpy()
      for i in list(env_indices)[:4]:
        pid = int(self._env_path[i])
        length = int(self._path_len[pid])
        stride = max(1, length // 30)
        pts = self._paths[pid, :length:stride].cpu().numpy()
        o = origins[i]
        for p in pts:
          visualizer.add_sphere(
            (float(p[0] + o[0]), float(p[1] + o[1]), 0.002),
            0.0015,
            (0.25, 0.45, 0.95, 1.0),
          )
    except Exception:
      self._debug_vis_ok = False  # never let visualization break the viewer


@dataclass(kw_only=True)
class SignaturePathCommandCfg(CommandTermCfg):
  entity_name: str = "lego_car"
  paths: tuple = ()
  """Tuple of (P_i, 2) float arrays: env-local (paper-frame) waypoints in
  meters, arc-length resampled at `path_spacing` (track_trajectory.load_path_world)."""
  lookahead: float = 0.006
  path_spacing: float = 0.002
  search_window: int = 40
  lookahead_window: int = 64
  local_err_window: int = 25
  err_gate_mm: float = 3.0
  finish_tol: float = 0.003
  init_xy_noise: float = 0.010
  init_yaw_noise: float = math.radians(15.0)
  tip_offset_x: float = -0.073
  # Never resample mid-episode; a new path arrives on env reset.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: "ManagerBasedRlEnv") -> SignaturePathCommand:
    return SignaturePathCommand(self, env)


# -- reward / termination terms -------------------------------------------------
# NOTE weights: mjlab's RewardManager computes weight * raw * step_dt. See
# signature_env_cfg.py for the mapping to rl/signature_env.py's per-step
# weights.


def gated_progress_rate(
  env: "ManagerBasedRlEnv", command_name: str = "signature_path"
) -> torch.Tensor:
  """Accuracy-gated path progress, in mm/s (a rate, so that weight * rate *
  dt reproduces the SB3 env's `w_progress * progress_mm * gate` per step)."""
  term = env.command_manager.get_term(command_name)
  return term.gate * term.progress_mm / env.step_dt


def tracking_error_sq(
  env: "ManagerBasedRlEnv", command_name: str = "signature_path"
) -> torch.Tensor:
  """Squared local tracking error, mm^2. Use with a negative weight."""
  term = env.command_manager.get_term(command_name)
  return term.err_mm.square()


def termination_flag(env: "ManagerBasedRlEnv", term_key: str) -> torch.Tensor:
  """1.0 on the step a given termination fires (for one-time bonuses)."""
  return env.termination_manager.get_term(term_key).float()


def signature_finished(
  env: "ManagerBasedRlEnv", command_name: str = "signature_path"
) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return term.finished


def off_path(
  env: "ManagerBasedRlEnv",
  command_name: str = "signature_path",
  limit_mm: float = 20.0,
) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return term.err_mm > limit_mm
