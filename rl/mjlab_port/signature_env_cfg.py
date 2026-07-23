"""mjlab task config for signature tracing: the LEGO car entity plus the
SignaturePathCommand MDP (signature_mdp.py).

Reward-weight mapping: mjlab's RewardManager computes weight * raw * step_dt
(step_dt = 0.002 * 10 = 0.02 s), while rl/signature_env.py applies weights
per control step. The weights below reproduce the SB3 Run-3 reward exactly:

  SB3 per step                      -> mjlab (raw value, weight)
  +2.0  * progress_mm * gate           (gate*progress_mm/dt,  +2.0)
  -0.02 * err_mm^2                     (err_mm^2,             -1.0)
  -0.05 * ||a - a_prev||^2             (action_rate_l2,       -2.5)
  -0.05 (time penalty)                 (is_alive=1,           -2.5)
  +30 on finish                        (termination_flag,     +1500)
  -30 on off-path                      (termination_flag,     -1500)
"""

from __future__ import annotations

import glob
import os
import sys

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  generated_commands,
  is_alive,
  joint_vel_rel,
  last_action,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.utils.noise import GaussianNoiseCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

import signature_mdp
from lego_car_env_cfg import (
  _CAR_CFG,
  _WHEELS_CFG,
  _get_lego_car_cfg,
  lego_car_ppo_runner_cfg,
)

MJLAB_PORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(MJLAB_PORT_DIR))
if PROJECT_DIR not in sys.path:
  sys.path.insert(0, PROJECT_DIR)

import track_trajectory as tt  # noqa: E402  (needs PROJECT_DIR on sys.path)


def load_signature_paths() -> tuple:
  """All recorded signatures, world/paper frame meters, arc-length resampled -
  the exact same preprocessing the SB3 pipeline uses. If the LEGOCAR_TRAJECTORY
  environment variable is set (see play_car.py --trajectory), only that one
  file is used - every env then traces the same signature."""
  import trajectory_io as tio
  files = tio.find_trajectory_files(PROJECT_DIR)
  only = os.environ.get("LEGOCAR_TRAJECTORY")
  if only:
    files = [f for f in files if os.path.basename(f) == os.path.basename(only)]
    if not files:
      raise FileNotFoundError(f"LEGOCAR_TRAJECTORY={only} not found in {PROJECT_DIR}")
  if not files:
    raise FileNotFoundError(f"No trajectory .npz files found in {PROJECT_DIR}")
  return tuple(tt.load_path_world(f) for f in files)


# Domain randomization toggle: set LEGOCAR_DR=0 to train a clean (no-noise)
# baseline for A/B comparison; default on. Observation noise models the real
# closed-loop sensing gap (camera tip / IMU / encoder) that made BC wobble on
# hardware. Normally applied ONLY in training (the actor group's
# enable_corruption=not play), so play/eval sees clean observations.
DR = os.environ.get("LEGOCAR_DR", "1") != "0"

# Force observation noise ON even in play/eval mode, so a trained policy can be
# evaluated UNDER noise (the nominal-vs-DR robustness test). Default off. Needs
# DR on too (default) so the noise cfgs are actually attached to the terms.
EVAL_NOISE = os.environ.get("LEGOCAR_EVAL_NOISE", "0") != "0"


def _noise(std: float):
  """Additive Gaussian observation-noise cfg, or None when DR is disabled."""
  return GaussianNoiseCfg(operation="add", mean=0.0, std=std) if DR else None


def lego_car_signature_env_cfg(
  play: bool = False, num_envs: int = 16
) -> ManagerBasedRlEnvCfg:
  command_cfg = signature_mdp.SignaturePathCommandCfg(
    paths=load_signature_paths(),
    debug_vis=True,
  )
  if play:
    # Nominal starts in play mode, like evaluate/quick-eval in the SB3 stack.
    command_cfg.init_xy_noise = 0.0
    command_cfg.init_yaw_noise = 0.0

  # Noise stds are in each term's own units and applied only in training.
  # Tune via the constants; see LEGOCAR_DR to turn the whole set off.
  actor_terms = {
    "signature": ObservationTermCfg(
      func=generated_commands, params={"command_name": "signature_path"},
      noise=_noise(0.05),  # scaled tip features ~= camera position noise
    ),
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel, params={"asset_cfg": _CAR_CFG}, noise=_noise(0.005)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, params={"asset_cfg": _CAR_CFG}, noise=_noise(0.02)  # ~IMU gyro
    ),
    "wheel_vel": ObservationTermCfg(
      func=joint_vel_rel, params={"asset_cfg": _WHEELS_CFG}, noise=_noise(0.1)  # ~encoder
    ),
    "last_action": ObservationTermCfg(func=last_action),  # policy's own action, exact
  }
  if not play:
    print(f"[signature_env_cfg] observation-noise DR: {'ON' if DR else 'OFF'}")
  elif EVAL_NOISE:
    print(f"[signature_env_cfg] EVAL under observation noise: {'ON' if DR else 'cfgs OFF (set LEGOCAR_DR=1)'}")

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=(not play) or EVAL_NOISE),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "wheel_effort": JointEffortActionCfg(
      entity_name="lego_car",
      actuator_names=("joint_left", "joint_right"),
      scale=1.0,
    ),
  }

  rewards = {
    "gated_progress": RewardTermCfg(
      func=signature_mdp.gated_progress_rate, weight=2.0, params={}
    ),
    "tracking_error": RewardTermCfg(
      func=signature_mdp.tracking_error_sq, weight=-1.0, params={}
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-2.5, params={}),
    "time_penalty": RewardTermCfg(func=is_alive, weight=-2.5, params={}),
    "finish_bonus": RewardTermCfg(
      func=signature_mdp.termination_flag,
      weight=1500.0,
      params={"term_key": "finished"},
    ),
    "off_path_penalty": RewardTermCfg(
      func=signature_mdp.termination_flag,
      weight=-1500.0,
      params={"term_key": "off_path"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "finished": TerminationTermCfg(func=signature_mdp.signature_finished),
    "off_path": TerminationTermCfg(
      func=signature_mdp.off_path, params={"limit_mm": 20.0}
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"lego_car": _get_lego_car_cfg()},
      num_envs=num_envs,
      env_spacing=0.35,
    ),
    observations=observations,
    actions=actions,
    commands={"signature_path": command_cfg},
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="lego_car",
      body_name="chassis",
      distance=0.6,
      elevation=-25.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=10,  # 50 Hz control, matching rl/signature_env.py
    episode_length_s=45.0,
  )
  if play:
    cfg.episode_length_s = 1e10
  return cfg


def lego_car_signature_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = lego_car_ppo_runner_cfg()
  cfg.experiment_name = "lego_car_signature"
  cfg.logger = "tensorboard"  # no W&B login required
  cfg.num_steps_per_env = 48
  cfg.save_interval = 25
  cfg.max_iterations = 300
  return cfg
