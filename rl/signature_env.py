"""
signature_env.py - Gymnasium environment wrapping track_trajectory.SignatureTracker
for reinforcement learning (see rl/README.md for the full MDP formulation).

The physics, observation features, and (v, omega) action interface are exactly
the ones the pure-pursuit expert and the BC policy use - the RL policy plugs
into SignatureTracker's `controller` hook, so anything trained here can be
evaluated and deployed through the same code paths as learning/evaluate_bc.py.

What this file adds on top of the tracker:
  - action scaling: the policy acts in [-1, 1]^2, mapped to (v, omega) via
    ACTION_SCALE (so PPO's Gaussian exploration is well-conditioned);
  - fixed observation scaling (OBS_SCALE): deterministic per-feature scaling
    instead of running statistics, so deployment sees exactly the training
    scaling with no VecNormalize state to carry around;
  - frame skip: one policy action is held for `frame_skip` physics steps
    (default 10 -> 50 Hz control at the model's 2 ms timestep), keeping
    episodes a manageable few hundred steps;
  - reward: accuracy-GATED arc-length progress (progress earns nothing
    unless the tip is within ~err_gate_mm of its local stretch of path -
    this is what makes the objective speed-invariant; per-step error
    penalties alone are exploitable by sprinting, see rl/TRAINING_LOG.md
    runs 1-2), minus a quadratic tracking penalty (recovery gradient),
    action-rate penalty and time penalty, plus a completion bonus /
    off-path failure penalty. The default weights make accurate tracing
    NET-POSITIVE per step; keep it that way when retuning - if staying on
    the path pays worse than the -off_path_penalty, the optimal policy is
    to dive off the path immediately to end the episode cheaply;
  - episode logic: reset onto a randomly chosen signature with randomized
    initial pose (the tracker's own init_xy_noise/init_yaw_noise), terminate
    on completion or straying off the path, truncate at max_time;
  - optional domain randomization of the model's hand-tuned physical
    parameters (mass, friction, motor gear, wheel damping) at each reset.
    The tracker's PI inner loop keeps the *nominal* gear values it read at
    construction, so gear randomization deliberately shows the policy a
    motor-strength model mismatch, like a real motor would.
"""

import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

RL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RL_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import track_trajectory as tt

# Policy action a in [-1, 1]^2 maps to v = a[0] * V_MAX, omega = a[1] * OMEGA_MAX.
# V_MAX gives 2x headroom over the expert's 0.03 m/s nominal speed; OMEGA_MAX
# matches the expert's own omega clip in SignatureTracker._expert_action.
V_MAX = 0.06
OMEGA_MAX = 10.0
ACTION_SCALE = np.array([V_MAX, OMEGA_MAX], dtype=np.float32)

# Fixed per-feature scaling for the raw tracker observation
# [dx_local (m), dy_local (m), dist_to_final (m), at_end_flag]: dx/dy are of
# lookahead magnitude (~6 mm), dist_to_final spans the sheet (~0.1-0.3 m).
OBS_SCALE = np.array([0.01, 0.01, 0.1, 1.0], dtype=np.float32)

# Fractional half-ranges for uniform domain randomization (value *= U[1-f, 1+f]).
DEFAULT_DR_SCALES = {
    "chassis_mass": 0.20,
    "friction": 0.30,       # sliding friction of wheels and paper
    "gear": 0.20,           # actuator torque scale (seen as model mismatch by the PI loop)
    "wheel_damping": 0.30,
}


class SignatureEnv(gym.Env):
    """One episode = trace one signature. `path_worlds` is a list of
    world-frame (N, 2) paths (from track_trajectory.load_path_world); each
    reset picks one at random."""

    metadata = {"render_modes": []}

    def __init__(self, path_worlds, frame_skip: int = 10,
                 lookahead: float = tt.DEFAULT_LOOKAHEAD,
                 finish_tol: float = tt.DEFAULT_FINISH_TOL,
                 path_spacing: float = tt.DEFAULT_PATH_SPACING,
                 max_time: float = 60.0,
                 init_xy_noise: float = 0.010,
                 init_yaw_noise: float = np.radians(15.0),
                 domain_rand: bool = False, dr_scales: dict = None,
                 w_progress: float = 2.0, w_track: float = 0.02,
                 err_gate_mm: float = 3.0,
                 w_action_rate: float = 0.05, w_time: float = 0.05,
                 completion_bonus: float = 30.0, off_path_penalty: float = 30.0,
                 off_path_limit_mm: float = 20.0, obs_noise_std: float = 0.0):
        super().__init__()
        if not path_worlds:
            raise ValueError("path_worlds must contain at least one path")
        self.path_worlds = [np.asarray(p, dtype=np.float64) for p in path_worlds]
        self.frame_skip = int(frame_skip)
        self.lookahead = lookahead
        self.finish_tol = finish_tol
        self.path_spacing = path_spacing
        self.max_time = max_time
        self.init_xy_noise = init_xy_noise
        self.init_yaw_noise = init_yaw_noise
        self.domain_rand = domain_rand
        self.dr_scales = dict(DEFAULT_DR_SCALES if dr_scales is None else dr_scales)
        self.w_progress = w_progress
        self.w_track = w_track
        self.err_gate_mm = err_gate_mm
        self.w_action_rate = w_action_rate
        self.w_time = w_time
        self.completion_bonus = completion_bonus
        self.off_path_penalty = off_path_penalty
        self.off_path_limit_mm = off_path_limit_mm
        self.obs_noise_std = float(obs_noise_std)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.tracker = None
        self.path_world = None
        self._cmd = (0.0, 0.0)
        self._raw_obs = None
        self._max_episode_steps = None

    # -- episode lifecycle ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.path_world = self.path_worlds[int(self.np_random.integers(len(self.path_worlds)))]

        self._cmd = (0.0, 0.0)

        def controller(raw_obs, _env=self):
            _env._raw_obs = raw_obs
            return _env._cmd

        self.tracker = tt.SignatureTracker(
            self.path_world, lookahead=self.lookahead, finish_tol=self.finish_tol,
            path_spacing=self.path_spacing, controller=controller,
            init_xy_noise=self.init_xy_noise, init_yaw_noise=self.init_yaw_noise,
            seed=int(self.np_random.integers(2 ** 31)))

        if self.domain_rand:
            self._randomize_model()

        if self._max_episode_steps is None:
            dt = self.tracker.m.opt.timestep
            self._max_episode_steps = int(round(self.max_time / (dt * self.frame_skip)))

        self._elapsed = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        obs = self._initial_observation()
        self._prev_path_idx = self.tracker.follower.idx
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        self._cmd = (float(action[0] * V_MAX), float(action[1] * OMEGA_MAX))

        finished = False
        for _ in range(self.frame_skip):
            finished = self.tracker.step()
            if finished:
                break
        self._elapsed += 1

        tip = self.tracker.d.site_xpos[self.tracker.site_id][:2]
        idx = self.tracker.follower.idx
        # Error to the LOCAL stretch of path around the follower's index (+-50mm
        # of arc), not the global nearest point: where the signature folds back
        # near itself (or has pen-lift jump segments), the globally-nearest
        # point can belong to a different branch, which under-reports how far
        # the tip has strayed from the part it is supposed to be tracing.
        lo = max(0, idx - 25)
        hi = min(len(self.path_world), idx + 25)
        err_mm = float(np.min(np.linalg.norm(self.path_world[lo:hi] - tip, axis=1))) * 1000.0
        progress_mm = (idx - self._prev_path_idx) * self.path_spacing * 1000.0
        self._prev_path_idx = idx

        # Progress only counts when the tip is actually on the path: the
        # accuracy gate makes the progress reward speed-invariant. (A plain
        # per-step error penalty is NOT: its episode total shrinks the faster
        # the car goes, so run 2 learned to sprint sloppily - see
        # rl/TRAINING_LOG.md.)
        accuracy_gate = float(np.exp(-(err_mm / self.err_gate_mm) ** 2))
        reward = (self.w_progress * progress_mm * accuracy_gate
                  - self.w_track * err_mm ** 2
                  - self.w_action_rate * float(np.sum((action - self._prev_action) ** 2))
                  - self.w_time)
        self._prev_action = action

        terminated = False
        if finished:
            reward += self.completion_bonus
            terminated = True
        elif err_mm > self.off_path_limit_mm:
            reward -= self.off_path_penalty
            terminated = True
        truncated = (not terminated) and self._elapsed >= self._max_episode_steps

        info = {"err_mm": err_mm}
        if terminated or truncated:
            info["is_success"] = bool(finished)
        return self._scaled_obs(), float(reward), terminated, truncated, info

    # -- helpers ---------------------------------------------------------------

    def _scaled_obs(self) -> np.ndarray:
        obs = self._raw_obs / OBS_SCALE
        if self.obs_noise_std > 0.0:
            # Additive Gaussian sensor noise on the scaled obs (camera tip / IMU /
            # encoder), applied only in training envs (deployment reads the real
            # sensors). Makes the policy robust to the closed-loop sensing gap
            # that made BC wobble on hardware.
            obs = obs + self.np_random.normal(0.0, self.obs_noise_std, size=obs.shape)
        return obs.astype(np.float32)

    def _initial_observation(self) -> np.ndarray:
        """Builds the pre-first-action observation without stepping physics,
        using the same feature computation the tracker runs each step."""
        tr = self.tracker
        tip = tr.d.site_xpos[tr.site_id][:2].copy()
        yaw = tt.yaw_from_quat(tr.d.qpos[tr.chassis_qpos_adr + 3:tr.chassis_qpos_adr + 7])
        target, at_end = tr.follower.get_target(tip)
        dist_final = float(np.linalg.norm(self.path_world[-1] - tip))
        self._raw_obs = tr._build_observation(tip, yaw, target, at_end, dist_final)
        return self._scaled_obs()

    def _randomize_model(self) -> None:
        """Jitters the XML's hand-tuned physical parameters on the freshly
        loaded model (each reset starts from pristine nominal values), then
        re-settles briefly so contacts adjust to the new parameters."""
        m, d = self.tracker.m, self.tracker.d

        def u(frac: float) -> float:
            return float(self.np_random.uniform(1.0 - frac, 1.0 + frac))

        m.body_mass[m.body("chassis").id] *= u(self.dr_scales["chassis_mass"])
        m.geom_friction[m.geom("paper").id, 0] *= u(self.dr_scales["friction"])
        for body_name in ("wheel_left", "wheel_right"):
            gid = int(m.body(body_name).geomadr[0])
            m.geom_friction[gid, 0] *= u(self.dr_scales["friction"])
        for i in range(m.nu):
            m.actuator_gear[i, 0] *= u(self.dr_scales["gear"])
        for joint_name in ("joint_left", "joint_right"):
            m.dof_damping[m.joint(joint_name).dofadr[0]] *= u(self.dr_scales["wheel_damping"])

        mujoco.mj_setConst(m, d)
        d.ctrl[:] = 0.0
        for _ in range(50):
            mujoco.mj_step(m, d)


def make_sb3_controller(model):
    """Wraps a trained SB3 policy as a SignatureTracker `controller(raw_obs)
    -> (v, omega)` callable (the same hook learning/evaluate_bc.py uses), so
    evaluation and deployment reuse the exact training-time obs/action scaling."""
    def controller(raw_obs: np.ndarray):
        obs = (raw_obs / OBS_SCALE).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        return float(action[0] * V_MAX), float(action[1] * OMEGA_MAX)
    return controller
