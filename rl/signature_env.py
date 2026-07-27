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
    (default 50 -> 10 Hz control at the model's 2 ms timestep), matching the
    real robot's 10 Hz command cadence (drive_closed_loop.py CONTROL_DT=0.1)
    so the policy is trained at the frequency it is deployed at. The inner
    wheel-velocity PI still runs every physics step (500 Hz), standing in for
    the SPIKE hub's fast internal speed loop. NOTE: the per-step time/track/
    action-rate penalties are frequency-dependent (they accumulate once per
    control step); their default weights below are scaled for the 10 Hz default
    so the *per-second* objective matches the earlier 50 Hz tuning - if you
    change frame_skip, rescale them the same way (see the reward weights);
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
from collections import deque

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
    "vel_lag": 0.40,        # +-40% on the wheel speed-loop lag/dead time: the
                            # measured ~480ms comes from the firmware's reported
                            # speed, which may be filtered, so the true lag is
                            # uncertain - randomize it so the policy is robust to
                            # whatever it turns out to be, rather than tuned to one
                            # guess (only applied when vel_lag_tau > 0).
    "obs_delay": 0.50,      # +-50% (rounded to whole control steps) on the
                            # camera/BLE observation delay: it is even less well
                            # known than the wheel lag, so randomize it widely
                            # (only applied when obs_delay_steps > 0).
}


class SignatureEnv(gym.Env):
    """One episode = trace one signature. `path_worlds` is a list of
    world-frame (N, 2) paths (from track_trajectory.load_path_world); each
    reset picks one at random."""

    metadata = {"render_modes": []}

    def __init__(self, path_worlds, frame_skip: int = 50,
                 lookahead: float = tt.DEFAULT_LOOKAHEAD,
                 finish_tol: float = tt.DEFAULT_FINISH_TOL,
                 path_spacing: float = tt.DEFAULT_PATH_SPACING,
                 max_time: float = 60.0,
                 init_xy_noise: float = 0.010,
                 init_yaw_noise: float = np.radians(15.0),
                 domain_rand: bool = False, dr_scales: dict = None,
                 w_progress: float = 2.0, w_track: float = 0.10,
                 err_gate_mm: float = 3.0,
                 w_action_rate: float = 0.01, w_time: float = 0.25,
                 completion_bonus: float = 30.0, off_path_penalty: float = 30.0,
                 off_path_limit_mm: float = 20.0, obs_noise_std: float = 0.0,
                 vel_lag_tau: float = 0.0, vel_dead_time: float = 0.0,
                 v_max: float = V_MAX, omega_max: float = OMEGA_MAX,
                 obs_delay_steps: int = 0):
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
        self.vel_lag_tau = float(vel_lag_tau)
        self.vel_dead_time = float(vel_dead_time)
        # Speed ceiling this policy is trained against. The module default
        # V_MAX=0.06 is 2x the expert's 0.03 m/s; hardware could only realize
        # ~0.03 (see rl/TRAINING_LOG.md hardware section), so capping here lets
        # the policy spend its whole action range inside the achievable envelope
        # instead of relying on --policy-speed-scale at deploy time.
        # DEPLOYMENT MUST USE THE SAME VALUE - it is recorded in the run config
        # json and read back by evaluate_rl.py / drive_closed_loop.py.
        self.v_max = float(v_max)
        # Angular ceiling. Cap this WITH v_max, never alone: capping v_max at
        # 0.035 while OMEGA_MAX stayed 10.0 doubled the omega:v ratio available
        # during training, and the resulting policy oscillated on hardware
        # (7.0 mm RMS, off-path abort). Damping omega afterwards with
        # --policy-omega-scale recovers most of it (2.3 mm) but leaves a ~1.5 mm
        # inward bias, because the policy under-turns relative to what it asked
        # for. Capping here instead lets it learn the curvature it can achieve.
        self.omega_max = float(omega_max)
        # Observation delay in CONTROL steps: the real loop sees the world
        # through camera exposure + blob detection + BLE, none of which the sim
        # had. At 10 Hz one step is already 100 ms, the same order as the wheel
        # lag. Applied to the scaled obs the policy consumes.
        self.obs_delay_steps = int(obs_delay_steps)
        self._obs_buf = None

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

        # Per-episode wheel-loop lag: randomized around the nominal when domain
        # randomization is on (see DEFAULT_DR_SCALES["vel_lag"]), so the policy
        # sees a spread of plausible real lags instead of one point estimate.
        lag_tau, lag_dead = self.vel_lag_tau, self.vel_dead_time
        if self.domain_rand and self.vel_lag_tau > 0.0:
            f = self.dr_scales.get("vel_lag", 0.0)
            lag_tau *= float(self.np_random.uniform(1.0 - f, 1.0 + f))
            lag_dead *= float(self.np_random.uniform(1.0 - f, 1.0 + f))

        self.tracker = tt.SignatureTracker(
            self.path_world, lookahead=self.lookahead, finish_tol=self.finish_tol,
            path_spacing=self.path_spacing, controller=controller,
            init_xy_noise=self.init_xy_noise, init_yaw_noise=self.init_yaw_noise,
            vel_lag_tau=lag_tau, vel_dead_time=lag_dead,
            seed=int(self.np_random.integers(2 ** 31)))

        if self.domain_rand:
            self._randomize_model()

        if self._max_episode_steps is None:
            dt = self.tracker.m.opt.timestep
            self._max_episode_steps = int(round(self.max_time / (dt * self.frame_skip)))

        self._elapsed = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        # Per-episode observation delay, randomized like the wheel lag above.
        self._ep_obs_delay = self.obs_delay_steps
        if self.domain_rand and self.obs_delay_steps > 0:
            f = self.dr_scales.get("obs_delay", 0.0)
            self._ep_obs_delay = int(round(
                self.obs_delay_steps * float(self.np_random.uniform(1.0 - f, 1.0 + f))))
        self._obs_buf = None          # refilled on the first _scaled_obs() call
        obs = self._initial_observation()
        self._prev_path_idx = self.tracker.follower.idx
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        self._cmd = (float(action[0] * self.v_max), float(action[1] * self.omega_max))

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
        # Progress telescopes to the path length regardless of control rate, so
        # w_progress is frequency-invariant. The other three accumulate ONCE per
        # control step, so their episode totals scale with frame_skip - their
        # defaults are set for the 10 Hz default (see __init__ docstring).
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
        obs = obs.astype(np.float32)

        # Observation delay: hand the policy the state from N control steps ago.
        # Modelled AFTER the noise so each delivered frame carries the noise it
        # was captured with, as a real delayed measurement would.
        delay = getattr(self, "_ep_obs_delay", self.obs_delay_steps)
        if delay > 0:
            n = delay + 1
            if self._obs_buf is None or self._obs_buf.maxlen != n:
                self._obs_buf = deque([obs.copy()] * n, maxlen=n)
            self._obs_buf.append(obs.copy())
            obs = self._obs_buf[0]      # captured obs_delay_steps ticks ago
        return obs

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


def scales_from_config(model_path, v_default=V_MAX, om_default=OMEGA_MAX):
    """(v_max, omega_max) a policy was trained with, from its run config.

    Both must match at deployment: they set the action scaling, so a mismatch
    silently rescales everything the policy commands."""
    cfg = _find_config(model_path)
    if cfg is None:
        return float(v_default), float(om_default)
    import json
    with open(cfg) as f:
        d = json.load(f)
    return float(d.get("v_max", v_default)), float(d.get("omega_max", om_default))


def _find_config(model_path):
    """Config json for a model, falling back to the parent run for _best.zip."""
    stem = os.path.splitext(model_path)[0]
    cands = [stem + "_config.json"]
    if stem.endswith("_best"):
        cands.append(stem[:-len("_best")] + "_config.json")
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def v_max_from_config(model_path, default=V_MAX):
    """Read the v_max a policy was trained with from its sibling _config.json.

    Deploying a policy with the wrong V_MAX silently rescales every speed it
    commands (a 0.035-trained policy run at 0.06 moves 1.7x too fast), so both
    evaluate_rl.py and drive_closed_loop.py resolve it from the run config
    rather than the module constant."""
    import json
    stem = os.path.splitext(model_path)[0]
    # "<name>_best.zip" is a checkpoint of the "<name>" run and shares its
    # config, so fall back to the parent stem rather than silently returning
    # the module default (which would run a capped policy far too fast).
    candidates = [stem + "_config.json"]
    if stem.endswith("_best"):
        candidates.append(stem[:-len("_best")] + "_config.json")
    for cfg in candidates:
        if os.path.exists(cfg):
            with open(cfg) as f:
                return float(json.load(f).get("v_max", default))
    return float(default)


def make_sb3_controller(model, v_max: float = V_MAX,
                        omega_max: float = OMEGA_MAX):
    """Wraps a trained SB3 policy as a SignatureTracker `controller(raw_obs)
    -> (v, omega)` callable (the same hook learning/evaluate_bc.py uses), so
    evaluation and deployment reuse the exact training-time obs/action scaling."""
    def controller(raw_obs: np.ndarray):
        obs = (raw_obs / OBS_SCALE).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        return float(action[0] * v_max), float(action[1] * omega_max)
    return controller
