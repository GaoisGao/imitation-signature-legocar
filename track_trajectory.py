"""
track_trajectory.py - Step 4: drive the simulated LEGO car so its pencil tip
follows a recorded signature trajectory, using a classical (non-learning)
pure-pursuit path-tracking controller.

Pipeline:
  1. Load a trajectory .npz (from webapp.py's target_trajectory_*.npz, or
     coordinate_plane.py's trajectory_*_paper.npz).
  2. Convert paper mm coordinates to the MuJoCo world frame (meters), using
     the same 199 x 137 mm sheet convention as coordinate_plane.py; the
     paper geom in lego_car_with_pencil.xml is centered at the world origin.
  3. Smooth the path and resample it to a uniform arc-length spacing.
  4. Reset the car so the pencil tip starts at the first path point, facing
     the initial path direction.
  5. Each simulation step: read the pencil tip's world position (from the
     'pencil_trace' site) and the chassis heading, run a pure-pursuit
     steering law to get a desired linear/angular velocity, convert that to
     left/right wheel angular velocities, and drive the torque motors with a
     PI velocity controller to track those wheel speeds.
  6. After the run, compare the traced tip path to the target path and
     report/plot the tracking error.

The core simulation is the SignatureTracker class below (load once, then call
.step() repeatedly) so other programs (e.g. webapp.py, to render it live in a
browser) can drive the exact same controller instead of reimplementing it.

Usage:
    py -3.13 track_trajectory.py
    py -3.13 track_trajectory.py --trajectory target_trajectory_20260708_161115.npz
    py -3.13 track_trajectory.py --speed 0.03 --lookahead 0.006 --view
"""

import argparse
import glob
import os

import mujoco
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lego_car_with_pencil.xml")

# Physical sheet size, must match coordinate_plane.py. The paper geom in the
# MuJoCo model is centered at the world origin, so paper-mm (0,0) (the ID0
# marker corner) maps to world (-WIDTH/2, -HEIGHT/2) meters.
PLANE_WIDTH_MM = 199.0
PLANE_HEIGHT_MM = 137.0

WHEEL_VEL_KP = 0.004   # N*m per (rad/s) of wheel speed error
WHEEL_VEL_KI = 0.002   # N*m per (rad/s * s) of accumulated error
CTRL_LIMIT = 20.0      # clip on the raw ctrl (torque / gear) value

DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_PATH_SPACING = 0.002    # m, arc-length spacing of the controller's path
DEFAULT_LOOKAHEAD = 0.006       # m, pure-pursuit lookahead distance
DEFAULT_SPEED = 0.03            # m/s, nominal forward speed
DEFAULT_FINISH_TOL = 0.003      # m, distance to the final point counted as "done"


# -- trajectory loading & conversion --------------------------------------

def find_latest_trajectory_file(folder: str):
    """Newest recorded trajectory. Searches <folder>/datasets/trajectories
    (the current layout) and <folder> itself (legacy root location)."""
    patterns = ("target_trajectory_*.npz", "robot_trace_*.npz", "trajectory_*_paper.npz")
    candidates = []
    for d in (os.path.join(folder, "datasets", "trajectories"), folder):
        for pat in patterns:
            candidates += glob.glob(os.path.join(d, pat))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_trajectory_mm(path: str):
    """Returns (xy_mm (N,2), t (N,)) from a target_trajectory_*.npz or a
    coordinate_plane.py trajectory_*_paper.npz."""
    data = np.load(path)
    if "target_trajectory" in data.files:
        traj = data["target_trajectory"]
    else:
        traj_keys = [k for k in data.files if k != "count"]
        if not traj_keys:
            raise ValueError(f"No trajectory data found in {path}")
        traj = max((data[k] for k in traj_keys), key=len)
    return traj[:, :2], traj[:, 2]


def mm_to_world(xy_mm: np.ndarray) -> np.ndarray:
    """Paper mm coords (origin at ID0 marker) -> MuJoCo world meters
    (origin at the paper's center, matching the XML's paper geom)."""
    offset = np.array([PLANE_WIDTH_MM / 2000.0, PLANE_HEIGHT_MM / 2000.0])
    return xy_mm / 1000.0 - offset


def smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing with edge-replicated padding."""
    if window <= 1 or len(xy) < window:
        return xy
    half = window // 2
    kernel = np.ones(window) / window
    padded_x = np.pad(xy[:, 0], (half, half), mode="edge")
    padded_y = np.pad(xy[:, 1], (half, half), mode="edge")
    sx = np.convolve(padded_x, kernel, mode="valid")[:len(xy)]
    sy = np.convolve(padded_y, kernel, mode="valid")[:len(xy)]
    return np.column_stack([sx, sy])


def resample_by_arclength(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline to uniformly spaced points, `spacing` meters apart
    (by arc length), keeping the first and last point."""
    deltas = np.diff(xy, axis=0)
    seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < spacing:
        return xy[[0, -1]]
    targets = np.arange(0.0, total, spacing)
    x = np.interp(targets, cum, xy[:, 0])
    y = np.interp(targets, cum, xy[:, 1])
    out = np.column_stack([x, y])
    return np.vstack([out, xy[-1]])


def load_path_world(trajectory_path: str, smooth_window: int = DEFAULT_SMOOTH_WINDOW,
                     path_spacing: float = DEFAULT_PATH_SPACING) -> np.ndarray:
    """Full load -> world-frame -> smooth -> resample pipeline used by both
    the CLI and webapp.py."""
    xy_mm, _ = load_trajectory_mm(trajectory_path)
    path_world = mm_to_world(xy_mm)
    path_world = smooth_xy(path_world, smooth_window)
    return resample_by_arclength(path_world, path_spacing)


# -- pose helpers -----------------------------------------------------------

def yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# -- pure-pursuit path follower ----------------------------------------------

class PathFollower:
    """Classical pure-pursuit lookahead controller (non-learning)."""

    def __init__(self, path_xy: np.ndarray, lookahead: float, search_window: int = 40):
        self.path = path_xy
        self.lookahead = lookahead
        self.search_window = search_window
        self.idx = 0

    def _advance_nearest(self, pos: np.ndarray) -> None:
        end = min(len(self.path), self.idx + self.search_window)
        seg = self.path[self.idx:end]
        dists = np.linalg.norm(seg - pos, axis=1)
        self.idx += int(np.argmin(dists))

    def get_target(self, pos: np.ndarray):
        """Returns (target_xy, at_end: bool)."""
        self._advance_nearest(pos)
        for j in range(self.idx, len(self.path)):
            if np.linalg.norm(self.path[j] - pos) >= self.lookahead:
                return self.path[j], False
        return self.path[-1], True


# -- MuJoCo model introspection ----------------------------------------------

def build_joint_to_actuator_map(m: mujoco.MjModel) -> dict:
    mapping = {}
    for i in range(m.nu):
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        mapping[joint_name] = i
    return mapping


def free_joint_qpos_dof_adr(m: mujoco.MjModel, body_name: str):
    body = m.body(body_name)
    joint_id = body.jntadr[0]
    assert m.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, \
        f"expected a free joint on body '{body_name}'"
    return int(m.jnt_qposadr[joint_id]), int(m.jnt_dofadr[joint_id])


class WheelVelocityPI:
    """Inner-loop PI controller: converts a target wheel angular velocity
    into a torque-actuator ctrl value (the motors are torque, not velocity,
    actuators)."""

    def __init__(self, gear: float):
        self.gear = gear
        self.integral = 0.0

    def step(self, target_omega: float, actual_omega: float, dt: float) -> float:
        error = target_omega - actual_omega
        self.integral += error * dt
        torque = WHEEL_VEL_KP * error + WHEEL_VEL_KI * self.integral
        ctrl = torque / self.gear
        return float(np.clip(ctrl, -CTRL_LIMIT, CTRL_LIMIT))


# -- the reusable tracker -----------------------------------------------------

class SignatureTracker:
    """
    Owns a MuJoCo model/data pair and drives the car's pencil tip along
    `path_world` (world-frame meters, arc-length resampled) using pure
    pursuit + exact offset feedback linearization + a PI wheel-velocity
    inner loop. Call .step() repeatedly; it returns True once finished.

    `controller`, if given, replaces the built-in pure-pursuit control law:
    it's called each step as `controller(obs) -> (v, omega)` with the same
    observation (see _build_observation) the expert computes its own action
    from, and its (v, omega) chassis-frame command is fed into the same
    wheel-velocity PI loop. This is the hook learning/evaluate_bc.py uses to
    drop a trained behavior-cloning policy into the exact physics loop the
    expert was recorded from. `record=True` logs every step's (observation,
    action) pair (learning/collect_expert_data.py's use case); it works with
    either the default expert or a custom controller.

    `init_xy_noise` (m) / `init_yaw_noise` (rad) perturb the starting pose:
    the pencil tip is placed uniformly within a disk of radius init_xy_noise
    around path_world[0], with the heading off by uniform(+-init_yaw_noise).
    This models imprecise manual placement of the real robot. Expert-recovery
    demonstrations (learning/collect_expert_data.py --episodes-per-traj) and
    robustness evaluation (learning/evaluate_bc.py --episodes) both use it;
    `seed` makes a perturbation reproducible so BC and the expert can be
    evaluated from identical starting poses (paired comparison).
    """

    def __init__(self, path_world: np.ndarray, model_path: str = MODEL_PATH,
                 speed: float = DEFAULT_SPEED, lookahead: float = DEFAULT_LOOKAHEAD,
                 finish_tol: float = DEFAULT_FINISH_TOL, path_spacing: float = DEFAULT_PATH_SPACING,
                 settle_steps: int = 300, controller=None, record: bool = False,
                 init_xy_noise: float = 0.0, init_yaw_noise: float = 0.0, seed=None):
        self.path_world = path_world
        self.speed = speed
        self.lookahead = lookahead
        self.finish_tol = finish_tol
        self.controller = controller
        self.record = record
        self.observations = []
        self.actions = []
        self.init_xy_noise = init_xy_noise
        self.init_yaw_noise = init_yaw_noise
        self._rng = np.random.default_rng(seed)

        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)

        self.site_id = self.m.site("pencil_trace").id
        self.chassis_qpos_adr, self.chassis_dof_adr = free_joint_qpos_dof_adr(self.m, "chassis")

        self.joint_left = self.m.joint("joint_left")
        self.joint_right = self.m.joint("joint_right")
        self.wheel_left_y = float(self.m.body("wheel_left").pos[1])
        self.wheel_right_y = float(self.m.body("wheel_right").pos[1])
        wheel_geom_id = int(self.m.body("wheel_left").geomadr[0])
        self.wheel_radius = float(self.m.geom_size[wheel_geom_id, 0])
        # tip = chassis + R(yaw) @ (tip_offset_x, 0); the tip trails behind the chassis
        self.tip_offset_x = float(self.m.body("pencil").pos[0])

        self.joint_to_actuator = build_joint_to_actuator_map(self.m)
        gear_left = float(self.m.actuator_gear[self.joint_to_actuator["joint_left"], 0])
        gear_right = float(self.m.actuator_gear[self.joint_to_actuator["joint_right"], 0])
        self.pi_left = WheelVelocityPI(gear_left)
        self.pi_right = WheelVelocityPI(gear_right)

        self.follower = PathFollower(path_world, lookahead)
        self.finished = False
        self.step_count = 0
        self.tip_history = []
        self.motor_history = []  # (t, yaw_deg, left_wheel_deg, right_wheel_deg), all relative to start

        self._reset_pose(path_spacing, settle_steps)

        # Baselines for motor_history, captured post-settle so small settling
        # motion doesn't bias the "relative to start" deltas.
        self._qpos_left0 = float(self.d.qpos[self.joint_left.qposadr[0]])
        self._qpos_right0 = float(self.d.qpos[self.joint_right.qposadr[0]])
        self._prev_yaw = yaw_from_quat(self.d.qpos[self.chassis_qpos_adr + 3:self.chassis_qpos_adr + 7])
        self._cum_yaw = 0.0

    def _reset_pose(self, path_spacing: float, settle_steps: int) -> None:
        """Places the car so the pencil tip starts at path_world[0], facing
        the initial path direction (both optionally perturbed by
        init_xy_noise/init_yaw_noise), then lets it settle under gravity."""
        lookahead_pts = max(2, int(0.01 / path_spacing))  # ~10mm ahead, for initial heading
        p0 = self.path_world[0].copy()
        p1 = self.path_world[min(lookahead_pts, len(self.path_world) - 1)]
        yaw0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

        if self.init_yaw_noise > 0.0:
            yaw0 += float(self._rng.uniform(-self.init_yaw_noise, self.init_yaw_noise))
        if self.init_xy_noise > 0.0:
            # uniform over a disk: sqrt of a uniform radius fraction keeps
            # density constant instead of clustering at the center
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            r = self.init_xy_noise * float(np.sqrt(self._rng.uniform()))
            p0 += r * np.array([np.cos(ang), np.sin(ang)])

        chassis_xy0 = p0 - np.array([self.tip_offset_x * np.cos(yaw0), self.tip_offset_x * np.sin(yaw0)])

        a = self.chassis_qpos_adr
        self.d.qpos[a:a + 2] = chassis_xy0
        self.d.qpos[a + 2] = 0.02
        self.d.qpos[a + 3:a + 7] = quat_from_yaw(yaw0)
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

        for _ in range(settle_steps):
            self.d.ctrl[:] = 0.0
            mujoco.mj_step(self.m, self.d)

    def _build_observation(self, tip_pos: np.ndarray, yaw: float, target: np.ndarray,
                            at_end: bool, dist_to_final: float) -> np.ndarray:
        """
        [dx_local, dy_local, dist_to_final, at_end_flag]: the pure-pursuit
        lookahead target expressed in the chassis's local frame (forward,
        left), plus the remaining distance to the end of the path and
        whether the lookahead has run off the end of it. This is a
        sufficient statistic for the expert's own action (see
        _expert_action) - a BC policy trained to reproduce (obs -> action)
        on this feature set is imitating exactly what pure pursuit reacts
        to, not e.g. an absolute world position that wouldn't generalize
        across signatures.
        """
        dvec = target - tip_pos
        c, s = np.cos(yaw), np.sin(yaw)
        dx_local = c * dvec[0] + s * dvec[1]
        dy_local = -s * dvec[0] + c * dvec[1]
        return np.array([dx_local, dy_local, dist_to_final, 1.0 if at_end else 0.0], dtype=np.float32)

    def _expert_action(self, tip_pos: np.ndarray, yaw: float, target: np.ndarray,
                        at_end: bool, dist_to_final: float):
        """Pure-pursuit + exact offset feedback linearization: returns the
        chassis-frame (v, omega) command that steers the pencil tip toward
        `target` at the (end-of-path-aware) nominal speed."""
        # Desired tip velocity: straight toward the lookahead target, at the
        # nominal speed (slowed down near the very end of the path).
        dvec = target - tip_pos
        dist_to_target = float(np.hypot(*dvec))
        direction = dvec / dist_to_target if dist_to_target > 1e-6 else np.array([1.0, 0.0])

        speed = self.speed
        if at_end:
            speed = self.speed * min(1.0, dist_to_final / max(self.lookahead, 1e-6))

        v_tip_world = speed * direction

        # Exact feedback linearization: the pencil tip sits at a fixed offset
        # (tip_offset_x, 0) behind the chassis in its local frame, so
        #   tip_vel_local = (v, omega * tip_offset_x)
        # Invert this to get the chassis (v, omega) that produces the desired
        # tip velocity, instead of approximating the tip as its own unicycle.
        c, s = np.cos(yaw), np.sin(yaw)
        vx_local = c * v_tip_world[0] + s * v_tip_world[1]
        vy_local = -s * v_tip_world[0] + c * v_tip_world[1]

        v = vx_local
        omega = float(np.clip(vy_local / self.tip_offset_x, -10.0, 10.0))
        return v, omega

    def step(self) -> bool:
        """Advances the simulation by one control step. Returns True once finished."""
        if self.finished:
            return True

        m, d = self.m, self.d
        dt = m.opt.timestep

        tip_pos = d.site_xpos[self.site_id][:2].copy()
        self.tip_history.append((tip_pos[0], tip_pos[1], self.step_count * dt))

        yaw = yaw_from_quat(d.qpos[self.chassis_qpos_adr + 3:self.chassis_qpos_adr + 7])

        self._cum_yaw += _wrap_to_pi(yaw - self._prev_yaw)
        self._prev_yaw = yaw
        left_deg = np.degrees(d.qpos[self.joint_left.qposadr[0]] - self._qpos_left0)
        right_deg = np.degrees(d.qpos[self.joint_right.qposadr[0]] - self._qpos_right0)
        self.motor_history.append((self.step_count * dt, np.degrees(self._cum_yaw), left_deg, right_deg))

        target, at_end = self.follower.get_target(tip_pos)
        dist_to_final = float(np.linalg.norm(self.path_world[-1] - tip_pos))
        obs = self._build_observation(tip_pos, yaw, target, at_end, dist_to_final)

        if self.controller is None:
            v, omega = self._expert_action(tip_pos, yaw, target, at_end, dist_to_final)
        else:
            v, omega = self.controller(obs)

        if self.record:
            self.observations.append(obs)
            self.actions.append(np.array([v, omega], dtype=np.float32))

        if dist_to_final < self.finish_tol and at_end:
            self.finished = True
            v, omega = 0.0, 0.0

        # v_wheel(y_offset) = v - omega * y_offset, in the chassis frame (x fwd, y left, z up)
        v_left_wheel = v - omega * self.wheel_left_y
        v_right_wheel = v - omega * self.wheel_right_y
        target_omega_left = v_left_wheel / self.wheel_radius
        target_omega_right = v_right_wheel / self.wheel_radius

        actual_omega_left = d.qvel[self.joint_left.dofadr[0]]
        actual_omega_right = d.qvel[self.joint_right.dofadr[0]]

        d.ctrl[self.joint_to_actuator["joint_left"]] = self.pi_left.step(target_omega_left, actual_omega_left, dt)
        d.ctrl[self.joint_to_actuator["joint_right"]] = self.pi_right.step(target_omega_right, actual_omega_right, dt)

        mujoco.mj_step(m, d)
        self.step_count += 1
        return self.finished

    @property
    def elapsed_time(self) -> float:
        """Simulated seconds since the (post-settle) start of the run."""
        return self.step_count * self.m.opt.timestep

    def tip_history_array(self) -> np.ndarray:
        return np.array(self.tip_history)  # (N, 3): x, y, t

    def motor_history_array(self) -> np.ndarray:
        return np.array(self.motor_history)  # (N, 4): t, yaw_deg, left_wheel_deg, right_wheel_deg


def simulate_reference(path_world: np.ndarray, max_time: float = 60.0, **tracker_kwargs) -> np.ndarray:
    """
    Runs a fresh SignatureTracker to completion (or until max_time) and
    returns its motor_history_array(): (t, yaw_deg, left_wheel_deg,
    right_wheel_deg), all relative to the car's starting pose.

    This is the same pure-pursuit + PI wheel-velocity simulation used for the
    MuJoCo preview elsewhere (track_trajectory.main(), webapp.py's "Run
    simulation"), reused as a physically-grounded reference for driving the
    *real* Double Motor with IMU-heading feedback (see motor_dashboard.py):
    the simulated yaw and wheel-shaft angles it produces are a much better
    proxy for expected motion than a raw geometric turn/drive plan, since
    they already account for pure-pursuit path smoothing and per-wheel
    velocity coupling.
    """
    tracker = SignatureTracker(path_world, **tracker_kwargs)
    dt = tracker.m.opt.timestep
    max_steps = int(max_time / dt)
    for _ in range(max_steps):
        if tracker.step():
            break
    return tracker.motor_history_array()


# -- verification -------------------------------------------------------------

def compute_tracking_error_mm(tip_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """For each actual tip point, distance (mm) to the nearest target path point."""
    errors = [np.min(np.linalg.norm(target_xy - p, axis=1)) for p in tip_xy]
    return np.array(errors) * 1000.0


def plot_comparison(target_world: np.ndarray, tip_history: np.ndarray, out_png: str, show: bool = False):
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors_mm = compute_tracking_error_mm(tip_history[:, :2], target_world)

    fig, ax = plt.subplots(figsize=(6, 6 * PLANE_HEIGHT_MM / PLANE_WIDTH_MM))
    ax.plot(target_world[:, 0] * 1000, target_world[:, 1] * 1000, "--", color="steelblue", label="target")
    ax.plot(tip_history[:, 0] * 1000, tip_history[:, 1] * 1000, "-", color="crimson", linewidth=1.2, label="pencil tip")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Tracking error: max={errors_mm.max():.1f}mm  rms={np.sqrt(np.mean(errors_mm ** 2)):.1f}mm")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return errors_mm


# -- CLI ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--trajectory", type=str, default=None,
                    help="Path to a target_trajectory_*.npz or trajectory_*_paper.npz. "
                         "Defaults to the most recently modified one in this folder.")
    ap.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--path-spacing", type=float, default=DEFAULT_PATH_SPACING,
                    help="Arc-length spacing (m) for the controller's resampled path (default 2mm)")
    ap.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD,
                    help="Pure-pursuit lookahead distance (m). Smaller tracks tight cursive "
                         "loops more faithfully; too small can get jittery (default 6mm)")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Nominal forward speed (m/s)")
    ap.add_argument("--finish-tol", type=float, default=DEFAULT_FINISH_TOL,
                    help="Distance (m) to the final point counted as 'done'")
    ap.add_argument("--max-time", type=float, default=60.0, help="Safety cutoff on simulated seconds")
    ap.add_argument("--view", action="store_true", help="Open a live MuJoCo viewer window while running")
    ap.add_argument("--show", action="store_true", help="Display the verification plot interactively (blocks)")
    ap.add_argument("--output", type=str, default=None, help="Where to save the actual-vs-target plot PNG")
    args = ap.parse_args()

    trajectory_path = args.trajectory or find_latest_trajectory_file(BASE_DIR)
    if trajectory_path is None:
        raise SystemExit("No trajectory .npz found. Pass --trajectory explicitly.")
    print(f"Loading trajectory: {trajectory_path}")

    path_world = load_path_world(trajectory_path, args.smooth_window, args.path_spacing)
    print(f"Path: {len(path_world)} points after resampling, "
          f"length ~{np.sum(np.hypot(*np.diff(path_world, axis=0).T)) * 1000:.1f} mm")

    tracker = SignatureTracker(path_world, model_path=args.model, speed=args.speed,
                                lookahead=args.lookahead, finish_tol=args.finish_tol,
                                path_spacing=args.path_spacing)

    viewer_ctx = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(tracker.m, tracker.d)

    dt = tracker.m.opt.timestep
    max_steps = int(args.max_time / dt)

    for step in range(max_steps):
        finished = tracker.step()

        if viewer_ctx is not None:
            viewer_ctx.sync()
            if not viewer_ctx.is_running():
                break

        if finished:
            print(f"Reached the end of the path at step {step} (t={step * dt:.2f}s).")
            break
    else:
        print("Stopped: reached --max-time without finishing the path.")

    if viewer_ctx is not None:
        viewer_ctx.close()

    tip_history = tracker.tip_history_array()

    timestamp = os.path.splitext(os.path.basename(trajectory_path))[0]
    out_npz = os.path.join(BASE_DIR, f"sim_traced_{timestamp}.npz")
    np.savez(out_npz, actual_trajectory=tip_history, target_world=path_world)
    print(f"Saved actual traced trajectory to {out_npz}")

    out_png = args.output or os.path.join(BASE_DIR, f"sim_traced_{timestamp}.png")
    errors_mm = plot_comparison(path_world, tip_history, out_png, show=args.show)
    print(f"Tracking error (mm): max={errors_mm.max():.2f}  mean={errors_mm.mean():.2f}  "
          f"rms={np.sqrt(np.mean(errors_mm ** 2)):.2f}")
    print(f"Saved verification plot to {out_png}")


if __name__ == "__main__":
    main()
