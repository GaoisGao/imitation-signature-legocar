"""
motor_dashboard.py - Live web dashboard that drives the LEGO Double Motor
along a recorded signature trajectory using real-time IMU-heading feedback,
and plots actual vs. expected left/right motor position, motor velocity, and
yaw as it drives.

Reference generation: instead of a blind open-loop turn/drive command
sequence, the "expected" curve is produced by actually simulating the
trajectory in MuJoCo with track_trajectory.py's pure-pursuit controller
(the same simulated robot webapp.py's "Run simulation" button drives) via
tt.simulate_reference(). That gives a physically-grounded (yaw_deg,
left_wheel_deg, right_wheel_deg) time series, which is then time-rescaled
from the simulation's own speed to a rough estimate of the real robot's
driving pace (--nominal-mm-per-s).

--policy PATH swaps that reference source: instead of rolling out
track_trajectory.py's pure-pursuit expert, it rolls out a trained
learning/train_bc.py policy (see learning/evaluate_bc.py, same
SignatureTracker(controller=...) hook) to produce the (yaw_deg,
left_wheel_deg, right_wheel_deg) reference. This is how a BC policy
reaches the real robot: since there's no live position feedback on
hardware (see below), the policy can't run in a closed loop against the
real world directly - instead its full trajectory is rolled out once in
MuJoCo (where it does have position feedback) to get a physically-grounded
reference curve, which is then replayed here exactly like the expert's,
via the same real-time IMU-yaw-corrected tank-drive loop. Keep
--sim-lookahead/--sim-path-spacing at collect_expert_data.py's defaults
(tt.DEFAULT_LOOKAHEAD / tt.DEFAULT_PATH_SPACING) unless the policy was
retrained with different values - the policy's observations are only
valid at the lookahead/path-spacing it was trained on.

Real-time control: the real robot is driven continuously with
movement_move_tank(left_speed, right_speed) at ~10 Hz (not discrete
turn-then-drive commands). At each tick we read the live IMU yaw
(dm.imu_device.yaw) and the reference's expected yaw at the current elapsed
time, and steer the differential speed proportionally to the heading error
(--yaw-kp), so wheel slip or drift gets corrected in real time instead of
compounding silently across an open-loop replay.

Because this is dead-reckoning without any camera/position feedback, the
robot's *starting* pose still has to be set manually and accurately - see
the "Setup" section rendered on the dashboard (also printed to the
console), which gives the initial heading and starting (x, y) in paper-mm
coordinates. A short --countdown-s pause after connecting (before the IMU
yaw axis is zeroed and driving begins) gives you time to finish positioning
it once you've clicked Start.

--simulate fakes the actual-motor/IMU telemetry (no hardware needed) so you
can check the dashboard plumbing before connecting to the robot.

Usage:
    py -3.13 -m pip install flask matplotlib numpy mujoco
    py -3.13 motor_dashboard.py --card-serial 9430
    py -3.13 motor_dashboard.py --simulate
    py -3.13 motor_dashboard.py --policy models/bc_policy.pt --simulate
    -> open http://127.0.0.1:5001
"""

import argparse
import glob
import io
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import run_lego_signature as rls
import track_trajectory as tt
import trajectory_io as tio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_DT = 0.1  # s, tank-drive control tick (~10 Hz; matches lelib.py's Controller.drive() pattern)
REFERENCE_MAX_POINTS = 600  # cap on the (downsampled) MuJoCo reference curve's resolution

app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "running": False,
    "finished": False,
    "message": "Idle.",
    "trajectory_path": None,
    "selected_trajectory": None,  # full path of the trajectory the UI has picked to run next
    "actual": [],        # (t, left_deg, right_deg), sampled once driving actually starts
    "expected": [],      # (t, left_deg, right_deg), precomputed reference
    "yaw_actual": [],    # (t, yaw_deg), sampled once driving actually starts
    "yaw_expected": [],  # (t, yaw_deg), precomputed reference
    "summary": None,
}
# Note: actual/yaw_actual intentionally start empty rather than seeded with a
# (0,0,...) placeholder - the real first sample isn't recorded until after
# the setup countdown, and a placeholder at t=0 sitting almost on top of that
# first real sample makes np.gradient divide by ~0 and spike wildly.


def _reset_state(trajectory_path):
    with state_lock:
        state["running"] = True
        state["finished"] = False
        state["message"] = "Starting..."
        state["trajectory_path"] = trajectory_path
        state["actual"] = []
        state["expected"] = []
        state["yaw_actual"] = []
        state["yaw_expected"] = []
        state["summary"] = None


def _load_paths(trajectory_path, args):
    """Returns (target_path_mm, axle_path_mm): the recorded pen path and the
    pen-offset-corrected path the axle actually needs to drive, both in
    paper-mm coordinates. Shared by the setup-heading calc and the
    trajectory preview plot so they never drift out of sync with each other."""
    xy_mm, _ = tt.load_trajectory_mm(trajectory_path)
    xy_mm = tt.smooth_xy(xy_mm, args.smooth_window)
    fine_path = tt.resample_by_arclength(xy_mm, spacing=1.0)
    axle_path = rls.apply_pen_offset(fine_path, args.pen_offset_mm)
    return fine_path, axle_path


def _setup_info(trajectory_path, args):
    """Returns (initial_heading_deg, start_x_mm, start_y_mm): where to point
    the robot and where its driven axle should start, so a human can
    physically position it before the IMU yaw axis gets zeroed."""
    _, axle_path = _load_paths(trajectory_path, args)
    coarse_path = tt.resample_by_arclength(axle_path, spacing=args.segment_spacing_mm)
    initial_heading_deg, _, _ = rls.build_commands(coarse_path, args.min_turn_deg)
    start_xy = axle_path[0]
    return float(initial_heading_deg), float(start_xy[0]), float(start_xy[1])


def _load_bc_controller(policy_path):
    """Loads a learning/train_bc.py policy and wraps it as a
    track_trajectory.SignatureTracker `controller(obs) -> (v, omega)`
    callable, reusing learning/evaluate_bc.py's normalization + inference
    wrapper so training and deployment see identical input scaling."""
    learning_dir = os.path.join(BASE_DIR, "learning")
    if learning_dir not in sys.path:
        sys.path.insert(0, learning_dir)
    from bc_model import load_policy
    from evaluate_bc import make_bc_controller

    model, normalizer, config = load_policy(policy_path)
    print(f"Loaded BC policy from {policy_path} (hidden_size={config['hidden_size']})")
    return make_bc_controller(model, normalizer)


def _build_reference(trajectory_path, args, controller=None):
    """
    Simulates the trajectory in MuJoCo - by default with
    track_trajectory.py's pure-pursuit controller, or with `controller`
    (a loaded BC policy, see _load_bc_controller) if given - and returns
    (t, yaw_deg, left_deg, right_deg, duration_s): the expected yaw and
    wheel-shaft position over time, time-rescaled from the simulation's own
    pace to --nominal-mm-per-s (a rough estimate of how fast the real robot
    covers ground at --speed - re-tune this by timing an actual run).

    Sign conventions: the simulated wheel angles come out in MuJoCo's own
    convention (same sign for both wheels while driving straight - verified
    directly against the model). The real Double Motor's *encoder* reports
    the left wheel mirrored relative to that (see
    run_lego_signature.LEFT_MOTOR_ENCODER_SIGN), so that correction is
    applied here too, to make the reference directly comparable to
    dm.motor[MOTOR_LEFT/RIGHT].position. The yaw sign convention
    (--yaw-sign) is a similar assumption that hasn't been empirically
    checked against the real IMU yet - flip it if the actual/expected yaw
    plot comes out mirrored the way the left motor once did.
    """
    path_world = tt.load_path_world(trajectory_path, args.smooth_window, args.sim_path_spacing)
    ref = tt.simulate_reference(path_world, max_time=args.sim_max_time_s,
                                speed=args.sim_speed, lookahead=args.sim_lookahead,
                                path_spacing=args.sim_path_spacing, controller=controller)
    sim_t, yaw_deg, left_deg, right_deg = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]

    yaw_deg = yaw_deg * args.yaw_sign
    left_deg = left_deg * rls.LEFT_MOTOR_ENCODER_SIGN
    right_deg = right_deg * rls.RIGHT_MOTOR_ENCODER_SIGN

    total_path_mm = float(np.sum(np.hypot(*np.diff(path_world, axis=0).T))) * 1000.0
    sim_duration = float(sim_t[-1]) if len(sim_t) else 1.0
    real_duration = max(1.0, total_path_mm / args.nominal_mm_per_s)
    scale = real_duration / max(sim_duration, 1e-6)
    ref_t = sim_t * scale

    # The raw sim is sampled at MuJoCo's own ~2ms timestep (thousands of
    # points); downsample for a dashboard reference that's re-plotted on
    # every poll and looked up with np.interp in the real-time control loop.
    if len(ref_t) > REFERENCE_MAX_POINTS:
        coarse_t = np.linspace(ref_t[0], ref_t[-1], REFERENCE_MAX_POINTS)
        yaw_deg = np.interp(coarse_t, ref_t, yaw_deg)
        left_deg = np.interp(coarse_t, ref_t, left_deg)
        right_deg = np.interp(coarse_t, ref_t, right_deg)
        ref_t = coarse_t

    return ref_t, yaw_deg, left_deg, right_deg, real_duration


def _render_trajectory_plot_png(target_mm, axle_mm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6 * tt.PLANE_HEIGHT_MM / tt.PLANE_WIDTH_MM))
    ax.plot(target_mm[:, 0], target_mm[:, 1], "--", color="steelblue", label="pen path (target)")
    ax.plot(axle_mm[:, 0], axle_mm[:, 1], "-", color="crimson", linewidth=1.2, label="axle path (driven)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _hardware_worker(reference, args, stop_evt):
    from lelib import doubleMotor
    import legoeducation as le

    card_color = None
    if args.card_color:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}", None)
        if card_color is None:
            raise RuntimeError(f"Unknown --card-color '{args.card_color}'")

    with state_lock:
        state["message"] = f"Connecting to Double Motor (card serial {args.card_serial})..."

    dm = doubleMotor()
    dm.connect(card_serial=args.card_serial, card_color=card_color)
    dm.motor_reset_relative_position()

    with state_lock:
        state["message"] = f"Connected. Position the robot now - starting in {args.countdown_s:.0f}s..."
    time.sleep(args.countdown_s)
    dm.imu_reset_yaw_axis(0)

    ref_t, ref_yaw, _, _, ref_duration = reference
    start_time = time.time()
    try:
        with state_lock:
            state["message"] = "Driving (closed-loop IMU heading lock)..."
        while True:
            t = time.time() - start_time
            if t >= ref_duration or stop_evt.is_set():
                break

            expected_yaw = float(np.interp(t, ref_t, ref_yaw))
            actual_yaw = float(dm.imu_device.yaw)
            error = rls._angle_diff_deg(expected_yaw, actual_yaw)
            steer = float(np.clip(args.yaw_kp * error, -args.max_steer_pct, args.max_steer_pct))
            left_speed = float(np.clip(args.speed - steer, -100, 100))
            right_speed = float(np.clip(args.speed + steer, -100, 100))
            dm.movement_move_tank(left_speed, right_speed)

            left_pos = float(dm.motor[le.MOTOR_LEFT].position)
            right_pos = float(dm.motor[le.MOTOR_RIGHT].position)
            with state_lock:
                state["actual"].append((t, left_pos, right_pos))
                state["yaw_actual"].append((t, actual_yaw))
                state["message"] = f"Driving... t={t:.1f}/{ref_duration:.1f}s"

            tick_elapsed = (time.time() - start_time) - t
            time.sleep(max(0.0, CONTROL_DT - tick_elapsed))
    finally:
        dm.movement_stop()
        dm.disconnect()


def _simulate_worker(reference, args, stop_evt):
    """Fakes actual-motor/IMU telemetry (first-order lag + noise behind the
    expected reference) so the dashboard can be exercised without hardware."""
    ref_t, ref_yaw, ref_left, ref_right, ref_duration = reference

    with state_lock:
        state["message"] = f"Position the robot now (simulated) - starting in {args.countdown_s:.0f}s..."
    time.sleep(min(args.countdown_s, 1.0))  # don't make local testing painfully slow

    rng = np.random.default_rng()
    sim_yaw = 0.0
    sim_left, sim_right = 0.0, 0.0
    start_time = time.time()
    try:
        with state_lock:
            state["message"] = "Simulating (closed-loop IMU heading lock)..."
        while True:
            t = time.time() - start_time
            if t >= ref_duration or stop_evt.is_set():
                break

            expected_yaw = float(np.interp(t, ref_t, ref_yaw))
            expected_left = float(np.interp(t, ref_t, ref_left))
            expected_right = float(np.interp(t, ref_t, ref_right))

            error = rls._angle_diff_deg(expected_yaw, sim_yaw)
            steer = float(np.clip(args.yaw_kp * error, -args.max_steer_pct, args.max_steer_pct))
            sim_yaw += steer * CONTROL_DT * 0.5 + rng.normal(0, 0.3)
            sim_left += (expected_left - sim_left) * 0.3 + rng.normal(0, 0.4)
            sim_right += (expected_right - sim_right) * 0.3 + rng.normal(0, 0.4)

            with state_lock:
                state["actual"].append((t, sim_left, sim_right))
                state["yaw_actual"].append((t, sim_yaw))
                state["message"] = f"Simulating... t={t:.1f}/{ref_duration:.1f}s"
            time.sleep(CONTROL_DT)
    finally:
        stop_evt.set()


def _dedupe_by_time(arr: np.ndarray) -> np.ndarray:
    """Drops rows with truly duplicate (or ~0-apart) timestamps so
    np.gradient doesn't divide by ~0. The precomputed MuJoCo-derived
    reference can be sampled far finer than the real-time control loop's
    ticks (down to the sim's own ~2ms timestep, further compressed by the
    real-pace rescale), so this threshold must stay well below either."""
    if len(arr) < 2:
        return arr
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 1e-6])
    return arr[keep]


def _error_stats(actual_t, actual_v, expected_t, expected_v):
    exp_at_actual = np.interp(actual_t, expected_t, expected_v)
    err = actual_v - exp_at_actual
    return {
        "final_error": float(err[-1]),
        "max_abs_error": float(np.max(np.abs(err))),
        "rms_error": float(np.sqrt(np.mean(err ** 2))),
    }


def _compute_summary(actual, expected, yaw_actual, yaw_expected):
    if len(actual) < 2 or len(expected) < 2 or len(yaw_actual) < 2 or len(yaw_expected) < 2:
        return None
    a = np.array(actual)
    e = np.array(expected)
    ya = np.array(yaw_actual)
    ye = np.array(yaw_expected)
    left = _error_stats(a[:, 0], a[:, 1], e[:, 0], e[:, 1])
    right = _error_stats(a[:, 0], a[:, 2], e[:, 0], e[:, 2])
    yaw = _error_stats(ya[:, 0], ya[:, 1], ye[:, 0], ye[:, 1])
    return {"left": left, "right": right, "yaw": yaw}


def _drive_worker(trajectory_path, args, controller=None):
    stop_evt = threading.Event()
    source = "BC policy" if controller is not None else "pure-pursuit expert"
    try:
        with state_lock:
            state["message"] = f"Simulating reference path in MuJoCo ({source})..."
        reference = _build_reference(trajectory_path, args, controller=controller)
        ref_t, ref_yaw, ref_left, ref_right, ref_duration = reference
        with state_lock:
            state["expected"] = list(zip(ref_t.tolist(), ref_left.tolist(), ref_right.tolist()))
            state["yaw_expected"] = list(zip(ref_t.tolist(), ref_yaw.tolist()))
            state["message"] = f"Reference ready ({source}, ~{ref_duration:.1f}s estimated). Starting..."

        if args.simulate:
            _simulate_worker(reference, args, stop_evt)
        else:
            _hardware_worker(reference, args, stop_evt)

        with state_lock:
            state["message"] = "Finished driving."
    except Exception as exc:
        stop_evt.set()
        with state_lock:
            state["message"] = f"Run failed: {exc}"
    finally:
        with state_lock:
            actual_snapshot = list(state["actual"])
            expected_snapshot = list(state["expected"])
            yaw_actual_snapshot = list(state["yaw_actual"])
            yaw_expected_snapshot = list(state["yaw_expected"])
        summary = _compute_summary(actual_snapshot, expected_snapshot, yaw_actual_snapshot, yaw_expected_snapshot)
        with state_lock:
            state["summary"] = summary
            state["running"] = False
            state["finished"] = True


def _render_plot_png(actual, expected, yaw_actual, yaw_expected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(actual) < 2 or len(expected) < 2:
        return None

    a = _dedupe_by_time(np.array(actual))
    e = _dedupe_by_time(np.array(expected))
    ya = _dedupe_by_time(np.array(yaw_actual)) if len(yaw_actual) >= 2 else None
    ye = _dedupe_by_time(np.array(yaw_expected)) if len(yaw_expected) >= 2 else None
    if len(a) < 2 or len(e) < 2:
        return None

    v_exp_left = np.gradient(e[:, 1], e[:, 0])
    v_exp_right = np.gradient(e[:, 2], e[:, 0])
    v_act_left = np.gradient(a[:, 1], a[:, 0])
    v_act_right = np.gradient(a[:, 2], a[:, 0])

    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))

    axes[0, 0].plot(e[:, 0], e[:, 1], "--", color="steelblue", label="expected")
    axes[0, 0].plot(a[:, 0], a[:, 1], "-", color="crimson", linewidth=1.2, label="actual")
    axes[0, 0].set_title("Left motor position (deg)")
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].legend()

    axes[0, 1].plot(e[:, 0], e[:, 2], "--", color="steelblue", label="expected")
    axes[0, 1].plot(a[:, 0], a[:, 2], "-", color="crimson", linewidth=1.2, label="actual")
    axes[0, 1].set_title("Right motor position (deg)")
    axes[0, 1].set_xlabel("t (s)")
    axes[0, 1].legend()

    axes[1, 0].plot(e[:, 0], v_exp_left, "--", color="steelblue", label="expected")
    axes[1, 0].plot(a[:, 0], v_act_left, "-", color="crimson", linewidth=1.0, label="actual")
    axes[1, 0].set_title("Left motor velocity (deg/s)")
    axes[1, 0].set_xlabel("t (s)")
    axes[1, 0].legend()

    axes[1, 1].plot(e[:, 0], v_exp_right, "--", color="steelblue", label="expected")
    axes[1, 1].plot(a[:, 0], v_act_right, "-", color="crimson", linewidth=1.0, label="actual")
    axes[1, 1].set_title("Right motor velocity (deg/s)")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].legend()

    if ya is not None and ye is not None and len(ya) >= 2 and len(ye) >= 2:
        yaw_rate_exp = np.gradient(ye[:, 1], ye[:, 0])
        yaw_rate_act = np.gradient(ya[:, 1], ya[:, 0])
        axes[2, 0].plot(ye[:, 0], ye[:, 1], "--", color="steelblue", label="expected")
        axes[2, 0].plot(ya[:, 0], ya[:, 1], "-", color="crimson", linewidth=1.2, label="actual")
        axes[2, 0].set_title("IMU yaw (deg)")
        axes[2, 0].set_xlabel("t (s)")
        axes[2, 0].legend()

        axes[2, 1].plot(ye[:, 0], yaw_rate_exp, "--", color="steelblue", label="expected")
        axes[2, 1].plot(ya[:, 0], yaw_rate_act, "-", color="crimson", linewidth=1.0, label="actual")
        axes[2, 1].set_title("Yaw rate (deg/s)")
        axes[2, 1].set_xlabel("t (s)")
        axes[2, 1].legend()
    else:
        axes[2, 0].axis("off")
        axes[2, 1].axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _list_trajectory_files():
    """Every recorded signature trajectory in the project folder (webapp.py's
    target_trajectory_*.npz or coordinate_plane.py's trajectory_*_paper.npz),
    newest first."""
    return list(reversed(tio.find_trajectory_files(BASE_DIR)))


def _resolve_trajectory_path():
    """The trajectory currently selected in the UI (see /select_trajectory),
    which is also what /start will drive next."""
    with state_lock:
        return state["selected_trajectory"]


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/trajectory_list")
def trajectory_list():
    candidates = _list_trajectory_files()
    with state_lock:
        selected = state["selected_trajectory"]
    return jsonify({
        "trajectories": [os.path.basename(p) for p in candidates],
        "selected": os.path.basename(selected) if selected else None,
    })


@app.route("/select_trajectory", methods=["POST"])
def select_trajectory():
    data = request.get_json(silent=True) or {}
    name = data.get("trajectory")
    if not name:
        return jsonify({"selected": False, "message": "No trajectory name given."})

    # Resolve the basename the UI sent against the known trajectory files
    # (they live under datasets/trajectories, not the project root).
    base = os.path.basename(name)
    path = next((p for p in _list_trajectory_files() if os.path.basename(p) == base), None)
    if path is None or not os.path.isfile(path):
        return jsonify({"selected": False, "message": f"File not found: {name}"})

    with state_lock:
        if state["running"]:
            return jsonify({"selected": False, "message": "Cannot switch trajectory while a run is in progress."})
        state["selected_trajectory"] = path

    return jsonify({"selected": True, "trajectory": os.path.basename(path)})


@app.route("/trajectory_plot")
def trajectory_plot():
    trajectory_path = _resolve_trajectory_path()
    if trajectory_path is None:
        return "", 204
    try:
        target_path, axle_path = _load_paths(trajectory_path, CLI_ARGS)
    except Exception:
        return "", 204
    png = _render_trajectory_plot_png(target_path, axle_path, os.path.basename(trajectory_path))
    return Response(png, mimetype="image/png")


@app.route("/setup_info")
def setup_info():
    trajectory_path = _resolve_trajectory_path()
    if trajectory_path is None:
        return jsonify({"available": False})
    try:
        heading_deg, start_x_mm, start_y_mm = _setup_info(trajectory_path, CLI_ARGS)
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)})
    return jsonify({
        "available": True,
        "trajectory": os.path.basename(trajectory_path),
        "initial_heading_deg": heading_deg,
        "start_x_mm": start_x_mm,
        "start_y_mm": start_y_mm,
        "countdown_s": CLI_ARGS.countdown_s,
    })


@app.route("/start", methods=["POST"])
def start():
    with state_lock:
        if state["running"]:
            return jsonify({"started": False, "message": "A run is already in progress."})
        trajectory_path = state["selected_trajectory"]

    if trajectory_path is None:
        return jsonify({"started": False, "message": "No recorded trajectory .npz found."})

    _reset_state(trajectory_path)
    threading.Thread(target=_drive_worker, args=(trajectory_path, CLI_ARGS, BC_CONTROLLER), daemon=True).start()
    mode = "SIMULATED" if CLI_ARGS.simulate else "real robot"
    source = "BC policy" if BC_CONTROLLER is not None else "pure-pursuit expert"
    return jsonify({"started": True,
                     "message": f"Starting run ({mode}, {source}) on {os.path.basename(trajectory_path)}"})


@app.route("/telemetry_status")
def telemetry_status():
    with state_lock:
        return jsonify({
            "running": state["running"],
            "finished": state["finished"],
            "message": state["message"],
            "summary": state["summary"],
            "trajectory": os.path.basename(state["trajectory_path"]) if state["trajectory_path"] else None,
            "points": len(state["actual"]),
        })


@app.route("/telemetry_plot")
def telemetry_plot():
    with state_lock:
        actual = list(state["actual"])
        expected = list(state["expected"])
        yaw_actual = list(state["yaw_actual"])
        yaw_expected = list(state["yaw_expected"])
    png = _render_plot_png(actual, expected, yaw_actual, yaw_expected)
    if png is None:
        return "", 204
    return Response(png, mimetype="image/png")


PAGE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Motor Telemetry Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 16px; }
    h1 { font-size: 1.3rem; }
    #status_div, #summary_div, #setup_div { font-size: 1.05rem; padding: 10px 14px; border-radius: 6px; background: #f0f0f0; margin: 12px 0; }
    #setup_div { background: #fff8e1; border: 1px solid #e0c46c; }
    #plot_img, #trajectory_img { width: 100%; margin-top: 12px; border: 1px solid #ccc; border-radius: 8px; display: none; }
    h2 { font-size: 1.1rem; margin-top: 28px; }
    .hint { color: #666; font-size: 0.9rem; }
    button, select { font-size: 1rem; padding: 8px 16px; border-radius: 6px; border: 1px solid #888; background: #fff; cursor: pointer; }
    button:hover { background: #f0f0f0; }
    select:disabled { cursor: not-allowed; opacity: 0.6; }
    table { border-collapse: collapse; margin-top: 8px; }
    td, th { padding: 4px 10px; text-align: right; border-bottom: 1px solid #ddd; }
  </style>
</head>
<body>
  <h1>Motor Telemetry Dashboard</h1>
  <p class="hint">Drives the last recorded (or most recent saved) trajectory on the Double Motor with
     real-time IMU-heading feedback, and plots actual vs. expected left/right motor position, motor
     velocity, and yaw live, while it drives. The reference trajectory comes from a MuJoCo rollout of
     either the pure-pursuit expert or a trained BC policy (start this dashboard with --policy to use
     one) - the status line below names which, once you click Start.</p>

  <h2>Trajectory</h2>
  <label for="trajectory_select">Choose from history:</label>
  <select id="trajectory_select"></select>
  <p class="hint">Lists every target_trajectory_*.npz / trajectory_*_paper.npz recorded in the
     project folder, newest first. Picking one updates the setup info and preview plot below.</p>

  <h2>Setup: position the robot before starting</h2>
  <div id="setup_div">Loading setup info...</div>

  <button id="start_btn">Start Run</button>
  <div id="status_div">Idle.</div>

  <h2>Expected Trajectory</h2>
  <p class="hint">The recorded pen path (target) and the pen-offset-corrected axle path the robot
     actually plans to drive, loaded from the trajectory .npz.</p>
  <img id="trajectory_img">

  <h2>Motor &amp; IMU Telemetry</h2>
  <img id="plot_img">
  <div id="summary_div" style="display:none"></div>

  <script>
    function refreshTrajectoryPlot() {
      const img = document.getElementById('trajectory_img');
      img.onerror = () => { img.style.display = 'none'; };
      img.onload = () => { img.style.display = 'block'; };
      img.src = '/trajectory_plot?ts=' + Date.now();
    }
    refreshTrajectoryPlot();

    async function refreshTrajectoryList() {
      try {
        const res = await fetch('/trajectory_list');
        const data = await res.json();
        const sel = document.getElementById('trajectory_select');
        const hadFocus = document.activeElement === sel;
        if (hadFocus) return;  // don't yank options out from under an open dropdown

        sel.innerHTML = '';
        data.trajectories.forEach((name) => {
          const opt = document.createElement('option');
          opt.value = name;
          opt.text = name;
          sel.appendChild(opt);
        });
        if (data.selected) sel.value = data.selected;
      } catch (e) {}
    }
    refreshTrajectoryList();
    setInterval(refreshTrajectoryList, 5000);

    document.getElementById('trajectory_select').addEventListener('change', async (e) => {
      const name = e.target.value;
      const res = await fetch('/select_trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trajectory: name }),
      });
      const data = await res.json();
      if (data.selected) {
        refreshTrajectoryPlot();
        refreshSetupInfo();
      } else {
        alert(data.message);
        refreshTrajectoryList();
      }
    });

    async function refreshSetupInfo() {
      try {
        const res = await fetch('/setup_info');
        const data = await res.json();
        const div = document.getElementById('setup_div');
        if (!data.available) {
          div.innerText = 'No trajectory found yet.';
          return;
        }
        div.innerHTML =
          '<b>' + data.trajectory + '</b><br>' +
          'Point the robot\\'s heading at <b>' + data.initial_heading_deg.toFixed(1) +
          '&deg;</b> from the paper\\'s +X axis (the edge from marker ID0 towards ID1), ' +
          'with its driven axle starting at <b>(' + data.start_x_mm.toFixed(1) + ', ' +
          data.start_y_mm.toFixed(1) + ') mm</b> on the paper.<br>' +
          'After clicking Start, you have <b>' + data.countdown_s.toFixed(0) +
          's</b> after it connects to finish positioning it before the IMU yaw axis ' +
          'is zeroed and it starts driving.';
      } catch (e) {}
    }
    refreshSetupInfo();

    document.getElementById('start_btn').addEventListener('click', async () => {
      const res = await fetch('/start', { method: 'POST' });
      const data = await res.json();
      document.getElementById('status_div').innerText = data.message;
      document.getElementById('summary_div').style.display = 'none';
      if (data.started) { refreshTrajectoryPlot(); refreshSetupInfo(); }
    });

    function renderSummary(s) {
      const div = document.getElementById('summary_div');
      if (!s) { div.style.display = 'none'; return; }
      function row(label, d) {
        return '<tr><td>' + label + '</td><td>' + d.final_error.toFixed(1) + '</td><td>' +
          d.max_abs_error.toFixed(1) + '</td><td>' + d.rms_error.toFixed(1) + '</td></tr>';
      }
      div.innerHTML = '<b>Actual vs. expected</b><table>' +
        '<tr><th></th><th>final err</th><th>max |err|</th><th>rms err</th></tr>' +
        row('Left motor (deg)', s.left) +
        row('Right motor (deg)', s.right) +
        row('Yaw (deg)', s.yaw) +
        '</table>';
      div.style.display = 'block';
    }

    async function poll() {
      try {
        const res = await fetch('/telemetry_status');
        const data = await res.json();
        document.getElementById('status_div').innerText = data.message +
          (data.trajectory ? ' (' + data.trajectory + ', ' + data.points + ' samples)' : '');
        document.getElementById('trajectory_select').disabled = data.running;
        if (data.points > 1) {
          const img = document.getElementById('plot_img');
          img.src = '/telemetry_plot?ts=' + Date.now();
          img.style.display = 'block';
        }
        if (data.finished && data.summary) {
          renderSummary(data.summary);
        }
      } catch (e) {}
    }
    setInterval(poll, 500);
    poll();
  </script>
</body>
</html>
"""


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=str, default=None,
                    help="Path to a target_trajectory_*.npz or trajectory_*_paper.npz. "
                         "Defaults to the most recently modified one in this folder.")
    ap.add_argument("--wheel-diameter-mm", type=float, default=rls.WHEEL_DIAMETER_MM)
    ap.add_argument("--pen-offset-mm", type=float, default=rls.PEN_OFFSET_MM)
    ap.add_argument("--track-width-mm", type=float, default=rls.TRACK_WIDTH_MM)
    ap.add_argument("--segment-spacing-mm", type=float, default=rls.DEFAULT_SEGMENT_SPACING_MM)
    ap.add_argument("--min-turn-deg", type=float, default=rls.DEFAULT_MIN_TURN_DEG)
    ap.add_argument("--speed", type=int, default=rls.DEFAULT_SPEED,
                    help="Base tank-drive speed, 0-100%%, before steering correction")
    ap.add_argument("--smooth-window", type=int, default=tt.DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--card-serial", type=str, default=rls.DEFAULT_CARD_SERIAL,
                    help="Bluetooth connection card serial for the Double Motor")
    ap.add_argument("--card-color", type=str, default=rls.DEFAULT_CARD_COLOR,
                    help="Bluetooth connection card color, e.g. AZURE, RED, BLUE.")

    ap.add_argument("--policy", type=str, default=None,
                    help="Path to a trained BC policy (learning/train_bc.py's output, e.g. "
                         "models/bc_policy.pt). If given, the MuJoCo reference is rolled out "
                         "from this policy instead of the pure-pursuit expert. Keep "
                         "--sim-lookahead/--sim-path-spacing at their defaults unless the "
                         "policy was retrained with different values.")
    ap.add_argument("--sim-speed", type=float, default=tt.DEFAULT_SPEED,
                    help="Nominal forward speed (m/s) for the MuJoCo reference simulation "
                         "(pure-pursuit expert only; ignored when --policy is given)")
    ap.add_argument("--sim-lookahead", type=float, default=tt.DEFAULT_LOOKAHEAD,
                    help="Pure-pursuit lookahead (m) for the MuJoCo reference simulation. Also "
                         "sets the BC policy's lookahead search distance when --policy is given "
                         "- must match what learning/collect_expert_data.py used to train it.")
    ap.add_argument("--sim-path-spacing", type=float, default=tt.DEFAULT_PATH_SPACING,
                    help="Arc-length spacing (m) for the MuJoCo reference simulation's path")
    ap.add_argument("--sim-max-time-s", type=float, default=60.0,
                    help="Safety cutoff (s) on the MuJoCo reference simulation")
    ap.add_argument("--nominal-mm-per-s", type=float, default=60.0,
                    help="TUNE: estimated real-world driving pace (mm/s) at --speed, used to "
                         "time-rescale the simulated reference to a realistic duration. "
                         "Re-measure by timing an actual run and adjust.")
    ap.add_argument("--yaw-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="TUNE: flip to -1 if the actual/expected yaw plot comes out mirrored "
                         "(unverified against real IMU polarity)")
    ap.add_argument("--yaw-kp", type=float, default=1.2,
                    help="Steering correction, in %% speed differential per degree of heading error")
    ap.add_argument("--max-steer-pct", type=float, default=40.0,
                    help="Clamp on the steering correction so it can't dominate forward speed")
    ap.add_argument("--countdown-s", type=float, default=6.0,
                    help="Pause after connecting (before zeroing the IMU yaw axis and driving) "
                         "so you can finish positioning the robot")

    ap.add_argument("--simulate", action="store_true",
                    help="Fake the actual-motor/IMU telemetry instead of connecting to hardware, "
                         "to test the dashboard without a robot")
    ap.add_argument("--port", type=int, default=5001)
    return ap


CLI_ARGS = None
BC_CONTROLLER = None

if __name__ == "__main__":
    CLI_ARGS = build_argparser().parse_args()
    if CLI_ARGS.policy:
        BC_CONTROLLER = _load_bc_controller(CLI_ARGS.policy)
    with state_lock:
        state["selected_trajectory"] = CLI_ARGS.trajectory or tt.find_latest_trajectory_file(BASE_DIR)
    print(f"Open http://127.0.0.1:{CLI_ARGS.port} to view the live motor telemetry dashboard.")
    app.run(host="127.0.0.1", port=CLI_ARGS.port, threaded=True, debug=False)
