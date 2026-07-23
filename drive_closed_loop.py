"""
drive_closed_loop.py - Step 5 (closed loop): drive the real LEGO Double Motor
so its pencil tip follows a recorded signature, using CAMERA-in-the-loop
position feedback plus the built-in IMU for heading.

This is the closed-loop counterpart to:
  * track_trajectory.py     - the same pure-pursuit controller in MuJoCo (sim),
  * run_lego_signature.py   - open-loop dead-reckoning on the real robot,
  * rl/deploy/openloop_deploy.py - open-loop tape replay of the RL policy.

Open loop drifts because nothing corrects wheel slip / heading error. Here we
close the loop: every control tick we MEASURE where the pencil tip actually is
(overhead camera tracking the red dot on the tip, mapped to paper mm through
the same ArUco homography as record_trajectory.py) and where the chassis is
pointing (IMU yaw), then re-run pure pursuit from that live pose. Position
comes from the camera (absolute, no accumulated drift); heading comes from the
IMU (fast, but its zero/sign must be aligned to the paper frame - see below).

Everything is in the PAPER MILLIMETRE frame, no MuJoCo world-metre frame:
  origin (0,0)   = ArUco marker ID0 corner
  +X (to 199mm)  = toward ID1
  +Y (to 137mm)  = toward ID2
The target path (target_trajectory_*.npz / trajectory_*_paper.npz) is already
in this frame, and the camera gives tip positions in it directly.

Heading alignment (the one extra thing closed loop needs):
  The camera sees only ONE point (the tip), so it cannot give orientation.
  The IMU yaw is used instead, but its zero/sign are arbitrary and its raw value
  is in DECIDEGREES (tenths of a degree - IMU_YAW_SCALE=0.1 converts to degrees),
  so paper heading = heading_offset + yaw_sign * radians(raw_yaw * yaw_scale).
  By DEFAULT `drive` establishes heading_offset automatically: it nudges the
  robot forward and measures the tip's travel direction from the camera, so no
  paper reference or manual aiming is needed. It then drives the tip to the
  start; the chassis heading on arrival is arbitrary because pure pursuit
  reorients it smoothly (no in-place spin) once tracing begins. Use
  --manual-start to instead place the tip on the start facing the initial
  heading yourself. Run `imu-check` first to confirm --yaw-sign (default +1,
  CCW-positive).

Subcommands (run them in this order the first time):
  1. preview      Camera only, NO robot. Confirms the red tip is detected, the
                  4 ArUco markers lock, and the target path overlays correctly.
                      py -3.13 drive_closed_loop.py preview --trajectory datasets/trajectories/target_trajectory_20260710_111912.npz
  2. imu-check    Connect the robot, zero yaw, print live IMU yaw while you
                  rotate the robot BY HAND. Confirm the sign (see --yaw-sign).
                      py -3.13 drive_closed_loop.py imu-check --card-serial 2312 --card-color magenta
  3. motor-check  Drive forward/left/right to confirm the wiring flags
                  (--swap-motors / --invert-left / --invert-right). No camera.
                      py -3.13 drive_closed_loop.py motor-check --card-serial 2312 --card-color magenta
  4. drive        The closed-loop run (auto-approaches the start by default).
                      py -3.13 drive_closed_loop.py drive --trajectory datasets/trajectories/target_trajectory_20260710_111912.npz --card-serial 2312 --card-color magenta

Tracing with a trained policy instead of pure pursuit:
  Pass --policy models/bc_policy.pt (behaviour cloning) or --policy
  models/rl_policy.zip (RL). The policy replaces only the tracking control law;
  the auto-approach still uses pure pursuit. Because the policies were trained
  in the sim's METRE frame, the observation is converted mm->m at the boundary
  (see policy_obs_m), the follower lookahead is fixed to the training value
  (6mm), and the policy's m/s speed output is scaled back to mm/s (--speed is
  then ignored). BC is a distillation of the pure-pursuit expert, so expect it
  to roughly match, not beat, pure pursuit.

Motor calibration (deg/s per 100%) is read from rl/deploy/motor_calibration.json;
if it is missing, run `rl/deploy/openloop_deploy.py calibrate` first (or pass
--degs-per-100pct). Wheel wiring flags (--swap-motors / --invert-left /
--invert-right) are identified with the `motor-check` subcommand.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import coordinate_plane as cp
import track_trajectory as tt
from record_trajectory import ColorMarkerTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(BASE_DIR, "rl", "deploy", "motor_calibration.json")
# Closed-loop run outputs (per-tick log .npz + trace .png) live here, parallel
# to datasets/sim_traces (sim) and datasets/trajectories|plots (raw recordings).
CLOSEDLOOP_TRACE_DIR = os.path.join(BASE_DIR, "datasets", "closedloop_traces")

# -- robot geometry, in millimetres (read off lego_car_with_pencil.xml) -------
# Wheels at chassis y = +-36mm; wheel geom radius 0.02m; pencil tip at chassis
# x = -73mm (behind the chassis origin, which is the pure-pursuit reference
# point - the same convention track_trajectory.py's controller was tuned with).
# Re-measure on your real build if it differs and pass the --*-mm overrides.
WHEEL_LEFT_Y_MM = -36.0
WHEEL_RIGHT_Y_MM = 36.0
WHEEL_RADIUS_MM = 20.0
TIP_OFFSET_MM = -73.0          # tip trails the chassis origin along +x_local

# -- hardware wiring (CONFIRMED via `drive_closed_loop.py motor-check`) -------
# On this robot: a FORWARD command drove BACKWARD (so both motors are inverted),
# and with the invert applied LEFT/RIGHT came out reversed (so no swap). Net
# confirmed config: invert both, no swap. Forward drives to the front, LEFT
# turns the nose left, RIGHT turns it right. Override with --swap-motors /
# --no-invert-left / --no-invert-right if you rewire or change robots.
SWAP_MOTORS_DEFAULT = False
INVERT_LEFT_DEFAULT = True
INVERT_RIGHT_DEFAULT = True

# -- control defaults (track_trajectory.py's sim defaults, converted to mm) ---
CONTROL_DT = 0.1               # s, control tick (matches BLE / openloop_deploy.py)
DEFAULT_SPEED_MM_S = 30.0      # = sim DEFAULT_SPEED 0.03 m/s
DEFAULT_LOOKAHEAD_MM = 12.0    # sim uses 6mm; larger is steadier on the slow 10Hz real loop
# A trained BC/RL policy must see the SAME lookahead its observations were built
# with (the sim's tt.DEFAULT_LOOKAHEAD = 6mm), or the obs are out of distribution.
POLICY_LOOKAHEAD_MM = tt.DEFAULT_LOOKAHEAD * 1000.0
DEFAULT_PATH_SPACING_MM = 2.0  # = sim 0.002 m
DEFAULT_FINISH_TOL_MM = 5.0    # sim uses 3mm; a touch looser for the real robot
DEFAULT_OFF_PATH_LIMIT_MM = 30.0   # abort if the tip strays this far from the path
DEFAULT_SMOOTH_WINDOW = tt.DEFAULT_SMOOTH_WINDOW
OMEGA_MAX = 10.0               # rad/s clip, matches the sim expert
DEFAULT_DEGS_PER_100 = 660.0   # SPIKE medium no-load spec, used until calibrated
# The hub reports imu_device.yaw as a raw int16 in DECIDEGREES (tenths of a
# degree) - legoeducation does not scale it - so a 30 deg turn reads 300.
# Multiply by this to get degrees. Verified 2026-07-22 on this robot.
IMU_YAW_SCALE = 0.1

PAPER_W_MM = cp.PLANE_WIDTH_MM
PAPER_H_MM = cp.PLANE_HEIGHT_MM


# -- path loading -------------------------------------------------------------

def load_path_mm(trajectory_path, smooth_window=DEFAULT_SMOOTH_WINDOW,
                 spacing_mm=DEFAULT_PATH_SPACING_MM):
    """Load a trajectory .npz and return an arc-length-resampled (N,2) path in
    paper mm. Reuses track_trajectory's smoothing/resampling, which are
    unit-agnostic (here fed mm instead of metres)."""
    xy_mm, _ = tt.load_trajectory_mm(trajectory_path)
    xy_mm = tt.smooth_xy(xy_mm, smooth_window)
    return tt.resample_by_arclength(xy_mm, spacing_mm)


def initial_heading_rad(path_mm, spacing_mm):
    """Heading (rad, from +X, CCW+) of the path's first ~10mm - the direction
    the robot should face at the start so the zeroed IMU aligns with paper."""
    lookahead_pts = max(2, int(round(10.0 / spacing_mm)))
    p0 = path_mm[0]
    p1 = path_mm[min(lookahead_pts, len(path_mm) - 1)]
    return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


# -- control law (pure pursuit + exact offset feedback linearization, in mm) --

def pure_pursuit_control(tip_mm, yaw_rad, follower, speed_mm_s, lookahead_mm,
                         tip_offset_mm=TIP_OFFSET_MM, omega_max=OMEGA_MAX):
    """Given the measured tip position (paper mm) and chassis yaw (rad),
    return (v_mm_s, omega_rad_s, target_mm, at_end, dist_to_final_mm).

    Identical maths to track_trajectory.SignatureTracker._expert_action, but in
    mm and taking the tip/yaw as live measurements instead of sim ground truth.
    """
    target, at_end = follower.get_target(tip_mm)
    dist_to_final = float(np.linalg.norm(follower.path[-1] - tip_mm))

    dvec = target - tip_mm
    dist_to_target = float(np.hypot(*dvec))
    direction = dvec / dist_to_target if dist_to_target > 1e-6 else np.array([1.0, 0.0])

    speed = speed_mm_s
    if at_end:
        speed = speed_mm_s * min(1.0, dist_to_final / max(lookahead_mm, 1e-6))
    v_tip = speed * direction  # desired tip velocity in the paper frame (mm/s)

    # Rotate the desired tip velocity into the chassis local frame, then invert
    # tip_vel_local = (v, omega * tip_offset) to recover the chassis command.
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    vx_local = c * v_tip[0] + s * v_tip[1]
    vy_local = -s * v_tip[0] + c * v_tip[1]
    v = float(vx_local)
    omega = float(np.clip(vy_local / tip_offset_mm, -omega_max, omega_max))
    return v, omega, target, at_end, dist_to_final


def policy_obs_m(tip_mm, yaw_rad, follower, path_mm):
    """Build the 4-dim observation a trained BC/RL policy expects, in METRES:
    [dx_local, dy_local, dist_to_final, at_end]. Identical to
    track_trajectory.SignatureTracker._build_observation, but our tip/target are
    in mm so the three length components are divided by 1000 (the policies were
    trained in the sim's metre frame). Returns (obs_m, target_mm, at_end, dist_final_mm)."""
    target, at_end = follower.get_target(tip_mm)
    dist_final_mm = float(np.linalg.norm(path_mm[-1] - tip_mm))
    dvec = target - tip_mm
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    dx_local = c * dvec[0] + s * dvec[1]
    dy_local = -s * dvec[0] + c * dvec[1]
    obs_m = np.array([dx_local / 1000.0, dy_local / 1000.0,
                      dist_final_mm / 1000.0, 1.0 if at_end else 0.0], dtype=np.float32)
    return obs_m, target, at_end, dist_final_mm


def vomega_to_wheel_degs(v_mm_s, omega, left_y_mm=WHEEL_LEFT_Y_MM,
                         right_y_mm=WHEEL_RIGHT_Y_MM, radius_mm=WHEEL_RADIUS_MM):
    """Chassis (v, omega) -> (left, right) wheel angular speed in deg/s.
    Same differential-drive conversion as SignatureTracker.step / openloop_deploy."""
    vl = v_mm_s - omega * left_y_mm      # wheel-contact forward speed, mm/s
    vr = v_mm_s - omega * right_y_mm
    wl = vl / radius_mm * 180.0 / np.pi  # deg/s at the wheel shaft
    wr = vr / radius_mm * 180.0 / np.pi
    return wl, wr


def _wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# -- camera front-end ---------------------------------------------------------

class TipCamera:
    """Overhead camera that returns the red pencil-tip position in paper mm.

    Locks the ArUco homography once all four corner markers are visible, then
    reuses it (the robot body may cover a marker mid-run), refreshing whenever
    all four are seen again. Reuses coordinate_plane's homography and
    record_trajectory's colour tracker so the calibration matches the recorder.
    """

    def __init__(self, camera_index=2, dict_id=0, color="red", width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera id={camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the freshest frame
        except Exception:
            pass
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        self.tracker = ColorMarkerTracker(color=color)
        self.H = None

    def poll(self):
        """Grab one frame. Returns (tip_mm or None, frame, locked, n_markers)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None, self.H is not None, 0

        corners, ids, _ = self.detector.detectMarkers(frame)
        n_markers = 0
        if ids is not None and len(ids) > 0:
            n_markers = len(ids)
            H_new, _ = cp._build_homography(ids, corners)
            if H_new is not None:
                self.H = H_new

        tip_mm = None
        if self.H is not None:
            pos, _ = self.tracker.detect(frame)
            if pos is not None:
                tip_mm = cp.pixels_to_paper(self.H, np.array([pos], dtype=np.float32))[0]
        return tip_mm, frame, self.H is not None, n_markers

    def release(self):
        self.cap.release()


def _draw_overlay(frame, H, path_mm, tip_mm=None, target_mm=None, status=""):
    """Draw the paper grid, target path, tip and lookahead target on a frame."""
    disp = frame.copy()
    if H is not None:
        cp._draw_grid_overlay(disp, H)
        pts = np.array([cp._project(H, p) for p in path_mm], dtype=np.int32)
        cv2.polylines(disp, [pts], False, (255, 150, 0), 1, cv2.LINE_AA)
        # Start marker (green dot) + initial-heading arrow: place the tip on the
        # dot and aim the robot's FRONT (the no-pen end) along the arrow.
        if len(path_mm):
            start = path_mm[0]
            k = min(6, len(path_mm) - 1)  # ~12mm ahead at 2mm spacing
            hd = path_mm[k] - start
            n = float(np.hypot(*hd))
            p0 = cp._pt(cp._project(H, start))
            cv2.circle(disp, p0, 7, (0, 255, 0), 2)
            if n > 1e-6:
                arrow_tip = start + hd / n * 25.0
                cv2.arrowedLine(disp, p0, cp._pt(cp._project(H, arrow_tip)),
                                (0, 255, 0), 2, tipLength=0.3)
        if target_mm is not None:
            cv2.circle(disp, cp._pt(cp._project(H, target_mm)), 5, (0, 200, 255), -1)
        if tip_mm is not None:
            cv2.circle(disp, cp._pt(cp._project(H, tip_mm)), 6, (0, 0, 255), 2)
    cv2.putText(disp, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    return disp


def _gif_frame(bgr, width):
    """Downscale a BGR display frame to `width` px and convert to RGB for GIF."""
    h, w = bgr.shape[:2]
    if width and w > width:
        bgr = cv2.resize(bgr, (int(width), int(round(h * width / w))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_gif(frames_rgb, path, fps):
    """Write RGB frames to an animated GIF via Pillow. Best-effort: never fatal."""
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        dur = max(20, int(round(1000.0 / max(fps, 1e-6))))
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=dur, loop=0, optimize=True)
        print(f"Saved GIF ({len(imgs)} frames) to {path}")
    except Exception as exc:
        print(f"(GIF save failed: {exc})")


# -- hardware helpers (mirror rl/deploy/openloop_deploy.py) --------------------

def _connect(args):
    from lelib import doubleMotor
    import legoeducation as le

    card_color = None
    if args.card_color:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}", None)
        if card_color is None:
            raise SystemExit(f"Unknown --card-color '{args.card_color}'")
    print(f"Connecting to Double Motor (card serial {args.card_serial})...")
    dm = doubleMotor()
    dm.connect(card_serial=args.card_serial, card_color=card_color)
    dm.motor_reset_relative_position()
    return dm, le


def _tank(dm, args, left_pct, right_pct):
    if args.invert_left:
        left_pct = -left_pct
    if args.invert_right:
        right_pct = -right_pct
    if args.swap_motors:
        left_pct, right_pct = right_pct, left_pct
    dm.movement_move_tank(float(left_pct), float(right_pct))


def _load_degs_per_100(args):
    if args.degs_per_100pct is not None:
        return args.degs_per_100pct
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            cal = json.load(f)
        print(f"Using calibration: {cal['degs_per_100pct']:.1f} deg/s per 100% "
              f"(from {cal.get('date', '?')})")
        return float(cal["degs_per_100pct"])
    print(f"No calibration file at {CALIBRATION_PATH} - using default "
          f"{DEFAULT_DEGS_PER_100} deg/s per 100%. Run "
          f"`rl/deploy/openloop_deploy.py calibrate` for accuracy.")
    return DEFAULT_DEGS_PER_100


def load_policy_controller(policy_path):
    """Load a trained policy and return (controller_fn, kind_str). controller_fn
    maps a METRE-frame obs (from policy_obs_m) -> (v m/s, omega rad/s):

      * .pt  -> behaviour-cloning MLP (learning/bc_model.load_policy), applying
               the training-time observation normalizer;
      * .zip -> RL PPO policy (rl/signature_env.make_sb3_controller), applying
               the fixed OBS_SCALE / ACTION_SCALE it was trained with.

    Both were trained in the sim's metre frame, so the caller feeds metres and
    scales the returned v back to mm/s."""
    ext = os.path.splitext(policy_path)[1].lower()
    if not os.path.exists(policy_path):
        raise SystemExit(f"Policy file not found: {policy_path}")
    if ext == ".pt":
        sys.path.insert(0, os.path.join(BASE_DIR, "learning"))
        import torch
        from bc_model import load_policy
        model, normalizer, cfg = load_policy(policy_path)

        def controller(obs_m):
            x = normalizer.transform(np.asarray(obs_m, np.float32)[None, :]).astype(np.float32)
            with torch.no_grad():
                a = model(torch.from_numpy(x)).numpy()[0]
            return float(a[0]), float(a[1])
        return controller, f"BC (hidden={cfg['hidden_size']})"
    if ext == ".zip":
        sys.path.insert(0, os.path.join(BASE_DIR, "rl"))
        from stable_baselines3 import PPO
        from signature_env import make_sb3_controller
        return make_sb3_controller(PPO.load(policy_path)), "RL (PPO)"
    raise SystemExit(f"Unknown policy type '{ext}' (expected .pt for BC or .zip for RL)")


def _send_vomega(dm, args, v, omega, degs_per_100):
    """Convert a chassis (v mm/s, omega rad/s) command to tank percents and send
    it (applying the swap/invert wiring flags). Returns (left_pct, right_pct)."""
    wl, wr = vomega_to_wheel_degs(v, omega, args.wheel_left_y_mm,
                                  args.wheel_right_y_mm, args.wheel_radius_mm)
    lp = float(np.clip(wl / degs_per_100 * 100.0, -100, 100))
    rp = float(np.clip(wr / degs_per_100 * 100.0, -100, 100))
    _tank(dm, args, lp, rp)
    return lp, rp


def _yaw_paper(dm, args, heading_offset):
    """Chassis heading in the paper frame (rad): calibrated offset + scaled,
    signed IMU reading. `heading_offset` is set once at start (manual placement
    or the auto-approach's camera-measured nudge)."""
    return _wrap_to_pi(heading_offset + args.yaw_sign
                       * np.radians(float(dm.imu_device.yaw) * args.yaw_scale))


def _pause(cam, args, path_mm, seconds, msg):
    """Idle (robot already stopped) for `seconds`, keeping the camera window
    live and counting down. Returns False if ESC was pressed (abort)."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        tip, frame, locked, n = cam.poll()
        if frame is not None and not args.no_display:
            left = seconds - (time.time() - t0)
            cv2.imshow("closed-loop drive",
                       _draw_overlay(frame, cam.H, path_mm, tip, None, f"{msg} {left:.1f}s"))
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False
        else:
            time.sleep(0.02)
    return True


# -- subcommand: preview (camera only) ----------------------------------------

def cmd_preview(args):
    path_mm = load_path_mm(args.trajectory, args.smooth_window, args.path_spacing)
    heading = np.degrees(initial_heading_rad(path_mm, args.path_spacing))
    print(f"Path: {len(path_mm)} points, initial heading {heading:.1f} deg from +X.")
    print("Move the robot so the red tip sits on the orange path start; check it "
          "tracks in mm. ESC to quit.")
    cam = TipCamera(args.camera, args.dict, args.color)
    try:
        while True:
            tip_mm, frame, locked, n = cam.poll()
            if frame is None:
                continue
            status = f"markers {n}/4  {'LOCKED' if locked else 'NOT locked'}"
            if tip_mm is not None:
                status += f"  tip=({tip_mm[0]:.0f},{tip_mm[1]:.0f})mm"
            else:
                status += "  tip: NOT detected"
            cv2.imshow("closed-loop preview", _draw_overlay(frame, cam.H, path_mm, tip_mm, None, status))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


# -- subcommand: imu-check ----------------------------------------------------

def cmd_imu_check(args):
    dm, le = _connect(args)
    try:
        dm.imu_reset_yaw_axis(0)
        print("IMU yaw zeroed. Rotate the robot BY HAND and watch the sign:")
        print("  turning COUNTER-CLOCKWISE (viewed from above) should read POSITIVE")
        print("  if it reads negative, pass --yaw-sign -1 to `drive`.")
        print(f"  (raw is decidegrees; degrees = raw * {args.yaw_scale}. A 90 deg "
              f"turn should read ~90 deg / ~{int(90/args.yaw_scale)} raw.)")
        print("Ctrl+C to stop.")
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            raw = float(dm.imu_device.yaw)
            print(f"  imu yaw = {raw * args.yaw_scale:8.1f} deg   (raw {raw:8.0f})", end="\r")
            time.sleep(0.1)
        print()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        dm.movement_stop()
        dm.disconnect()


# -- subcommand: motor-check (open-loop actuation sign test) ------------------

def cmd_motor_check(args):
    """Drive forward, then turn left, then turn right, using the SAME
    (v, omega) -> wheel deg/s -> tank chain (swap/invert included) as `drive`.
    No camera. Lets you confirm forward/left/right physically match the control
    convention BEFORE closing the loop - the fastest way to catch a swap/sign
    error that makes closed loop steer the wrong way."""
    degs_per_100 = _load_degs_per_100(args)
    dm, le = _connect(args)

    def run(v, omega, secs, label):
        wl, wr = vomega_to_wheel_degs(v, omega, args.wheel_left_y_mm,
                                      args.wheel_right_y_mm, args.wheel_radius_mm)
        lp = float(np.clip(wl / degs_per_100 * 100.0, -100, 100))
        rp = float(np.clip(wr / degs_per_100 * 100.0, -100, 100))
        print(f"  {label}  (cmd L={lp:+.0f}% R={rp:+.0f}%)")
        t0 = time.time()
        while time.time() - t0 < secs:
            _tank(dm, args, lp, rp)
            time.sleep(CONTROL_DT)
        dm.movement_stop()
        time.sleep(0.8)

    try:
        print(f"Wiring flags: swap_motors={args.swap_motors}  "
              f"invert_left={args.invert_left}  invert_right={args.invert_right}")
        run(args.speed, 0.0, args.seconds,
            "FORWARD  -> expect straight toward the FRONT (pen trailing behind)")
        run(0.4 * args.speed, 2.0, args.seconds,
            "LEFT     -> expect the nose to swing LEFT (counter-clockwise, viewed from above)")
        run(0.4 * args.speed, -2.0, args.seconds,
            "RIGHT    -> expect the nose to swing RIGHT (clockwise, viewed from above)")
        print("\nDiagnosis:")
        print("  * FORWARD went BACKWARD        -> add --invert-left --invert-right")
        print("  * FORWARD veered to one side   -> that wheel is inverted: --invert-left or --invert-right")
        print("  * FORWARD ok but L/R REVERSED  -> toggle the swap: add --no-swap-motors (it is on by default)")
        print("  * all three correct            -> wiring is good; any remaining closed-loop")
        print("                                    drift is the IMU yaw sign (run imu-check).")
    finally:
        dm.movement_stop()
        dm.disconnect()


# -- auto-approach: get to the start from an arbitrary pose -------------------

def _calibrate_and_approach(cam, dm, args, path_mm, degs_per_100):
    """Bring the robot from wherever it sits to the path start, with NO paper
    reference needed. Two stages:

      1. Heading self-calibration - nudge straight forward and measure the tip's
         travel direction from the camera; that IS the chassis heading in the
         paper frame, so no manual aiming is required.
      2. Approach - pure-pursuit the tip to path_mm[0].

    The chassis heading on arrival is arbitrary, and that is fine: the pure-
    pursuit tracker controls the TIP, so it reorients the chassis smoothly (turn
    rate bounded by speed/tip_offset) as it starts tracing - WITHOUT an in-place
    spin, which would sweep the 73mm-offset tip off the start and trip the
    off-path abort. Returns the paper-frame heading offset for `_yaw_paper`, or
    None if aborted / the heading could not be measured. (Pen is always down, so
    the nudge+approach leave a short mark - start near the origin to minimise it.)
    """
    goal = path_mm[0]

    def show(tip, frame, status, target=None):
        if frame is not None and not args.no_display:
            cv2.imshow("closed-loop drive",
                       _draw_overlay(frame, cam.H, path_mm, tip, target, status))
            return (cv2.waitKey(1) & 0xFF) == 27
        return False

    # 1) heading self-calibration nudge --------------------------------------
    dm.imu_reset_yaw_axis(0)
    print("Heading calibration: nudging forward to measure heading from the camera...")
    pts, t0 = [], time.time()
    while time.time() - t0 < args.nudge_max_s:
        tip, frame, locked, n = cam.poll()
        if tip is not None:
            pts.append((tip[0], tip[1], float(dm.imu_device.yaw)))
        _send_vomega(dm, args, args.approach_speed, 0.0, degs_per_100)
        if show(tip if pts else None, frame, f"heading calib: nudging ({len(pts)})"):
            dm.movement_stop(); return None
        if len(pts) >= 2 and np.hypot(pts[-1][0] - pts[0][0],
                                      pts[-1][1] - pts[0][1]) >= args.nudge_mm:
            break
        time.sleep(CONTROL_DT)
    dm.movement_stop()
    time.sleep(0.4)
    if len(pts) < 3:
        print("Heading calibration failed: tip not tracked during the nudge.")
        return None
    k = max(1, len(pts) // 4)
    p0 = np.mean([[p[0], p[1]] for p in pts[:k]], axis=0)
    p1 = np.mean([[p[0], p[1]] for p in pts[-k:]], axis=0)
    disp = p1 - p0
    moved = float(np.hypot(*disp))
    if moved < 5.0:
        print(f"Heading calibration failed: robot moved only {moved:.1f}mm "
              f"(on the ground? try a bigger --nudge-mm / --approach-speed).")
        return None
    heading_est = float(np.arctan2(disp[1], disp[0]))
    imu_after = float(np.mean([p[2] for p in pts[-k:]]))
    heading_offset = _wrap_to_pi(heading_est - args.yaw_sign
                                 * np.radians(imu_after * args.yaw_scale))
    print(f"Measured heading {np.degrees(heading_est):.1f} deg from a {moved:.0f}mm nudge.")

    # 2) approach the start point --------------------------------------------
    print(f"Approaching start ({goal[0]:.0f},{goal[1]:.0f})mm...")
    last_tip = np.array([pts[-1][0], pts[-1][1]])
    t0 = time.time()
    while True:
        if time.time() - t0 > args.approach_max_s:
            print("Approach timed out."); dm.movement_stop(); return None
        tip, frame, locked, n = cam.poll()
        if tip is None:
            tip = last_tip
        else:
            last_tip = tip
        yaw = _yaw_paper(dm, args, heading_offset)
        dvec = goal - tip
        dist = float(np.hypot(*dvec))
        if dist < args.approach_tol:
            break
        direction = dvec / dist
        speed = args.approach_speed * min(1.0, dist / (3.0 * args.approach_tol))
        speed = max(speed, 0.3 * args.approach_speed)  # don't stall short of the goal
        vtx, vty = speed * direction
        c, s = np.cos(yaw), np.sin(yaw)
        v = c * vtx + s * vty
        vy = -s * vtx + c * vty
        omega = float(np.clip(vy / args.tip_offset_mm, -OMEGA_MAX, OMEGA_MAX))
        _send_vomega(dm, args, v, omega, degs_per_100)
        if show(tip, frame, f"approach: {dist:.0f}mm to start", goal):
            dm.movement_stop(); return None
        time.sleep(CONTROL_DT)
    dm.movement_stop()
    time.sleep(0.3)
    return heading_offset


# -- subcommand: drive (closed loop) ------------------------------------------

def cmd_drive(args):
    path_mm = load_path_mm(args.trajectory, args.smooth_window, args.path_spacing)
    heading0 = (np.radians(args.initial_heading_deg)
                if args.initial_heading_deg is not None
                else initial_heading_rad(path_mm, args.path_spacing))
    degs_per_100 = _load_degs_per_100(args)

    # Control law: pure pursuit (default) or a trained policy (--policy). A policy
    # must use its training lookahead (6mm) and sets its own speed (--speed unused).
    policy_controller = None
    track_lookahead = args.lookahead
    if args.policy:
        policy_controller, kind = load_policy_controller(args.policy)
        track_lookahead = POLICY_LOOKAHEAD_MM
        print(f"Policy: {args.policy} [{kind}] - obs converted mm->m, lookahead "
              f"fixed to {track_lookahead:.0f}mm (training value), --speed ignored.")
    follower = tt.PathFollower(path_mm, track_lookahead)

    print(f"Trajectory: {args.trajectory}")
    print(f"Path: {len(path_mm)} points, start ({path_mm[0,0]:.0f},{path_mm[0,1]:.0f})mm, "
          f"initial heading {np.degrees(heading0):.1f} deg from +X.")

    cam = TipCamera(args.camera, args.dict, args.color)
    # Wait for the tip to be detected and the frame to lock before connecting.
    print("Waiting for ArUco lock + red tip detection (ESC to abort)...")
    while True:
        tip_mm, frame, locked, n = cam.poll()
        if frame is not None:
            status = f"markers {n}/4  {'LOCKED' if locked else '...'}  " \
                     f"tip {'ok' if tip_mm is not None else 'NOT found'}"
            cv2.imshow("closed-loop drive", _draw_overlay(frame, cam.H, path_mm, tip_mm, None, status))
        if locked and tip_mm is not None:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            cam.release()
            cv2.destroyAllWindows()
            return

    dm, le = _connect(args)
    log = []
    misses = 0
    last_tip = tip_mm
    record_gif = args.gif is not None
    gif_frames = []
    gif_tick = 0
    try:
        if args.manual_start:
            print(f"Position the PENCIL TIP on the path start, robot facing "
                  f"{np.degrees(heading0):.0f} deg. Zeroing IMU + starting in "
                  f"{args.countdown:.0f}s...")
            time.sleep(args.countdown)
            dm.imu_reset_yaw_axis(0)
            heading_offset = heading0
        else:
            print(f"Auto-approach: put the robot anywhere on the sheet with the red "
                  f"tip visible; it drives to the start itself. Starting in "
                  f"{args.countdown:.0f}s...")
            time.sleep(args.countdown)
            heading_offset = _calibrate_and_approach(cam, dm, args, path_mm,
                                                     degs_per_100)
            if heading_offset is None:
                print("Approach aborted; not starting the trace.")
                return
            print(f"At the start - pausing {args.start_pause_s:.0f}s before tracing...")
            if not _pause(cam, args, path_mm, args.start_pause_s, "at start, tracing in"):
                print("Aborted during the pause.")
                return
            print("Beginning the trace (pure pursuit will orient it)...")

        start = time.time()
        print("Driving (closed loop - Ctrl+C to abort)...")
        while True:
            t = time.time() - start
            if t >= args.max_time:
                print("\nReached --max-time.")
                break

            tip_mm, frame, locked, n = cam.poll()
            if tip_mm is None:
                misses += 1
                tip_mm = last_tip  # hold last known position for this tick
                if misses > args.max_misses:
                    print(f"\nAborting: red tip lost for {misses} consecutive ticks.")
                    break
            else:
                misses = 0
                last_tip = tip_mm

            yaw = _yaw_paper(dm, args, heading_offset)
            if policy_controller is None:
                v, omega, target, at_end, dist_final = pure_pursuit_control(
                    tip_mm, yaw, follower, args.speed, track_lookahead,
                    tip_offset_mm=args.tip_offset_mm)
            else:
                obs_m, target, at_end, dist_final = policy_obs_m(tip_mm, yaw, follower, path_mm)
                v_m, omega = policy_controller(obs_m)
                # m/s -> mm/s, optionally slowed. Scaling v AND omega by the same
                # factor slows traversal while keeping the path shape (curvature
                # = omega/v is unchanged) - useful when an RL policy is too
                # aggressive for the 10 Hz loop it wasn't trained at.
                v = v_m * 1000.0 * args.policy_speed_scale
                omega = omega * args.policy_speed_scale

            # nearest-point tracking error (mm) for the abort test + logging
            err_mm = float(np.min(np.linalg.norm(path_mm - tip_mm, axis=1)))

            done = dist_final < args.finish_tol and at_end
            if done:
                print(f"\nReached the end of the path (t={t:.1f}s).")
            # Grace at the start: the chassis may reorient for the first moment,
            # so don't off-path-abort until it has settled onto the path.
            if err_mm > args.off_path_limit and not done and t > args.start_grace_s:
                print(f"\nAborting: tip strayed {err_mm:.0f}mm off the path "
                      f"(> --off-path-limit {args.off_path_limit:.0f}mm).")
                break

            if done:
                wl = wr = 0.0
            else:
                wl, wr = vomega_to_wheel_degs(v, omega, args.wheel_left_y_mm,
                                              args.wheel_right_y_mm, args.wheel_radius_mm)
            lp = float(np.clip(wl / degs_per_100 * 100.0, -100, 100))
            rp = float(np.clip(wr / degs_per_100 * 100.0, -100, 100))
            _tank(dm, args, lp, rp)

            log.append((t, tip_mm[0], tip_mm[1], np.degrees(yaw),
                        target[0], target[1], v, omega, lp, rp, err_mm))

            if frame is not None and (not args.no_display or record_gif):
                status = f"t={t:4.1f}s err={err_mm:4.0f}mm yaw={np.degrees(yaw):6.1f} " \
                         f"L={lp:5.1f}% R={rp:5.1f}%"
                disp = _draw_overlay(frame, cam.H, path_mm, tip_mm, target, status)
                if record_gif and (gif_tick % max(1, args.gif_every) == 0):
                    gif_frames.append(_gif_frame(disp, args.gif_width))
                gif_tick += 1
                if not args.no_display:
                    cv2.imshow("closed-loop drive", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("\nESC - aborting.")
                        break

            if done:
                break

            # hold the tick to ~CONTROL_DT
            time.sleep(max(0.0, CONTROL_DT - ((time.time() - start) - t)))
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        dm.movement_stop()
        dm.disconnect()
        cam.release()
        cv2.destroyAllWindows()

    _save_and_report(np.array(log), path_mm, args.trajectory)

    if record_gif and gif_frames:
        tag = (os.path.splitext(os.path.basename(args.policy))[0] if args.policy
               else "purepursuit")
        os.makedirs(CLOSEDLOOP_TRACE_DIR, exist_ok=True)
        gif_path = (args.gif if isinstance(args.gif, str) and args.gif != "AUTO"
                    else os.path.join(CLOSEDLOOP_TRACE_DIR,
                                      f"closedloop_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"))
        _write_gif(gif_frames, gif_path, args.gif_fps)


def _save_and_report(log, path_mm, trajectory_path):
    if len(log) < 2:
        print("Too few logged ticks - nothing saved.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(CLOSEDLOOP_TRACE_DIR, exist_ok=True)
    out_npz = os.path.join(CLOSEDLOOP_TRACE_DIR, f"closedloop_log_{ts}.npz")
    np.savez(out_npz, log=log, target_mm=path_mm,
             columns="t,tip_x_mm,tip_y_mm,yaw_deg,target_x_mm,target_y_mm,"
                     "v_mm_s,omega_rad_s,left_pct,right_pct,err_mm")
    print(f"Saved closed-loop log to {out_npz}")

    tip_xy = log[:, 1:3]
    err = np.array([np.min(np.linalg.norm(path_mm - p, axis=1)) for p in tip_xy])
    print(f"Tracking error (mm): max={err.max():.1f}  mean={err.mean():.1f}  "
          f"rms={np.sqrt(np.mean(err ** 2)):.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7 * PAPER_H_MM / PAPER_W_MM))
        ax.plot(path_mm[:, 0], path_mm[:, 1], "--", color="steelblue", label="target")
        ax.plot(tip_xy[:, 0], tip_xy[:, 1], "-", color="crimson", lw=1.3,
                label=f"pencil tip (rms {np.sqrt(np.mean(err**2)):.1f}mm)")
        ax.set_xlim(0, PAPER_W_MM)
        ax.set_ylim(0, PAPER_H_MM)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Closed-loop trace  max={err.max():.1f}mm rms={np.sqrt(np.mean(err**2)):.1f}mm")
        ax.set_aspect("equal")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(CLOSEDLOOP_TRACE_DIR, f"closedloop_trace_{ts}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved trace plot to {out_png}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")


# -- CLI ----------------------------------------------------------------------

def _default_trajectory():
    return tt.find_latest_trajectory_file(BASE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cam_flags(p):
        p.add_argument("--camera", type=int, default=2, help="Camera device id (default 2)")
        p.add_argument("--dict", type=int, default=0, help="ArUco dict id (default 0=DICT_4X4_50)")
        p.add_argument("--color", default="red", help="Tip marker colour (default red)")
        p.add_argument("--trajectory", default=None,
                       help="Target npz (default: newest target/robot trajectory found)")
        p.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
        p.add_argument("--path-spacing", type=float, default=DEFAULT_PATH_SPACING_MM,
                       help="Arc-length resample spacing, mm (default 2)")

    def hw_flags(p):
        p.add_argument("--card-serial", required=True)
        p.add_argument("--card-color", default=None)
        p.add_argument("--swap-motors", dest="swap_motors", action="store_true",
                       default=SWAP_MOTORS_DEFAULT,
                       help="Swap L/R motor commands (default on: jog confirmed "
                            "this robot's motors are wired swapped)")
        p.add_argument("--no-swap-motors", dest="swap_motors", action="store_false",
                       help="Disable the default motor swap")
        p.add_argument("--invert-left", dest="invert_left", action="store_true",
                       default=INVERT_LEFT_DEFAULT,
                       help="Negate left motor (default on: motor-check showed forward reversed)")
        p.add_argument("--no-invert-left", dest="invert_left", action="store_false",
                       help="Disable the default left-motor invert")
        p.add_argument("--invert-right", dest="invert_right", action="store_true",
                       default=INVERT_RIGHT_DEFAULT,
                       help="Negate right motor (default on: motor-check showed forward reversed)")
        p.add_argument("--no-invert-right", dest="invert_right", action="store_false",
                       help="Disable the default right-motor invert")

    def ctrl_flags(p):
        p.add_argument("--policy", default=None,
                       help="Trace with a trained policy instead of pure pursuit: a BC .pt "
                            "or RL .zip (e.g. models/bc_policy.pt). Obs is converted mm->m at "
                            "the boundary; lookahead is fixed to the training value and the "
                            "policy sets its own speed (--speed ignored).")
        p.add_argument("--speed", type=float, default=DEFAULT_SPEED_MM_S, help="Tip speed, mm/s")
        p.add_argument("--policy-speed-scale", type=float, default=1.0,
                       help="Scale a --policy's (v, omega) output by this factor "
                            "(both, so the path shape is preserved). <1 slows an RL "
                            "policy that's too aggressive for the 10 Hz loop.")
        p.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD_MM,
                       help="Pure-pursuit lookahead, mm (default 12)")
        p.add_argument("--finish-tol", type=float, default=DEFAULT_FINISH_TOL_MM,
                       help="Distance to the final point counted as done, mm")
        p.add_argument("--off-path-limit", type=float, default=DEFAULT_OFF_PATH_LIMIT_MM,
                       help="Abort if the tip strays this far from the path, mm")
        p.add_argument("--tip-offset-mm", type=float, default=TIP_OFFSET_MM)
        p.add_argument("--wheel-left-y-mm", type=float, default=WHEEL_LEFT_Y_MM)
        p.add_argument("--wheel-right-y-mm", type=float, default=WHEEL_RIGHT_Y_MM)
        p.add_argument("--wheel-radius-mm", type=float, default=WHEEL_RADIUS_MM)
        p.add_argument("--yaw-sign", type=float, default=1.0, choices=[1.0, -1.0],
                       help="Flip to -1 if imu-check shows CCW as negative")
        p.add_argument("--yaw-scale", type=float, default=IMU_YAW_SCALE,
                       help="Multiply raw IMU yaw to get degrees (hub is decidegrees, 0.1)")
        p.add_argument("--initial-heading-deg", type=float, default=None,
                       help="Override the auto start heading (deg from +X)")
        p.add_argument("--degs-per-100pct", type=float, default=None,
                       help="Override the motor calibration constant")
        p.add_argument("--max-time", type=float, default=90.0, help="Safety cutoff, s")
        p.add_argument("--max-misses", type=int, default=15,
                       help="Abort after this many consecutive tip-detection misses")
        p.add_argument("--countdown", type=float, default=8.0)
        p.add_argument("--no-display", action="store_true", help="Skip the live camera window")
        # -- record the camera overlay as an animated GIF (the tracing phase) --
        p.add_argument("--gif", nargs="?", const="AUTO", default=None,
                       help="Record the camera overlay of the trace as a GIF. Bare --gif "
                            "auto-names it in datasets/closedloop_traces/; or give a path.")
        p.add_argument("--gif-width", type=int, default=640,
                       help="Downscale GIF frames to this width, px (default 640)")
        p.add_argument("--gif-fps", type=float, default=10.0,
                       help="GIF playback frames per second (default 10, ~real time)")
        p.add_argument("--gif-every", type=int, default=1,
                       help="Record every Nth control tick into the GIF (raise to shrink the file)")
        # -- auto-approach to the start (default on) --
        p.add_argument("--manual-start", action="store_true",
                       help="Skip auto-approach: place the tip on the start facing "
                            "the initial heading yourself, IMU is zeroed there")
        p.add_argument("--approach-speed", type=float, default=20.0,
                       help="Tip speed during the nudge/approach, mm/s")
        p.add_argument("--approach-tol", type=float, default=6.0,
                       help="Stop approaching when within this distance of the start, mm")
        p.add_argument("--approach-max-s", type=float, default=40.0,
                       help="Safety timeout for the approach, s")
        p.add_argument("--nudge-mm", type=float, default=20.0,
                       help="Forward nudge distance used to measure heading, mm")
        p.add_argument("--nudge-max-s", type=float, default=3.0,
                       help="Safety timeout for the heading-calibration nudge, s")
        p.add_argument("--start-pause-s", type=float, default=2.0,
                       help="Pause (robot stopped) after reaching the start, before tracing")
        p.add_argument("--start-grace-s", type=float, default=2.0,
                       help="Suppress the off-path abort for this long at the start, "
                            "so the chassis can reorient onto the path without halting")

    p = sub.add_parser("preview", help="Camera only, no robot")
    cam_flags(p)

    p = sub.add_parser("imu-check", help="Verify IMU yaw sign (robot, no driving)")
    hw_flags(p)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--yaw-scale", type=float, default=IMU_YAW_SCALE,
                   help="Multiply raw IMU yaw to get degrees (hub is decidegrees, 0.1)")

    p = sub.add_parser("motor-check", help="Forward/left/right actuation sign test (robot, no camera)")
    hw_flags(p)
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED_MM_S, help="Tip speed, mm/s")
    p.add_argument("--seconds", type=float, default=2.0, help="Seconds per motion")
    p.add_argument("--wheel-left-y-mm", type=float, default=WHEEL_LEFT_Y_MM)
    p.add_argument("--wheel-right-y-mm", type=float, default=WHEEL_RIGHT_Y_MM)
    p.add_argument("--wheel-radius-mm", type=float, default=WHEEL_RADIUS_MM)
    p.add_argument("--degs-per-100pct", type=float, default=None)

    p = sub.add_parser("drive", help="Closed-loop camera+IMU pure-pursuit run")
    cam_flags(p)
    hw_flags(p)
    ctrl_flags(p)

    args = ap.parse_args()
    if getattr(args, "trajectory", None) is None and args.cmd in ("preview", "drive"):
        args.trajectory = _default_trajectory()
        if args.trajectory is None:
            raise SystemExit("No trajectory .npz found. Pass --trajectory explicitly.")
        print(f"Using latest trajectory: {args.trajectory}")

    {"preview": cmd_preview, "imu-check": cmd_imu_check,
     "motor-check": cmd_motor_check, "drive": cmd_drive}[args.cmd](args)


if __name__ == "__main__":
    main()
