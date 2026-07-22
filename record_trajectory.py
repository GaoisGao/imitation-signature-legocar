"""
Signature / Robot Trajectory Recorder
======================================
Uses the overhead camera to track the pen tip and record its path over time,
directly in the paper (mm) coordinate frame via live ArUco calibration.

The tip color tracked depends on which situation is being recorded:
  * --mode demo   : a human signing with a RED-tipped pen
                    -> the reference path  (saved as target_trajectory_<ts>.npz)
  * --mode robot  : the LEGO car's RED pen dot during deployment
                    -> an executed trace   (saved as robot_trace_<ts>.npz)
Both default to red; override with --color red|cyan (cyan preset kept for
setups that use a cyan LEGO tip).

Start/stop control:
  * By default the button on a LEGO color sensor (attached to the demo pen)
    toggles recording: press once to start, press again to stop + save.
    The sensor is connected over Bluetooth (default serial 7036, card color
    magenta). Pass --no-sensor to disable and use the keyboard instead.
  * The keyboard still works as a fallback in the camera window:
        's' start/stop   'r' reset current recording   'q' quit

Every finished recording is saved TWICE, via trajectory_io.save_trajectory:
  * a trainable .npz  (key 'target_trajectory' = (N,3) [x_mm, y_mm, t]), and
  * a .png visualization of the raw trajectory.
Raw camera pixels are kept as an extra 'pixel_trajectory' key for debugging.

Usage:
    pip install opencv-python numpy matplotlib
    py -3.13 record_trajectory.py                  # human demo (cyan tip), sensor button
    py -3.13 record_trajectory.py --mode robot      # deployment trace (red tip)
    py -3.13 record_trajectory.py --no-sensor        # keyboard control only
    py -3.13 record_trajectory.py --sensor-serial 7036 --sensor-card-color magenta

Calibration: all four ArUco corner markers (IDs 0-3) must be visible to lock
the paper frame. Recording only accumulates points while calibration is
locked; a warning is shown otherwise.
"""

import argparse
import os

import cv2
import numpy as np

import coordinate_plane as cp
import trajectory_io as tio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def connect_color_sensor(serial: str, card_color_name: str):
    """Connect to the LEGO color sensor over Bluetooth. Returns a connected
    lelib.colorSensor, or None if the hardware / library is unavailable (the
    recorder then falls back to keyboard control)."""
    try:
        import legoeducation as le
        import lelib
    except Exception as exc:
        print(f"  legoeducation/lelib not importable ({exc}); using keyboard.")
        return None
    const_name = "LEGO_COLOR_" + card_color_name.upper()
    card_color = getattr(le, const_name, None)
    if card_color is None:
        print(f"  Unknown card color '{card_color_name}'; using keyboard.")
        return None
    try:
        sensor = lelib.colorSensor()
        sensor.connect(card_serial=str(serial), card_color=card_color)
        print(f"  Color sensor connected (serial {serial}, card {card_color_name}). "
              f"Press its button to start/stop recording.")
        return sensor
    except Exception as exc:
        print(f"  Could not connect to color sensor ({exc}); using keyboard.")
        return None


# HSV color presets for the pen tip. Red wraps the hue circle, so it needs two
# ranges; cyan is a single range around hue ~90 (OpenCV's 0-179 hue scale).
#   demo (human signature)      -> cyan  LEGO part on the pen tip
#   robot (deployment tracing)  -> red   dot on the robot's pen tip
COLOR_PRESETS = {
    "red":  [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    # Cyan/turquoise LEGO part: broad hue (78-104 covers cyan->azure) with a
    # lenient saturation/value floor so it survives typical overhead lighting.
    # Use --show-mask + click-to-sample (see main) to tune to your exact part.
    "cyan": [((78, 60, 50), (104, 255, 255))],
}
COLOR_BY_MODE = {"demo": "red", "robot": "red"}


class ColorMarkerTracker:
    """Tracks a colored marker (the pen tip) and returns its pixel centroid."""

    def __init__(self, color: str = "red", min_area: int = 100):
        if color not in COLOR_PRESETS:
            raise ValueError(f"unknown color '{color}', choose from {list(COLOR_PRESETS)}")
        self.color = color
        self.ranges = [(np.array(lo), np.array(hi)) for lo, hi in COLOR_PRESETS[color]]
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        """Detects the marker, returns ((cx, cy) or None, mask)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in self.ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else (mask | m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None, mask

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), mask


class RedMarkerTracker(ColorMarkerTracker):
    """Red pen-tip tracker (robot deployment). Kept for backward compatibility
    with webapp.py / rl/deploy/deploy_dashboard.py."""

    def __init__(self, min_area: int = 100):
        super().__init__(color="red", min_area=min_area)


class CyanMarkerTracker(ColorMarkerTracker):
    """Cyan pen-tip tracker (human signature demonstration)."""

    def __init__(self, min_area: int = 100):
        super().__init__(color="cyan", min_area=min_area)


def _save_recording(pixel_traj, paper_traj, mode, camera_index, smooth_window, resample_hz):
    """Smooth + resample a finished recording and save it (npz + png)."""
    if len(paper_traj) < 5:
        print(f"  Too few calibrated points ({len(paper_traj)}), discarding.")
        return None

    arr = np.array(paper_traj)          # (N, 3): x_mm, y_mm, t
    xy, t = arr[:, :2], arr[:, 2]
    xy = cp.smooth_trajectory(xy, smooth_window)
    xy, t = cp.resample_trajectory(xy, t, resample_hz)

    prefix = "target_trajectory" if mode == "demo" else "robot_trace"
    pixel_xy = np.array(pixel_traj)[:, :2] if pixel_traj else None
    meta = {
        "mode": mode,
        "camera_index": camera_index,
        "smooth_window": smooth_window,
        "resample_hz": resample_hz,
        "raw_points": len(paper_traj),
        "frame": "paper_mm",
    }
    npz_path, png_path = tio.save_trajectory(
        xy, t, prefix=prefix, pixel_xy=pixel_xy, meta=meta)
    print(f"  Saved {len(xy)} points -> {os.path.relpath(npz_path, BASE_DIR)}")
    print(f"                        -> {os.path.relpath(png_path, BASE_DIR)}")
    return npz_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=2, help="Camera device id (default 2)")
    ap.add_argument("--dict", type=int, default=0, help="ArUco dictionary id (default 0 = DICT_4X4_50)")
    ap.add_argument("--mode", choices=["demo", "robot"], default="demo",
                    help="'demo' = human signature reference; "
                         "'robot' = deployment trace")
    ap.add_argument("--color", choices=list(COLOR_PRESETS), default=None,
                    help="Pen-tip color to track. Default: red for both modes.")
    ap.add_argument("--smooth-window", type=int, default=5, help="Moving-average window (samples)")
    ap.add_argument("--resample-hz", type=float, default=60.0, help="Uniform resample rate (Hz)")
    ap.add_argument("--no-sensor", action="store_true",
                    help="Do not use the LEGO color-sensor button; keyboard control only.")
    ap.add_argument("--sensor-serial", default="7036", help="Color-sensor Bluetooth serial (default 7036)")
    ap.add_argument("--sensor-card-color", default="magenta",
                    help="Color-sensor pairing card color (default magenta)")
    ap.add_argument("--show-mask", action="store_true",
                    help="Open a second window showing the color-detection mask (for tuning).")
    args = ap.parse_args()

    color = args.color or COLOR_BY_MODE[args.mode]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = ColorMarkerTracker(color=color)
    aruco_dict = cv2.aruco.getPredefinedDictionary(args.dict)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    print("=" * 60)
    print(f"Trajectory Recorder  [mode={args.mode}, tip color={color}, camera={args.camera}]")
    print("=" * 60)
    sensor = None if args.no_sensor else connect_color_sensor(args.sensor_serial,
                                                              args.sensor_card_color)
    if sensor is None:
        print("  Keyboard control: 's' start/stop, 'r' reset, 'q' quit.")
    else:
        print("  Sensor button toggles recording. Keyboard 'q' quits, 'r' resets.")
    print("  Show all 4 ArUco corners (IDs 0-3) to lock the paper frame.")
    print(f"  Click the pen tip in the window to sample its HSV (tuning '{color}').")
    print("=" * 60)

    # Click-to-sample: clicking a pixel prints its HSV and a suggested range,
    # so the tracked color can be tuned to the actual part under real lighting.
    last_frame = {"bgr": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and last_frame["bgr"] is not None:
            bgr = last_frame["bgr"][y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
            hh, ss, vv = int(hsv[0]), int(hsv[1]), int(hsv[2])
            print(f"  sampled ({x},{y}): BGR={tuple(int(v) for v in bgr)}  HSV=({hh},{ss},{vv})  "
                  f"suggest range (({max(0, hh-10)},{max(0, ss-60)},{max(0, vv-60)}), "
                  f"({min(179, hh+10)},255,255))")

    cv2.namedWindow("Trajectory Recorder")
    cv2.setMouseCallback("Trajectory Recorder", on_mouse)

    H = None                 # latest homography (paper mm -> pixels)
    recording = False
    paper_traj = []          # (x_mm, y_mm, t) while calibrated
    pixel_traj = []          # (x_px, y_px, t)
    saved = 0
    start_time = None
    clock = cv2.getTickCount
    freq = cv2.getTickFrequency()
    prev_pressed = False     # for button rising-edge detection
    last_toggle_t = -1.0     # debounce (seconds)

    def start_recording():
        nonlocal recording, paper_traj, pixel_traj, start_time
        recording = True
        paper_traj, pixel_traj = [], []
        start_time = clock()
        print("Recording started...")

    def stop_and_save():
        nonlocal recording, saved
        recording = False
        print("Recording stopped.")
        if _save_recording(pixel_traj, paper_traj, args.mode,
                           args.camera, args.smooth_window, args.resample_hz):
            saved += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        last_frame["bgr"] = frame
        now = clock() / freq

        # -- calibration: refresh homography from the 4 corner markers --
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            H_new, _ = cp._build_homography(ids, corners)
            if H_new is not None:
                H = H_new
        calibrated = H is not None
        if calibrated:
            cp._draw_grid_overlay(display, H)

        # -- pen-tip detection (cyan for demo, red for robot) --
        pos, mask = tracker.detect(frame)
        if pos is not None:
            cv2.circle(display, pos, 8, (0, 255, 0), 2)
            cv2.circle(display, pos, 2, (0, 255, 0), -1)
        if args.show_mask:
            cv2.imshow("color mask", mask)

        # -- accumulate while recording (only paper points if calibrated) --
        if recording and pos is not None:
            t = (clock() - start_time) / freq
            pixel_traj.append((pos[0], pos[1], t))
            if calibrated:
                paper_xy = cp.pixels_to_paper(H, np.array([pos], dtype=np.float32))[0]
                paper_traj.append((float(paper_xy[0]), float(paper_xy[1]), t))

        if recording and len(pixel_traj) > 1:
            pts = np.array([(x, y) for x, y, _ in pixel_traj], dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 0, 255), 2)

        # -- status HUD --
        if recording:
            status, scolor = "REC", (0, 0, 255)
        elif calibrated:
            status, scolor = "CALIBRATED", (0, 220, 0)
        else:
            status, scolor = "CALIBRATING...", (0, 180, 255)
        h = display.shape[0]
        trig = "sensor button" if sensor is not None else "press 's'"
        cv2.putText(display, f"[{args.mode}/{color}] {status}  ({trig})", (10, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, scolor, 2)
        cv2.putText(display,
                    f"paper pts: {len(paper_traj)} | saved: {saved}",
                    (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        if recording and not calibrated:
            cv2.putText(display, "NOT CALIBRATED - points not being recorded!",
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("Trajectory Recorder", display)

        # -- resolve control inputs: sensor button (rising edge) or keyboard --
        toggle = reset = quit_now = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_now = True
        elif key == ord('s'):
            toggle = True
        elif key == ord('r'):
            reset = True

        if sensor is not None:
            try:
                pressed = sensor.button_pressed()
            except Exception:
                pressed = False
            # trigger once on release->press, debounced so a normal press
            # (held for several frames) counts as a single toggle
            if pressed and not prev_pressed and (now - last_toggle_t) > 0.4:
                toggle = True
                last_toggle_t = now
            prev_pressed = pressed

        # -- apply controls --
        if toggle:
            if not recording:
                if calibrated:
                    start_recording()
                else:
                    print("Cannot start: paper frame not calibrated (show all 4 markers).")
            else:
                stop_and_save()
        if reset:
            recording = False
            paper_traj, pixel_traj = [], []
            print("Reset.")
        if quit_now:
            if recording:
                stop_and_save()
            break

    cap.release()
    cv2.destroyAllWindows()
    if sensor is not None:
        try:
            sensor.disconnect()
        except Exception:
            pass
    print(f"\nDone. {saved} recording(s) saved under {os.path.relpath(tio.DATASET_DIR, BASE_DIR)}/")


if __name__ == "__main__":
    main()
