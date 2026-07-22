"""
webapp.py - Local web UI tying together ArUco calibration, red-dot tracking,
pixel-to-paper trajectory conversion, and a live MuJoCo car-tracking replay,
all in one browser page (no raw MuJoCo viewer window).

Flow:
  1. Open the page; the live camera feed is shown with the ArUco grid overlay.
     Once all 4 markers (IDs 0-3) are visible, the plane is calibrated.
  2. Press ENTER to start recording. The red marker is tracked every frame
     and immediately converted to paper (mm) coordinates using the live
     homography, together with a timestamp.
  3. Press ENTER again to stop. The recorded trajectory is smoothed and
     resampled, saved as target_trajectory_<timestamp>.npz (key
     'target_trajectory', an (N, 3) array of (x, y, t)), and plotted on
     the page.
  4. Click "Run simulation" to drive the simulated LEGO car (track_trajectory.py's
     pure-pursuit controller) along the last recorded trajectory. The MuJoCo
     simulation is rendered offscreen and streamed into the page as a second
     video feed, alongside a tracking-error plot once it finishes.

This must run under an interpreter that has mujoco installed (see run
instructions below) -- on this machine that's Python 3.13, not the default.

Usage:
    py -3.13 -m pip install flask opencv-python numpy matplotlib mujoco
    py -3.13 webapp.py
    -> open http://127.0.0.1:5000 in a browser
"""

import io
import os
import threading
import time
from datetime import datetime

import cv2 as cv
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")

from flask import Flask, Response, jsonify, render_template_string

import coordinate_plane as cp
import track_trajectory as tt
import trajectory_io as tio
from record_trajectory import RedMarkerTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_INDEX = 2
SMOOTH_WINDOW = 5
RESAMPLE_HZ = 60.0

SIM_RENDER_WIDTH = 640
SIM_RENDER_HEIGHT = 480
SIM_TARGET_FPS = 30
SIM_MAX_TIME_S = 60.0  # safety cutoff, matches track_trajectory.py's --max-time default

app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "frame_jpeg": None,
    "calibrated": False,
    "recording": False,
    "trajectory": [],       # list of (x_mm, y_mm, t)
    "trajectory_px": [],    # parallel list of (x_px, y_px), for on-screen drawing only
    "record_start_time": None,
    "last_saved_path": None,
    "last_plot_png": None,

    "sim_running": False,
    "sim_finished": False,
    "sim_message": "Idle.",
    "sim_frame_jpeg": None,
    "sim_error_stats": None,
    "sim_plot_png": None,
    "sim_trajectory_path": None,
}

stop_event = threading.Event()


def camera_loop():
    aruco_dict = cv.aruco.getPredefinedDictionary(0)
    detector = cv.aruco.ArucoDetector(aruco_dict, cv.aruco.DetectorParameters())
    tracker = RedMarkerTracker()   # human-demo pen tip is red

    cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {CAMERA_INDEX}")
        return

    H = None  # last known homography (kept if markers are briefly occluded)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            cv.aruco.drawDetectedMarkers(display, corners, ids)
            H_new, _ = cp._build_homography(ids, corners)
            if H_new is not None:
                H = H_new

        calibrated = H is not None
        if calibrated:
            cp._draw_grid_overlay(display, H)

        pos, _ = tracker.detect(frame)
        if pos is not None:
            cv.circle(display, pos, 8, (0, 255, 0), 2)
            cv.circle(display, pos, 2, (0, 255, 0), -1)

        with state_lock:
            recording = state["recording"]
            start_t = state["record_start_time"]

        if recording and pos is not None and calibrated:
            t = time.time() - start_t
            paper_xy = cp.pixels_to_paper(H, np.array([pos], dtype=np.float32))[0]
            with state_lock:
                state["trajectory"].append((float(paper_xy[0]), float(paper_xy[1]), t))
                state["trajectory_px"].append(pos)

        with state_lock:
            state["calibrated"] = calibrated
            pts_px = list(state["trajectory_px"]) if recording else []

        if len(pts_px) > 1:
            cv.polylines(display, [np.array(pts_px, dtype=np.int32)], False, (0, 0, 255), 2)

        status_text = "RECORDING" if recording else ("CALIBRATED" if calibrated else "CALIBRATING...")
        status_color = (0, 0, 255) if recording else ((0, 220, 0) if calibrated else (0, 180, 255))
        cv.putText(display, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv.LINE_AA)

        ok, buf = cv.imencode(".jpg", display, [int(cv.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with state_lock:
                state["frame_jpeg"] = buf.tobytes()

    cap.release()


def process_and_save(trajectory):
    """Smooths, resamples, saves and plots a finished (x_mm, y_mm, t) recording."""
    if len(trajectory) < 5:
        return "Too few points, discarded.", False

    arr = np.array(trajectory)          # (N, 3): x_mm, y_mm, t
    xy = arr[:, :2]
    t = arr[:, 2]

    xy = cp.smooth_trajectory(xy, SMOOTH_WINDOW)
    xy, t = cp.resample_trajectory(xy, t, RESAMPLE_HZ)

    # Save the trainable .npz + the raw-trajectory .png through the shared saver
    # (same format record_trajectory.py produces), then read the PNG back to
    # stream it to the browser.
    meta = {"mode": "demo", "camera_index": CAMERA_INDEX,
            "smooth_window": SMOOTH_WINDOW, "resample_hz": RESAMPLE_HZ,
            "raw_points": len(arr), "frame": "paper_mm", "source": "webapp"}
    npz_path, png_path = tio.save_trajectory(xy, t,
                                             prefix="target_trajectory", meta=meta)
    with open(png_path, "rb") as f:
        png_bytes = f.read()

    with state_lock:
        state["last_saved_path"] = npz_path
        state["last_plot_png"] = png_bytes

    return f"Saved {len(xy)} points to {os.path.basename(npz_path)}", True


# -- MuJoCo simulation, rendered offscreen and streamed to the browser --------

def simulation_worker(trajectory_path):
    try:
        with state_lock:
            state["sim_message"] = f"Loading {os.path.basename(trajectory_path)}..."

        path_world = tt.load_path_world(trajectory_path)
        tracker = tt.SignatureTracker(path_world)

        renderer = mujoco.Renderer(tracker.m, height=SIM_RENDER_HEIGHT, width=SIM_RENDER_WIDTH)
        cam = mujoco.MjvCamera()
        cam.lookat = [0, 0, 0]
        cam.distance = 0.35
        cam.azimuth = 90
        cam.elevation = -89  # not exactly -90: that hits a gimbal-lock-like degeneracy in the renderer

        dt = tracker.m.opt.timestep
        steps_per_frame = max(1, round((1.0 / SIM_TARGET_FPS) / dt))
        max_steps = int(SIM_MAX_TIME_S / dt)

        finished = False
        for _ in range(max_steps):
            for _ in range(steps_per_frame):
                finished = tracker.step()
                if finished:
                    break

            renderer.update_scene(tracker.d, camera=cam)
            frame_rgb = renderer.render()
            frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            ok, buf = cv.imencode(".jpg", frame_bgr, [int(cv.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with state_lock:
                    state["sim_frame_jpeg"] = buf.tobytes()
                    state["sim_message"] = f"Simulating... step {tracker.step_count}"

            if finished:
                break
        else:
            with state_lock:
                state["sim_message"] = "Stopped: reached the time limit without finishing."

        tip_history = tracker.tip_history_array()
        buf = io.BytesIO()
        errors_mm = tt.plot_comparison(path_world, tip_history, buf, show=False)

        with state_lock:
            state["sim_plot_png"] = buf.getvalue()
            state["sim_error_stats"] = {
                "max_mm": float(errors_mm.max()),
                "mean_mm": float(errors_mm.mean()),
                "rms_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
            }
            state["sim_finished"] = True
            state["sim_message"] = (f"Done. max={errors_mm.max():.1f}mm  "
                                     f"rms={np.sqrt(np.mean(errors_mm ** 2)):.1f}mm")
    except Exception as exc:
        with state_lock:
            state["sim_message"] = f"Simulation failed: {exc}"
    finally:
        with state_lock:
            state["sim_running"] = False


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


def gen_frames():
    while True:
        with state_lock:
            frame = state["frame_jpeg"]
        if frame is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def gen_sim_frames():
    while True:
        with state_lock:
            frame = state["sim_frame_jpeg"]
        if frame is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/sim_feed")
def sim_feed():
    return Response(gen_sim_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        calibrated = state["calibrated"]
        recording = state["recording"]
        n = len(state["trajectory"])
        last_saved = state["last_saved_path"]

    if recording:
        message = f"Recording... {n} points captured. Press ENTER to stop and save."
    elif calibrated:
        message = "Calibrated. Press ENTER to start recording."
    else:
        message = "Waiting for all 4 ArUco markers (IDs 0-3) to be visible..."

    return jsonify({
        "calibrated": calibrated,
        "recording": recording,
        "points": n,
        "message": message,
        "last_saved": last_saved,
    })


@app.route("/toggle_recording", methods=["POST"])
def toggle_recording():
    with state_lock:
        if not state["recording"] and not state["calibrated"]:
            return jsonify({"recording": False, "message": "Cannot start: markers not calibrated yet.", "saved": False})

        if not state["recording"]:
            state["recording"] = True
            state["trajectory"] = []
            state["trajectory_px"] = []
            state["record_start_time"] = time.time()
            return jsonify({"recording": True, "message": "Recording started. Press ENTER to stop and save.", "saved": False})

        state["recording"] = False
        trajectory = list(state["trajectory"])

    message, saved = process_and_save(trajectory)
    return jsonify({"recording": False, "message": message, "saved": saved})


@app.route("/plot")
def plot():
    with state_lock:
        png = state["last_plot_png"]
    if png is None:
        return "", 204
    return Response(png, mimetype="image/png")


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    with state_lock:
        if state["sim_running"]:
            return jsonify({"started": False, "message": "Simulation already running."})
        trajectory_path = state["last_saved_path"]

    if trajectory_path is None:
        trajectory_path = tt.find_latest_trajectory_file(BASE_DIR)
    if trajectory_path is None:
        return jsonify({"started": False, "message": "No recorded trajectory found yet."})

    with state_lock:
        state["sim_running"] = True
        state["sim_finished"] = False
        state["sim_plot_png"] = None
        state["sim_error_stats"] = None
        state["sim_message"] = "Starting simulation..."
        state["sim_trajectory_path"] = trajectory_path

    threading.Thread(target=simulation_worker, args=(trajectory_path,), daemon=True).start()
    return jsonify({"started": True, "message": f"Running simulation on {os.path.basename(trajectory_path)}"})


@app.route("/sim_status")
def sim_status():
    with state_lock:
        return jsonify({
            "running": state["sim_running"],
            "finished": state["sim_finished"],
            "message": state["sim_message"],
            "error_stats": state["sim_error_stats"],
            "trajectory": os.path.basename(state["sim_trajectory_path"]) if state["sim_trajectory_path"] else None,
        })


@app.route("/sim_plot")
def sim_plot():
    with state_lock:
        png = state["sim_plot_png"]
    if png is None:
        return "", 204
    return Response(png, mimetype="image/png")


PAGE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Signature Trajectory Capture</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 16px; }
    h1 { font-size: 1.3rem; }
    h2 { font-size: 1.1rem; margin-top: 36px; }
    img.feed { width: 100%; border-radius: 8px; border: 1px solid #ccc; }
    #status, #sim_status_div { font-size: 1.05rem; padding: 10px 14px; border-radius: 6px; background: #f0f0f0; margin: 12px 0; }
    #plot, #sim_plot_img { width: 100%; margin-top: 12px; border: 1px solid #ccc; border-radius: 8px; display: none; }
    .hint { color: #666; font-size: 0.9rem; }
    button { font-size: 1rem; padding: 8px 16px; border-radius: 6px; border: 1px solid #888; background: #fff; cursor: pointer; }
    button:hover { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>Signature Trajectory Capture</h1>
  <p class="hint">Point the camera at the ArUco sheet. Once calibrated, click on this page and press ENTER to start/stop recording.</p>
  <img class="feed" id="video" src="/video_feed">
  <div id="status">Connecting...</div>
  <img id="plot">

  <h2>MuJoCo Simulation</h2>
  <p class="hint">Drives the simulated LEGO car's pencil tip along the last recorded (or most recent saved) trajectory.</p>
  <button id="run_sim_btn">Run simulation</button>
  <img class="feed" id="sim_video" src="/sim_feed">
  <div id="sim_status_div">Idle.</div>
  <img id="sim_plot_img">

  <script>
    async function pollStatus() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').innerText = data.message;
      } catch (e) {}
    }
    setInterval(pollStatus, 400);
    pollStatus();

    document.addEventListener('keydown', async (e) => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      const res = await fetch('/toggle_recording', { method: 'POST' });
      const data = await res.json();
      document.getElementById('status').innerText = data.message;
      if (!data.recording && data.saved) {
        const img = document.getElementById('plot');
        img.src = '/plot?ts=' + Date.now();
        img.style.display = 'block';
      }
    });

    document.getElementById('run_sim_btn').addEventListener('click', async () => {
      const res = await fetch('/run_simulation', { method: 'POST' });
      const data = await res.json();
      document.getElementById('sim_status_div').innerText = data.message;
      document.getElementById('sim_plot_img').style.display = 'none';
    });

    async function pollSimStatus() {
      try {
        const res = await fetch('/sim_status');
        const data = await res.json();
        document.getElementById('sim_status_div').innerText = data.message;
        if (data.finished) {
          const img = document.getElementById('sim_plot_img');
          img.src = '/sim_plot?ts=' + Date.now();
          img.style.display = 'block';
        }
      } catch (e) {}
    }
    setInterval(pollSimStatus, 500);
    pollSimStatus();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    try:
        app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
    finally:
        stop_event.set()
