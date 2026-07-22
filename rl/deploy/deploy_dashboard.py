"""deploy_dashboard.py - browser UI for open-loop deployment of the trained
RL policy on the real LEGO car (wraps openloop_deploy.py; same style as
motor_dashboard.py / webapp.py).

Usage:
    py -3.13 rl/deploy/deploy_dashboard.py
    -> open http://localhost:5050

Panels, in workflow order:
  0. Record   - overhead camera + ArUco + red-tip tracking (webapp.py's
                pipeline): record a signature right here; it lands in the
                Tape panel's dropdown. Also usable during DRIVE to capture
                the robot's own trace for Compare.
  1. Tape     - pick a recorded signature, generate the policy's command tape
  2. Robot    - card serial + wiring flags; Jog (identify motors) and
                Calibrate (wheels off the ground -> deg/s per percent)
  3. Drive    - speed scale + countdown, live progress, big STOP button
  4. Compare  - after recording the drawn line with the camera pipeline,
                pick that npz and see the gap plot + numbers
"""

import glob
import json
import os
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(DEPLOY_DIR)
PROJECT_DIR = os.path.dirname(RL_DIR)
for p in (PROJECT_DIR, RL_DIR, DEPLOY_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

import coordinate_plane as cp
import openloop_deploy as od
import trajectory_io as tio
from record_trajectory import RedMarkerTracker

SETTINGS_PATH = os.path.join(DEPLOY_DIR, "dashboard_settings.json")

app = Flask(__name__)

state_lock = threading.Lock()
state = {
  "busy": False,
  "task": None,
  "message": "Ready.",
  "progress": None,          # 0..1 while driving
  "tape_info": None,         # summary dict from make_tape
  "drive_summary": None,     # text lines after a drive
  "compare_stats": None,     # dict from run_compare
  "calibration": None,
}
stop_evt = threading.Event()


def _set(**kw):
  with state_lock:
    state.update(kw)


def _load_settings():
  if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH) as f:
      return json.load(f)
  return {}


def _save_settings(d):
  keep = {k: d.get(k) for k in ("card_serial", "card_color", "swap_motors",
                                "invert_left", "invert_right", "speed_scale",
                                "countdown")}
  with open(SETTINGS_PATH, "w") as f:
    json.dump(keep, f, indent=2)


def _hw_args(d) -> SimpleNamespace:
  return SimpleNamespace(
    card_serial=d.get("card_serial", ""),
    card_color=d.get("card_color") or None,
    swap_motors=bool(d.get("swap_motors")),
    invert_left=bool(d.get("invert_left")),
    invert_right=bool(d.get("invert_right")),
    degs_per_100pct=None,
  )


def _load_calibration():
  if os.path.exists(od.CALIBRATION_PATH):
    with open(od.CALIBRATION_PATH) as f:
      return json.load(f)
  return None


def _run_in_thread(task_name, fn, *fn_args):
  def runner():
    try:
      fn(*fn_args)
    except Exception as exc:  # surface any error in the UI
      _set(message=f"ERROR in {task_name}: {exc}")
    finally:
      _set(busy=False, task=None, progress=None)

  with state_lock:
    if state["busy"]:
      return False
    state.update(busy=True, task=task_name, message=f"{task_name} running...")
  stop_evt.clear()
  threading.Thread(target=runner, daemon=True).start()
  return True


# -- camera recorder ---------------------------------------------------------
#
# Same pipeline as webapp.py: ArUco corner markers -> homography, red pencil
# tip -> paper-mm points while recording. Used twice in the workflow: to
# record the HUMAN's signature (before generating a tape), and - if the
# robot's pencil also carries the red marker - to record the ROBOT's trace
# live while it drives, giving the Compare panel its "drawn" npz.


class CameraRecorder:
  def __init__(self):
    self.lock = threading.Lock()
    self.running = False
    self.recording = False
    self.status = "camera off"
    self.frame_jpeg = None
    self.points = []            # (x_mm, y_mm, t) while recording
    self.calibrated = False
    self.last_saved = None
    self._thread = None
    self._t0 = None

  def start(self, index: int):
    with self.lock:
      if self.running:
        return
      self.running = True
      self.status = f"opening camera {index}..."
    self._thread = threading.Thread(target=self._loop, args=(index,), daemon=True)
    self._thread.start()

  def stop(self):
    with self.lock:
      if self.recording:
        self._finish_recording_locked()
      self.running = False
      self.status = "camera off"
      self.frame_jpeg = None

  def toggle_record(self):
    with self.lock:
      if not self.running:
        self.status = "start the camera first"
        return
      if not self.recording:
        if not self.calibrated:
          self.status = "cannot record: 4 ArUco markers not locked yet"
          return
        self.points = []
        self._t0 = time.time()
        self.recording = True
      else:
        self._finish_recording_locked()

  def _finish_recording_locked(self):
    self.recording = False
    pts = self.points
    self.points = []
    if len(pts) < 10:
      self.status = f"recording discarded ({len(pts)} points is too few)"
      return
    arr = np.array(pts)                       # (N, 3): x_mm, y_mm, t
    xy = cp.smooth_trajectory(arr[:, :2], 5)
    xy, t = cp.resample_trajectory(xy, arr[:, 2], 60.0)
    npz_path, _ = tio.save_trajectory(
        xy, t, prefix="target_trajectory",
        meta={"mode": "demo", "frame": "paper_mm", "source": "deploy_dashboard"})
    self.last_saved = os.path.basename(npz_path)
    self.status = f"saved {len(xy)} points -> {self.last_saved}"

  def _loop(self, index: int):
    try:
      self._loop_inner(index)
    except Exception as exc:
      # Never die silently: surface the error in the UI status line.
      with self.lock:
        self.status = f"camera ERROR: {exc}"
        self.running = False
        self.recording = False

  def _loop_inner(self, index: int):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened():
      with self.lock:
        self.status = f"cannot open camera {index}"
        self.running = False
      return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv.aruco.ArucoDetector(
      cv.aruco.getPredefinedDictionary(0), cv.aruco.DetectorParameters())
    tracker = RedMarkerTracker()
    H = None

    while True:
      with self.lock:
        if not self.running:
          break
      ok, frame = cap.read()
      if not ok:
        with self.lock:
          self.status = "camera read failed"
          self.running = False
        break

      corners, ids, _ = detector.detectMarkers(frame)
      if ids is not None and len(ids) > 0:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        H_new, _ = cp._build_homography(ids, corners)
        if H_new is not None:
          H = H_new

      pos, _ = tracker.detect(frame)
      if pos is not None:
        cv.circle(frame, pos, 8, (0, 255, 0), 2)

      with self.lock:
        self.calibrated = H is not None
        if self.recording and pos is not None and H is not None:
          mm = cp.pixels_to_paper(H, np.array([pos], dtype=np.float64))[0]
          self.points.append((float(mm[0]), float(mm[1]), time.time() - self._t0))
        n = len(self.points)
        rec = self.recording

      if rec:
        cv.putText(frame, f"REC {n} pts", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.9, (0, 0, 255), 2)
      else:
        txt = "CALIBRATED" if H is not None else "searching ArUco markers 0-3..."
        cv.putText(frame, txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                   (0, 200, 0) if H is not None else (0, 165, 255), 2)

      ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
      with self.lock:
        if ok:
          self.frame_jpeg = jpg.tobytes()
        if rec:
          self.status = f"recording... {n} points"
        else:
          self.status = ("calibrated - ready to record" if H is not None
                         else "searching ArUco markers 0-3...")
    cap.release()


recorder = CameraRecorder()


# -- workers ---------------------------------------------------------------


def _tape_worker(trajectory, model):
  _set(message="Rolling out the policy in simulation...")
  info = od.make_tape(model, trajectory, log=lambda m: _set(message=m))
  warn = ""
  if info["max_wheel_degs"] > 660:
    warn = (f"  |  WARNING: peak wheel speed {info['max_wheel_degs']:.0f} deg/s exceeds "
            f"the motor's ~660 deg/s - drive with speed scale <= "
            f"{660 / info['max_wheel_degs']:.2f}")
  _set(tape_info=info,
       message=f"Tape ready: {info['steps']} steps, {info['duration_s']:.1f}s, "
               f"peak {info['max_wheel_degs']:.0f} deg/s.{warn}")


def _jog_worker(hw):
  args = _hw_args(hw)
  _set(message="Connecting...")
  dm, le = od._connect(args)
  try:
    _set(message="Spinning the LEFT motor forward for 2s - watch which wheel moves...")
    dm.motor_set_speed(30, motor=le.MOTOR_LEFT)
    dm.motor_run(motor=le.MOTOR_LEFT)
    time.sleep(2)
    dm.motor_stop(motor=le.MOTOR_LEFT)
    time.sleep(1)
    _set(message="Spinning the RIGHT motor forward for 2s...")
    dm.motor_set_speed(30, motor=le.MOTOR_RIGHT)
    dm.motor_run(motor=le.MOTOR_RIGHT)
    time.sleep(2)
    dm.motor_stop(motor=le.MOTOR_RIGHT)
    _set(message="Jog done. If LEFT moved the right wheel, tick 'swap motors'. "
                 "If a wheel spun backward, tick the matching 'invert'.")
  finally:
    dm.movement_stop()
    dm.disconnect()


def _calibrate_worker(hw):
  args = _hw_args(hw)
  _set(message="Connecting...")
  dm, le = od._connect(args)
  try:
    _set(message="Calibrating: motors at 30% for 3s (wheels must be OFF the ground)...")
    l0, r0 = od._positions(dm, le, args)
    od._tank(dm, args, 30.0, 30.0)
    t0 = time.time()
    time.sleep(3.0)
    dm.movement_stop()
    elapsed = time.time() - t0
    l1, r1 = od._positions(dm, le, args)
    left_dps = (l1 - l0) / elapsed
    right_dps = (r1 - r0) / elapsed
    degs_per_100 = (abs(left_dps) + abs(right_dps)) / 2.0 * (100.0 / 30.0)
    cal = {"degs_per_100pct": degs_per_100, "left_dps": left_dps,
           "right_dps": right_dps, "pct": 30.0, "date": datetime.now().isoformat()}
    with open(od.CALIBRATION_PATH, "w") as f:
      json.dump(cal, f, indent=2)
    msg = (f"Calibrated: {degs_per_100:.0f} deg/s per 100% "
           f"(L {left_dps:.0f}, R {right_dps:.0f} deg/s at 30%).")
    if left_dps < 0 or right_dps < 0:
      msg += " WARNING: negative direction seen - check invert flags (Jog) and redo."
    _set(calibration=cal, message=msg)
  finally:
    dm.movement_stop()
    dm.disconnect()


def _drive_worker(hw, tape_path, speed_scale, countdown):
  tape = np.load(tape_path, allow_pickle=True)
  t_tape = tape["t"]
  duration = float(t_tape[-1]) / speed_scale
  args = _hw_args(hw)
  cal = _load_calibration()
  degs_per_100 = cal["degs_per_100pct"] if cal else od.DEFAULT_DEGS_PER_100

  _set(message="Connecting...")
  dm, le = od._connect(args)
  saturated = 0
  log = []
  try:
    for i in range(int(countdown), 0, -1):
      if stop_evt.is_set():
        return
      _set(message=f"Place the PENCIL TIP on the path start, car facing along "
                   f"the path. Driving in {i}s...")
      time.sleep(1)
    dm.imu_reset_yaw_axis(0)
    dm.motor_reset_relative_position()
    l_off, r_off = od._positions(dm, le, args)

    start = time.time()
    while True:
      t = time.time() - start
      if t >= duration or stop_evt.is_set():
        break
      t_sim = t * speed_scale
      wl = float(np.interp(t_sim, t_tape, tape["wheel_left_degs"])) * speed_scale
      wr = float(np.interp(t_sim, t_tape, tape["wheel_right_degs"])) * speed_scale
      lp_cmd = wl / degs_per_100 * 100.0
      rp_cmd = wr / degs_per_100 * 100.0
      if abs(lp_cmd) > 100 or abs(rp_cmd) > 100:
        saturated += 1
      od._tank(dm, args, float(np.clip(lp_cmd, -100, 100)),
               float(np.clip(rp_cmd, -100, 100)))

      lp, rp = od._positions(dm, le, args)
      yaw = float(dm.imu_device.yaw)
      log.append((t, lp - l_off, rp - r_off, yaw, lp_cmd, rp_cmd))
      _set(progress=t / duration,
           message=f"Driving... {t:.1f}/{duration:.1f}s  "
                   f"(cmd L {lp_cmd:.0f}% R {rp_cmd:.0f}%)")

      tick_elapsed = (time.time() - start) - t
      time.sleep(max(0.0, od.CONTROL_DT - tick_elapsed))
  finally:
    dm.movement_stop()
    dm.disconnect()

  log = np.array(log)
  summary = []
  if len(log):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(DEPLOY_DIR, f"drive_log_{ts}.npz")
    np.savez(out, log=log, tape_file=os.path.basename(tape_path),
             speed_scale=speed_scale, degs_per_100pct=degs_per_100,
             columns="t,left_pos_deg,right_pos_deg,yaw_deg,left_cmd_pct,right_cmd_pct")
    t_end = log[-1, 0] * speed_scale
    sim_l = float(np.interp(t_end, t_tape, tape["sim_left_pos_deg"]))
    sim_r = float(np.interp(t_end, t_tape, tape["sim_right_pos_deg"]))
    sim_y = float(np.interp(t_end, t_tape, tape["sim_yaw_deg"]))
    summary = [
      f"left wheel:  {log[-1, 1]:.0f} deg (sim {sim_l:.0f})",
      f"right wheel: {log[-1, 2]:.0f} deg (sim {sim_r:.0f})",
      f"final yaw:   {log[-1, 3]:.0f} deg (sim {sim_y:.0f}; IMU sign may differ)",
      f"saturated ticks: {saturated}",
      f"log saved: {os.path.basename(out)}",
    ]
  stopped = "stopped early" if stop_evt.is_set() else "finished"
  _set(drive_summary=summary,
       message=f"Drive {stopped}. Now record the drawn line with the camera "
               f"pipeline, then use the Compare panel.")


def _compare_worker(drawn, tape_path):
  _set(message="Comparing drawn trace against target + sim...")
  stats = od.run_compare(drawn, tape_path or None,
                         output=os.path.join(DEPLOY_DIR, "openloop_gap.png"),
                         log=lambda m: None)
  _set(compare_stats=stats,
       message=f"Gap: real rms {stats['rms_real_mm']:.1f}mm "
               f"(max {stats['max_real_mm']:.1f}) vs sim rms "
               f"{stats['rms_sim_mm']:.1f}mm (max {stats['max_sim_mm']:.1f}).")


# -- routes -----------------------------------------------------------------


def _file_lists():
  trajs = list(reversed(tio.find_trajectory_files(PROJECT_DIR)))
  tapes = sorted(glob.glob(os.path.join(DEPLOY_DIR, "tape_*.npz")),
                 key=os.path.getmtime, reverse=True)
  return ([os.path.basename(t) for t in trajs],
          [os.path.basename(t) for t in tapes])


def _resolve_traj(name):
  """Map a UI-supplied trajectory basename to its full path under
  datasets/trajectories (falling back to the project root)."""
  if not name:
    return None
  base = os.path.basename(name)
  for p in tio.find_trajectory_files(PROJECT_DIR):
    if os.path.basename(p) == base:
      return p
  return os.path.join(PROJECT_DIR, base)


@app.route("/status")
def status():
  trajs, tapes = _file_lists()
  with state_lock:
    s = dict(state)
  s["trajectories"] = trajs
  s["tapes"] = tapes
  s["calibration"] = _load_calibration()
  s["settings"] = _load_settings()
  with recorder.lock:
    s["camera_on"] = recorder.running
    s["camera_status"] = recorder.status
    s["camera_recording"] = recorder.recording
    s["last_recorded"] = recorder.last_saved
  return jsonify(s)


@app.route("/detect_cameras", methods=["POST"])
def detect_cameras():
  found = []
  for i in range(5):
    cap = cv.VideoCapture(i, cv.CAP_DSHOW)
    if cap.isOpened():
      found.append(i)
    cap.release()
  return jsonify({"found": found})


@app.route("/camera", methods=["POST"])
def camera():
  d = request.get_json(force=True)
  if d.get("action") == "start":
    recorder.start(int(d.get("index", 2)))
  else:
    recorder.stop()
  return jsonify({"ok": True})


@app.route("/record", methods=["POST"])
def record():
  recorder.toggle_record()
  return jsonify({"ok": True})


@app.route("/camera_feed")
def camera_feed():
  def gen():
    while True:
      with recorder.lock:
        if not recorder.running:
          break
        jpg = recorder.frame_jpeg
      if jpg:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
      time.sleep(0.05)
  return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/tape", methods=["POST"])
def tape():
  d = request.get_json(force=True)
  traj = _resolve_traj(d.get("trajectory"))
  model = os.path.join(PROJECT_DIR, "models", "rl_policy.zip")
  ok = _run_in_thread("tape", _tape_worker, traj, model)
  return jsonify({"started": ok})


@app.route("/jog", methods=["POST"])
def jog():
  d = request.get_json(force=True)
  _save_settings(d)
  return jsonify({"started": _run_in_thread("jog", _jog_worker, d)})


@app.route("/calibrate", methods=["POST"])
def calibrate():
  d = request.get_json(force=True)
  _save_settings(d)
  return jsonify({"started": _run_in_thread("calibrate", _calibrate_worker, d)})


@app.route("/drive", methods=["POST"])
def drive():
  d = request.get_json(force=True)
  _save_settings(d)
  if not d.get("tape"):
    return jsonify({"started": False, "error": "no tape selected"})
  tape_path = os.path.join(DEPLOY_DIR, d["tape"])
  ok = _run_in_thread("drive", _drive_worker, d,
                      tape_path, float(d.get("speed_scale", 0.5)),
                      float(d.get("countdown", 8)))
  return jsonify({"started": ok})


@app.route("/stop", methods=["POST"])
def stop():
  stop_evt.set()
  return jsonify({"ok": True})


@app.route("/compare", methods=["POST"])
def compare():
  d = request.get_json(force=True)
  if not d.get("drawn"):
    return jsonify({"started": False, "error": "no drawn file selected"})
  drawn = _resolve_traj(d["drawn"])
  tape_path = os.path.join(DEPLOY_DIR, d["tape"]) if d.get("tape") else None
  return jsonify({"started": _run_in_thread("compare", _compare_worker,
                                            drawn, tape_path)})


@app.route("/compare_plot")
def compare_plot():
  path = os.path.join(DEPLOY_DIR, "openloop_gap.png")
  if not os.path.exists(path):
    return ("no plot yet", 404)
  return send_file(path, mimetype="image/png")


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Open-loop policy deployment</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f4f2; color: #222; }
  header { background: #1f2937; color: #fff; padding: 12px 20px; font-size: 18px; font-weight: 600; }
  #msg { padding: 10px 20px; background: #fffbe6; border-bottom: 1px solid #e5e0c8;
         font-size: 14px; min-height: 20px; }
  main { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
         gap: 16px; padding: 16px 20px; max-width: 1500px; }
  section { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  h2 { margin: 0 0 4px; font-size: 15px; }
  .hint { color: #666; font-size: 12.5px; margin: 0 0 10px; }
  label { display: block; font-size: 13px; margin: 8px 0 2px; color: #444; }
  select, input[type=text], input[type=number] { width: 100%; padding: 6px; font-size: 13px;
         border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
  .chk { font-size: 13px; display: flex; align-items: center; gap: 4px; }
  button { padding: 8px 14px; font-size: 13.5px; border: 0; border-radius: 7px;
           background: #2563eb; color: #fff; cursor: pointer; margin-top: 10px; }
  button:disabled { background: #9ab; cursor: not-allowed; }
  button.warn { background: #dc2626; }
  button.go { background: #059669; font-size: 15px; padding: 10px 18px; }
  .out { font-family: ui-monospace, monospace; font-size: 12.5px; background: #f8f8f6;
         border-radius: 6px; padding: 8px; margin-top: 10px; white-space: pre-wrap; }
  #bar { height: 10px; background: #e5e7eb; border-radius: 5px; margin-top: 10px; overflow: hidden; }
  #barfill { height: 100%; width: 0; background: #059669; transition: width .3s; }
  img { max-width: 100%; border-radius: 6px; margin-top: 10px; }
</style></head><body>
<header>Open-loop policy deployment &mdash; LEGO signature car</header>
<div id="msg">Loading...</div>
<main>

<section style="grid-column: 1 / -1;">
  <h2>0 &middot; Record a signature (camera)</h2>
  <p class="hint">Point the overhead camera at the ArUco sheet, start it, then Record
  while you draw with the red-tipped pencil. The saved trajectory appears in the Tape
  panel automatically. Tip: if the robot's pencil also carries the red marker, hit
  Record again right before DRIVE to capture the robot's own trace for the Compare panel.</p>
  <div class="row">
    <span>Camera index</span>
    <input type="number" id="cam_index" value="2" min="0" max="9" style="width:70px">
    <button id="btn-cam" onclick="post('/camera', {action: camOn ? 'stop' : 'start', index: v('cam_index')})">Start camera</button>
    <button id="btn-rec" onclick="post('/record', {})">Record</button>
    <button onclick="detectCams()">Detect cameras</button>
    <span class="out" id="cam-status" style="margin-top:0">camera off</span>
  </div>
  <div class="out" id="rec-out" style="display:none"></div>
  <img id="cam-img" style="display:none; max-width: 640px;">
</section>

<section>
  <h2>1 &middot; Tape</h2>
  <p class="hint">Roll out the trained policy in simulation on a recorded signature;
  its wheel-speed commands become the replay tape.</p>
  <label>Signature trajectory</label>
  <select id="trajectory"></select>
  <button id="btn-tape" onclick="post('/tape', {trajectory: v('trajectory')})">Generate tape</button>
  <div class="out" id="tape-out">no tape generated this session</div>
</section>

<section>
  <h2>2 &middot; Robot setup</h2>
  <p class="hint">Jog identifies which motor is which (watch the wheels).
  Calibrate needs the wheels OFF the ground.</p>
  <label>Card serial</label>
  <input type="text" id="card_serial" placeholder="e.g. 12345">
  <label>Card color (optional)</label>
  <input type="text" id="card_color" placeholder="e.g. BLUE">
  <div class="row">
    <span class="chk"><input type="checkbox" id="swap_motors"><label for="swap_motors">swap motors</label></span>
    <span class="chk"><input type="checkbox" id="invert_left"><label for="invert_left">invert left</label></span>
    <span class="chk"><input type="checkbox" id="invert_right"><label for="invert_right">invert right</label></span>
  </div>
  <div class="row">
    <button id="btn-jog" onclick="post('/jog', hw())">Jog motors</button>
    <button id="btn-cal" onclick="if(confirm('Wheels off the ground?')) post('/calibrate', hw())">Calibrate</button>
  </div>
  <div class="out" id="cal-out">not calibrated</div>
</section>

<section>
  <h2>3 &middot; Drive (open loop)</h2>
  <p class="hint">During the countdown, place the pencil tip on the path's start
  point with the car facing along the path.</p>
  <label>Tape</label>
  <select id="tape"></select>
  <label>Speed scale (start at 0.5)</label>
  <input type="number" id="speed_scale" value="0.5" min="0.1" max="1" step="0.05">
  <label>Countdown (s)</label>
  <input type="number" id="countdown" value="8" min="0" max="60" step="1">
  <div class="row">
    <button class="go" id="btn-drive" onclick="post('/drive', Object.assign(hw(), {tape: v('tape'), speed_scale: v('speed_scale'), countdown: v('countdown')}))">DRIVE</button>
    <button class="warn" id="btn-stop" onclick="post('/stop', {})">STOP</button>
  </div>
  <div id="bar"><div id="barfill"></div></div>
  <div class="out" id="drive-out">no drive yet</div>
</section>

<section>
  <h2>4 &middot; Compare</h2>
  <p class="hint">After the drive: record the drawn line with the camera pipeline
  (webapp.py, or record_trajectory.py + coordinate_plane.py). The new npz will
  appear here.</p>
  <label>Drawn-line npz</label>
  <select id="drawn"></select>
  <label>Against tape</label>
  <select id="tape2"></select>
  <button id="btn-cmp" onclick="post('/compare', {drawn: v('drawn'), tape: v('tape2')})">Compare</button>
  <div class="out" id="cmp-out">no comparison yet</div>
  <img id="cmp-img" style="display:none">
</section>

</main>
<script>
const $ = id => document.getElementById(id);
const v = id => $(id).type === 'checkbox' ? $(id).checked : $(id).value;
function hw() { return {card_serial: v('card_serial'), card_color: v('card_color'),
  swap_motors: v('swap_motors'), invert_left: v('invert_left'),
  invert_right: v('invert_right'), speed_scale: v('speed_scale'), countdown: v('countdown')}; }
async function post(url, body) {
  try {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)});
    if (!r.ok) { $('msg').textContent = 'Server error ' + r.status + ' on ' + url; return; }
    const j = await r.json();
    if (j.error) $('msg').textContent = 'Error: ' + j.error;
  } catch (e) { $('msg').textContent = 'Request failed (' + url + '): ' + e; }
}
async function detectCams() {
  $('cam-status').textContent = 'probing camera indices 0-4...';
  try {
    const r = await fetch('/detect_cameras', {method: 'POST'});
    const j = await r.json();
    $('cam-status').textContent = 'cameras found at index: ' +
      (j.found.length ? j.found.join(', ') : 'none') +
      ' - the overhead camera is usually the highest USB one';
  } catch (e) { $('cam-status').textContent = 'detect failed: ' + e; }
}
function fill(sel, items, keep) {
  const cur = sel.value;
  sel.innerHTML = '';
  for (const it of items) { const o = document.createElement('option'); o.value = o.textContent = it; sel.appendChild(o); }
  if (keep && items.includes(cur)) sel.value = cur;
}
let settingsLoaded = false;
let camOn = false;
async function tick() {
  try {
    const s = await (await fetch('/status')).json();
    $('msg').textContent = (s.busy ? '[' + s.task + '] ' : '') + s.message;
    if (s.camera_on !== camOn) {
      camOn = s.camera_on;
      $('btn-cam').textContent = camOn ? 'Stop camera' : 'Start camera';
      $('cam-img').style.display = camOn ? 'block' : 'none';
      $('cam-img').src = camOn ? '/camera_feed?ts=' + Date.now() : '';
    }
    $('cam-status').textContent = s.camera_status;
    $('btn-rec').textContent = s.camera_recording ? 'Stop & save' : 'Record';
    $('btn-rec').className = s.camera_recording ? 'warn' : '';
    if (s.last_recorded) {
      $('rec-out').style.display = 'block';
      $('rec-out').textContent = 'last recording saved: ' + s.last_recorded;
    }
    fill($('trajectory'), s.trajectories, true);
    fill($('tape'), s.tapes, true);
    fill($('tape2'), s.tapes, true);
    fill($('drawn'), s.trajectories, true);
    for (const b of ['btn-tape','btn-jog','btn-cal','btn-drive','btn-cmp'])
      $(b).disabled = s.busy;
    $('barfill').style.width = (s.progress ? (s.progress * 100).toFixed(0) : 0) + '%';
    if (s.tape_info) $('tape-out').textContent =
      `${s.tape_info.trajectory}\\n${s.tape_info.steps} steps, ${s.tape_info.duration_s.toFixed(1)}s, ` +
      `peak ${s.tape_info.max_wheel_degs.toFixed(0)} deg/s` +
      (s.tape_info.max_wheel_degs > 660 ? `\\nWARNING: exceeds ~660 deg/s motor max - use speed scale <= ${(660/s.tape_info.max_wheel_degs).toFixed(2)}` : '');
    if (s.calibration) $('cal-out').textContent =
      `${s.calibration.degs_per_100pct.toFixed(0)} deg/s per 100%  (${s.calibration.date.slice(0,16)})`;
    if (s.drive_summary && s.drive_summary.length) $('drive-out').textContent = s.drive_summary.join('\\n');
    if (s.compare_stats) {
      $('cmp-out').textContent =
        `real drawn : rms ${s.compare_stats.rms_real_mm.toFixed(2)} mm, max ${s.compare_stats.max_real_mm.toFixed(2)} mm\\n` +
        `sim expected: rms ${s.compare_stats.rms_sim_mm.toFixed(2)} mm, max ${s.compare_stats.max_sim_mm.toFixed(2)} mm`;
      $('cmp-img').src = '/compare_plot?ts=' + Date.now();
      $('cmp-img').style.display = 'block';
    }
    if (!settingsLoaded && s.settings) {
      settingsLoaded = true;
      if (s.settings.card_serial) $('card_serial').value = s.settings.card_serial;
      if (s.settings.card_color) $('card_color').value = s.settings.card_color;
      $('swap_motors').checked = !!s.settings.swap_motors;
      $('invert_left').checked = !!s.settings.invert_left;
      $('invert_right').checked = !!s.settings.invert_right;
      if (s.settings.speed_scale) $('speed_scale').value = s.settings.speed_scale;
      if (s.settings.countdown) $('countdown').value = s.settings.countdown;
    }
  } catch (e) { $('msg').textContent = 'status error: ' + e; }
}
setInterval(tick, 600);
tick();
</script>
</body></html>"""


@app.route("/")
def index():
  return PAGE


if __name__ == "__main__":
  print("Open-loop deployment dashboard: http://localhost:5050")
  app.run(host="127.0.0.1", port=5050, debug=False)
