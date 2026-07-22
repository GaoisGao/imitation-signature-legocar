"""Open-loop deployment of the trained SB3 RL policy on the real LEGO car.

Open-loop = the policy runs ONCE in simulation, its wheel-speed commands are
recorded as a time-stamped "tape", and the real robot replays that tape with
no sensing. This measures the raw dynamics gap (motors, friction, geometry,
timing) in isolation - expect visible drift; that drift IS the measurement.
The closed-loop (camera-in-the-loop) deployment comes after this baseline.

Workflow (each stage is a subcommand):

  1. tape       Roll out the trained policy in MuJoCo, save the command tape.
                    py -3.13 rl/deploy/openloop_deploy.py tape
                    py -3.13 rl/deploy/openloop_deploy.py tape --trajectory target_trajectory_20260710_105402.npz
  2. jog        Identify motor wiring: spins the LEFT motor, then the RIGHT.
                Watch which wheel moves and which way; set --swap-motors /
                --invert-left / --invert-right on later commands to match.
                    py -3.13 rl/deploy/openloop_deploy.py jog --card-serial XXXX
  3. calibrate  Wheels OFF the ground: runs both motors at a fixed percent,
                reads the encoders, computes deg/s per 100% and saves it.
                    py -3.13 rl/deploy/openloop_deploy.py calibrate --card-serial XXXX
  4. drive      Replay the tape on the robot (10 Hz tank commands). Position
                the pencil tip on the path's start mark, facing the path
                direction, during the countdown. Logs encoders + IMU yaw.
                    py -3.13 rl/deploy/openloop_deploy.py drive --card-serial XXXX
                    py -3.13 rl/deploy/openloop_deploy.py drive --card-serial XXXX --speed-scale 0.5
  5. compare    After recording the drawn line with the usual camera pipeline
                (record_trajectory.py + coordinate_plane.py, or webapp.py),
                compare it against the target and the sim's expected trace.
                    py -3.13 rl/deploy/openloop_deploy.py compare --drawn trajectory_XXXX_paper.npz

The tape stores wheel speeds in deg/s; `drive` converts to percent via the
calibration constant (rl/deploy/motor_calibration.json, or --degs-per-100pct).
Percent commands that saturate at 100% are counted and reported - saturation
means the policy asks for more speed than the motors have.
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(DEPLOY_DIR)
PROJECT_DIR = os.path.dirname(RL_DIR)
for p in (PROJECT_DIR, RL_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

CALIBRATION_PATH = os.path.join(DEPLOY_DIR, "motor_calibration.json")
CONTROL_DT = 0.1          # s, replay tick (matches motor_dashboard.py / BLE latency)
DEFAULT_DEGS_PER_100 = 660.0  # SPIKE medium motor no-load speed, used until calibrated

# Paper frame offset: sim world frame is paper-centered; paper mm frame has
# its origin at the ID0 marker corner (see track_trajectory.mm_to_world).
PAPER_OFFSET_MM = np.array([199.0 / 2.0, 137.0 / 2.0])


# -- tape --------------------------------------------------------------------


def make_tape(model_path: str, trajectory_path: str = None, max_time: float = 60.0,
              output: str = None, log=print) -> dict:
  """Rolls out the trained policy in sim and saves the command tape.
  Returns a summary dict (tape_path, steps, duration_s, max_wheel_degs,
  finished, trajectory). Reusable by both the CLI and the dashboard UI."""
  import torch  # noqa: F401  (SB3 needs it)
  from stable_baselines3 import PPO

  import track_trajectory as tt
  from signature_env import SignatureEnv

  trajectory_path = trajectory_path or tt.find_latest_trajectory_file(PROJECT_DIR)
  if trajectory_path is None:
    raise RuntimeError("No trajectory .npz found.")
  log(f"Trajectory: {trajectory_path}")
  path_world = tt.load_path_world(trajectory_path)

  model = PPO.load(model_path)
  log(f"Policy: {model_path}")

  env = SignatureEnv([path_world], init_xy_noise=0.0, init_yaw_noise=0.0,
                     max_time=max_time)
  obs, _ = env.reset(seed=0)
  tr = env.tracker

  t_list, v_list, om_list, wl_list, wr_list = [], [], [], [], []
  done = False
  info = {}
  while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, info = env.step(action)
    done = term or trunc
    v, om = env._cmd
    # Same differential-drive conversion as SignatureTracker.step, using the
    # model's own geometry. deg/s at the wheel.
    wl = (v - om * tr.wheel_left_y) / tr.wheel_radius * 180.0 / np.pi
    wr = (v - om * tr.wheel_right_y) / tr.wheel_radius * 180.0 / np.pi
    t_list.append(env._elapsed * tr.m.opt.timestep * env.frame_skip)
    v_list.append(v)
    om_list.append(om)
    wl_list.append(wl)
    wr_list.append(wr)

  finished = bool(info.get("is_success"))
  if not finished:
    log("WARNING: the sim rollout did NOT finish cleanly - tape saved anyway.")

  t = np.array(t_list)
  wl = np.array(wl_list)
  wr = np.array(wr_list)
  tip = tr.tip_history_array()                      # (N, 3) world m + t
  motor = tr.motor_history_array()                  # (N, 4) t, yaw, wl_pos, wr_pos
  tip_mm = tip[:, :2] * 1000.0 + PAPER_OFFSET_MM    # expected trace, paper mm
  target_mm = path_world * 1000.0 + PAPER_OFFSET_MM

  name = os.path.splitext(os.path.basename(trajectory_path))[0]
  out = output or os.path.join(DEPLOY_DIR, f"tape_{name}.npz")
  np.savez(out, t=t, v=np.array(v_list), omega=np.array(om_list),
           wheel_left_degs=wl, wheel_right_degs=wr,
           sim_yaw_deg=np.interp(t, motor[:, 0], motor[:, 1]),
           sim_left_pos_deg=np.interp(t, motor[:, 0], motor[:, 2]),
           sim_right_pos_deg=np.interp(t, motor[:, 0], motor[:, 3]),
           expected_tip_mm=tip_mm, target_mm=target_mm,
           trajectory=os.path.basename(trajectory_path))
  dur = float(t[-1]) if len(t) else 0.0
  max_degs = float(max(np.abs(wl).max(), np.abs(wr).max())) if len(t) else 0.0
  log(f"Tape: {len(t)} steps, {dur:.1f}s, wheel speed max {max_degs:.0f} deg/s")
  log(f"Saved tape to {out}")
  return {"tape_path": out, "steps": len(t), "duration_s": dur,
          "max_wheel_degs": max_degs, "finished": finished,
          "trajectory": os.path.basename(trajectory_path)}


def cmd_tape(args):
  make_tape(args.model, args.trajectory, args.max_time, args.output)


# -- hardware helpers -----------------------------------------------------------


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


def _positions(dm, le, args):
  lp = float(dm.motor[le.MOTOR_LEFT].position)
  rp = float(dm.motor[le.MOTOR_RIGHT].position)
  if args.swap_motors:
    lp, rp = rp, lp
  return lp * (-1 if args.invert_left else 1), rp * (-1 if args.invert_right else 1)


def _tank(dm, args, left_pct, right_pct):
  if args.invert_left:
    left_pct = -left_pct
  if args.invert_right:
    right_pct = -right_pct
  if args.swap_motors:
    left_pct, right_pct = right_pct, left_pct
  dm.movement_move_tank(left_pct, right_pct)


def cmd_jog(args):
  dm, le = _connect(args)
  try:
    print("Spinning the LEFT motor forward for 2s - watch which wheel moves...")
    dm.motor_set_speed(30, motor=le.MOTOR_LEFT)
    dm.motor_run(motor=le.MOTOR_LEFT)
    time.sleep(2)
    dm.motor_stop(motor=le.MOTOR_LEFT)
    time.sleep(1)
    print("Spinning the RIGHT motor forward for 2s...")
    dm.motor_set_speed(30, motor=le.MOTOR_RIGHT)
    dm.motor_run(motor=le.MOTOR_RIGHT)
    time.sleep(2)
    dm.motor_stop(motor=le.MOTOR_RIGHT)
    print("If 'LEFT' moved the robot's right wheel, add --swap-motors to "
          "calibrate/drive. If a wheel spun backward for a forward command, "
          "add --invert-left / --invert-right.")
  finally:
    dm.movement_stop()
    dm.disconnect()


def cmd_calibrate(args):
  dm, le = _connect(args)
  try:
    input(f"Lift the car so BOTH wheels are off the ground, then press Enter. "
          f"Motors will run at {args.pct:.0f}% for {args.seconds:.0f}s...")
    l0, r0 = _positions(dm, le, args)
    _tank(dm, args, args.pct, args.pct)
    t0 = time.time()
    time.sleep(args.seconds)
    dm.movement_stop()
    elapsed = time.time() - t0
    l1, r1 = _positions(dm, le, args)
    left_dps = (l1 - l0) / elapsed
    right_dps = (r1 - r0) / elapsed
    degs_per_100 = (abs(left_dps) + abs(right_dps)) / 2.0 * (100.0 / args.pct)
    print(f"left: {left_dps:.1f} deg/s, right: {right_dps:.1f} deg/s at {args.pct:.0f}% "
          f"-> {degs_per_100:.1f} deg/s per 100%")
    if left_dps * right_dps < 0 or left_dps < 0:
      print("WARNING: a motor ran backward for a positive command - check the "
            "--invert flags (see jog) and re-run calibrate WITH those flags.")
    with open(CALIBRATION_PATH, "w") as f:
      json.dump({"degs_per_100pct": degs_per_100,
                 "left_dps": left_dps, "right_dps": right_dps,
                 "pct": args.pct, "date": datetime.now().isoformat()}, f, indent=2)
    print(f"Saved calibration to {CALIBRATION_PATH}")
  finally:
    dm.movement_stop()
    dm.disconnect()


def _load_degs_per_100(args) -> float:
  if args.degs_per_100pct is not None:
    return args.degs_per_100pct
  if os.path.exists(CALIBRATION_PATH):
    with open(CALIBRATION_PATH) as f:
      cal = json.load(f)
    print(f"Using calibration: {cal['degs_per_100pct']:.1f} deg/s per 100% "
          f"(from {cal['date']})")
    return float(cal["degs_per_100pct"])
  print(f"No calibration file - using default {DEFAULT_DEGS_PER_100} deg/s per "
        f"100% (SPIKE medium no-load spec). Run `calibrate` for accuracy.")
  return DEFAULT_DEGS_PER_100


def _find_latest_tape():
  tapes = glob.glob(os.path.join(DEPLOY_DIR, "tape_*.npz"))
  if not tapes:
    raise SystemExit("No tape found. Run the `tape` subcommand first.")
  return max(tapes, key=os.path.getmtime)


def cmd_drive(args):
  tape_path = args.tape or _find_latest_tape()
  tape = np.load(tape_path, allow_pickle=True)
  t_tape = tape["t"]
  duration = float(t_tape[-1]) / args.speed_scale
  degs_per_100 = _load_degs_per_100(args)
  print(f"Tape: {tape_path} ({float(t_tape[-1]):.1f}s sim, replay {duration:.1f}s "
        f"at speed-scale {args.speed_scale})")

  dm, le = _connect(args)
  saturated = 0
  log = []
  try:
    print(f"Position the PENCIL TIP on the path start, car facing along the "
          f"path. Starting in {args.countdown:.0f}s...")
    time.sleep(args.countdown)
    dm.imu_reset_yaw_axis(0)
    dm.motor_reset_relative_position()
    l_off, r_off = _positions(dm, le, args)

    start = time.time()
    print("Driving (open loop - Ctrl+C to abort)...")
    while True:
      t = time.time() - start
      if t >= duration:
        break
      # speed_scale < 1 stretches time and slows the wheels by the same factor
      t_sim = t * args.speed_scale
      wl = float(np.interp(t_sim, t_tape, tape["wheel_left_degs"])) * args.speed_scale
      wr = float(np.interp(t_sim, t_tape, tape["wheel_right_degs"])) * args.speed_scale
      lp_cmd = wl / degs_per_100 * 100.0
      rp_cmd = wr / degs_per_100 * 100.0
      if abs(lp_cmd) > 100 or abs(rp_cmd) > 100:
        saturated += 1
      _tank(dm, args, float(np.clip(lp_cmd, -100, 100)),
            float(np.clip(rp_cmd, -100, 100)))

      lp, rp = _positions(dm, le, args)
      yaw = float(dm.imu_device.yaw)
      log.append((t, lp - l_off, rp - r_off, yaw, lp_cmd, rp_cmd))

      tick_elapsed = (time.time() - start) - t
      time.sleep(max(0.0, CONTROL_DT - tick_elapsed))
  except KeyboardInterrupt:
    print("Aborted.")
  finally:
    dm.movement_stop()
    dm.disconnect()

  log = np.array(log)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  out = os.path.join(DEPLOY_DIR, f"drive_log_{ts}.npz")
  np.savez(out, log=log, tape_file=os.path.basename(tape_path),
           speed_scale=args.speed_scale, degs_per_100pct=degs_per_100,
           columns="t,left_pos_deg,right_pos_deg,yaw_deg,left_cmd_pct,right_cmd_pct")
  print(f"Saved drive log to {out}")

  if saturated:
    print(f"WARNING: {saturated} ticks saturated at 100% - the policy asked for "
          f"more speed than the motors have. Use --speed-scale to derate.")
  if len(log):
    t_end = log[-1, 0] * args.speed_scale
    sim_l = float(np.interp(t_end, t_tape, tape["sim_left_pos_deg"]))
    sim_r = float(np.interp(t_end, t_tape, tape["sim_right_pos_deg"]))
    sim_y = float(np.interp(t_end, t_tape, tape["sim_yaw_deg"]))
    print("\nOpen-loop summary (real vs sim reference):")
    print(f"  left wheel : {log[-1, 1]:8.1f} deg   vs {sim_l:8.1f} deg")
    print(f"  right wheel: {log[-1, 2]:8.1f} deg   vs {sim_r:8.1f} deg")
    print(f"  final yaw  : {log[-1, 3]:8.1f} deg   vs {sim_y:8.1f} deg "
          f"(IMU positive direction may be opposite - check sign)")
    print("\nNext: record the drawn line with the camera pipeline "
          "(record_trajectory.py then coordinate_plane.py, or webapp.py), then run "
          "the `compare` subcommand with that npz.")


def run_compare(drawn: str, tape_path: str = None, output: str = None,
                log=print) -> dict:
  """Compares the physically drawn trace against the tape's target and the
  sim-expected trace. Returns stats + plot path. Reused by CLI and dashboard."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  import track_trajectory as tt

  tape_path = tape_path or _find_latest_tape()
  tape = np.load(tape_path, allow_pickle=True)
  target_mm = tape["target_mm"]
  expected_mm = tape["expected_tip_mm"]

  drawn_mm, _ = tt.load_trajectory_mm(drawn)

  err_real = tt.compute_tracking_error_mm(drawn_mm / 1000.0, target_mm / 1000.0)
  err_sim = tt.compute_tracking_error_mm(expected_mm / 1000.0, target_mm / 1000.0)

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.plot(target_mm[:, 0], target_mm[:, 1], ":", color="steelblue", lw=1.5,
          label="target")
  ax.plot(expected_mm[:, 0], expected_mm[:, 1], "-.", color="darkorange", lw=1.2,
          label=f"sim expected (rms {np.sqrt(np.mean(err_sim ** 2)):.1f}mm)")
  ax.plot(drawn_mm[:, 0], drawn_mm[:, 1], "-", color="crimson", lw=1.4,
          label=f"real drawn (rms {np.sqrt(np.mean(err_real ** 2)):.1f}mm)")
  ax.set_xlabel("x (mm)")
  ax.set_ylabel("y (mm)")
  ax.set_title("Open-loop sim-to-real gap")
  ax.set_aspect("equal")
  ax.legend()
  fig.tight_layout()
  out = output or os.path.join(DEPLOY_DIR, "openloop_gap.png")
  fig.savefig(out, dpi=150)
  plt.close(fig)
  stats = {
    "rms_real_mm": float(np.sqrt(np.mean(err_real ** 2))),
    "max_real_mm": float(err_real.max()),
    "rms_sim_mm": float(np.sqrt(np.mean(err_sim ** 2))),
    "max_sim_mm": float(err_sim.max()),
    "plot_path": out,
  }
  log(f"real  drawn vs target: rms {stats['rms_real_mm']:.2f}mm  "
      f"max {stats['max_real_mm']:.2f}mm")
  log(f"sim expected vs target: rms {stats['rms_sim_mm']:.2f}mm  "
      f"max {stats['max_sim_mm']:.2f}mm")
  log(f"Saved comparison plot to {out}")
  return stats


def cmd_compare(args):
  run_compare(args.drawn, args.tape, args.output)


# -- CLI -----------------------------------------------------------------------


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)

  p = sub.add_parser("tape", help="Roll out the policy in sim, save command tape")
  p.add_argument("--model", default=os.path.join(PROJECT_DIR, "models", "rl_policy.zip"))
  p.add_argument("--trajectory", default=None)
  p.add_argument("--max-time", type=float, default=60.0)
  p.add_argument("--output", default=None)

  def hw_flags(p):
    p.add_argument("--card-serial", required=True)
    p.add_argument("--card-color", default=None)
    p.add_argument("--swap-motors", action="store_true")
    p.add_argument("--invert-left", action="store_true")
    p.add_argument("--invert-right", action="store_true")

  p = sub.add_parser("jog", help="Identify motor wiring/direction")
  hw_flags(p)

  p = sub.add_parser("calibrate", help="Measure deg/s per percent (wheels lifted)")
  hw_flags(p)
  p.add_argument("--pct", type=float, default=30.0)
  p.add_argument("--seconds", type=float, default=3.0)

  p = sub.add_parser("drive", help="Replay the tape on the robot (open loop)")
  hw_flags(p)
  p.add_argument("--tape", default=None, help="Tape npz (default: newest in rl/deploy)")
  p.add_argument("--speed-scale", type=float, default=1.0,
                 help="<1 slows the replay (and wheels) by this factor; start at 0.5")
  p.add_argument("--countdown", type=float, default=8.0)
  p.add_argument("--degs-per-100pct", type=float, default=None,
                 help="Override the motor calibration constant")

  p = sub.add_parser("compare", help="Compare the drawn line to target + sim trace")
  p.add_argument("--drawn", required=True,
                 help="Paper-frame npz of the drawn line (trajectory_*_paper.npz "
                      "or target_trajectory_*.npz from the camera pipeline)")
  p.add_argument("--tape", default=None)
  p.add_argument("--output", default=None)

  args = ap.parse_args()
  {"tape": cmd_tape, "jog": cmd_jog, "calibrate": cmd_calibrate,
   "drive": cmd_drive, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
  main()
