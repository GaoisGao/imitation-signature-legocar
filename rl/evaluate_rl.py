"""evaluate_rl.py - run the trained SB3 RL policy deterministically on
recorded signatures, print tracking stats, save target-vs-tip plots, and
optionally watch it live in the MuJoCo viewer.

Usage:
    py -3.13 rl/evaluate_rl.py                  # all recorded signatures, plots only
    py -3.13 rl/evaluate_rl.py --view           # live MuJoCo window per signature
    py -3.13 rl/evaluate_rl.py --trajectory target_trajectory_20260710_111912.npz --view
    py -3.13 rl/evaluate_rl.py --from-fit       # on the measured hardware plant

IMPORTANT: the default plant is IDEAL (no wheel speed-loop lag), which flatters
a policy badly - every checkpoint to date finishes on the ideal plant and aborts
off-path under the measured lag. Pass --from-fit before believing any result, and
set --frame-skip to the value in the policy's models/<name>_config.json.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

RL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(RL_DIR)
for p in (PROJECT_DIR, RL_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

import track_trajectory as tt
import trajectory_io as tio
from signature_env import SignatureEnv


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", default=os.path.join(PROJECT_DIR, "models", "rl_policy.zip"))
  ap.add_argument("--trajectory", default=None,
                  help="One trajectory .npz; default: every recorded signature")
  ap.add_argument("--view", action="store_true",
                  help="Open a live MuJoCo viewer window while tracing")
  ap.add_argument("--max-time", type=float, default=60.0)
  ap.add_argument("--frame-skip", type=int, default=50,
                  help="Physics steps per policy action; MUST match the value the "
                       "policy was trained at (see models/<name>_config.json)")
  ap.add_argument("--vel-lag-tau", type=float, default=0.0,
                  help="Wheel speed-loop time constant (s). 0 (default) evaluates on "
                       "the IDEAL plant, which flatters the policy - pass the measured "
                       "tau to see what hardware will actually do")
  ap.add_argument("--vel-dead-time", type=float, default=0.0,
                  help="Wheel speed-loop dead time (s); pair with --vel-lag-tau")
  ap.add_argument("--from-fit", action="store_true",
                  help="Take --vel-lag-tau/--vel-dead-time from the measured "
                       "rl/deploy/sysid/sysid_fit_speed.json instead of the flags")
  ap.add_argument("--obs-delay", type=int, default=0,
                  help="Observation delay in control steps; match the policy's "
                       "training value (see its _config.json)")
  args = ap.parse_args()

  if args.from_fit:
    fit_path = os.path.join(RL_DIR, "deploy", "sysid", "sysid_fit_speed.json")
    with open(fit_path) as f:
      fit = json.load(f)
    args.vel_lag_tau, args.vel_dead_time = fit["tau_s"], fit["dead_s"]
    print(f"Plant from {os.path.basename(fit_path)}: "
          f"tau={args.vel_lag_tau:.3f}s dead={args.vel_dead_time:.3f}s")

  from stable_baselines3 import PPO
  from signature_env import scales_from_config
  model = PPO.load(args.model)
  # Same rule as deployment: the speed ceiling comes from the policy's own run
  # config, so a capped policy is not evaluated at the module default.
  v_max, omega_max = scales_from_config(args.model)
  print(f"Loaded RL policy: {args.model} "
        f"(v_max={v_max:.3f} m/s, omega_max={omega_max:.1f} rad/s)")

  if args.trajectory:
    files = [args.trajectory]
  else:
    files = tio.find_trajectory_files(PROJECT_DIR)
  if not files:
    raise SystemExit("No trajectory .npz files found.")

  for traj_path in files:
    name = os.path.splitext(os.path.basename(traj_path))[0]
    path_world = tt.load_path_world(traj_path)
    env = SignatureEnv([path_world], frame_skip=args.frame_skip,
                       init_xy_noise=0.0, init_yaw_noise=0.0,
                       vel_lag_tau=args.vel_lag_tau,
                       vel_dead_time=args.vel_dead_time,
                       v_max=v_max, omega_max=omega_max,
                       obs_delay_steps=args.obs_delay,
                       max_time=args.max_time)
    obs, _ = env.reset(seed=0)

    viewer_ctx = None
    if args.view:
      from mujoco import viewer as mj_viewer
      viewer_ctx = mj_viewer.launch_passive(env.tracker.m, env.tracker.d)

    done, info = False, {}
    while not done:
      action, _ = model.predict(obs, deterministic=True)
      obs, _, term, trunc, info = env.step(action)
      done = term or trunc
      if viewer_ctx is not None:
        viewer_ctx.sync()
        if not viewer_ctx.is_running():
          break
    if viewer_ctx is not None:
      viewer_ctx.close()

    tip = env.tracker.tip_history_array()
    errors = tt.compute_tracking_error_mm(tip[:, :2], env.path_world)
    out_png = os.path.join(RL_DIR, f"rl_eval_{name}.png")
    tt.plot_comparison(env.path_world, tip, out_png)
    status = "finished" if info.get("is_success") else "DID NOT FINISH"
    print(f"{name}: {status} in {env.tracker.elapsed_time:.1f}s sim, "
          f"rms={np.sqrt(np.mean(errors ** 2)):.2f}mm max={errors.max():.2f}mm "
          f"-> {os.path.basename(out_png)}")


if __name__ == "__main__":
  main()
