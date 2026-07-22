"""evaluate_rl.py - run the trained SB3 RL policy deterministically on
recorded signatures, print tracking stats, save target-vs-tip plots, and
optionally watch it live in the MuJoCo viewer.

Usage:
    py -3.13 rl/evaluate_rl.py                  # all recorded signatures, plots only
    py -3.13 rl/evaluate_rl.py --view           # live MuJoCo window per signature
    py -3.13 rl/evaluate_rl.py --trajectory target_trajectory_20260710_111912.npz --view
"""

import argparse
import glob
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
  args = ap.parse_args()

  from stable_baselines3 import PPO
  model = PPO.load(args.model)
  print(f"Loaded RL policy: {args.model}")

  if args.trajectory:
    files = [args.trajectory]
  else:
    files = tio.find_trajectory_files(PROJECT_DIR)
  if not files:
    raise SystemExit("No trajectory .npz files found.")

  for traj_path in files:
    name = os.path.splitext(os.path.basename(traj_path))[0]
    path_world = tt.load_path_world(traj_path)
    env = SignatureEnv([path_world], init_xy_noise=0.0, init_yaw_noise=0.0,
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
