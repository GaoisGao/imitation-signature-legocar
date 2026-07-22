"""Launch a fleet of LEGO signature cars in mjlab's browser (viser) viewer.

Runs entirely on CPU (Warp CPU backend) - no NVIDIA GPU needed. Open the
printed URL (http://localhost:8080) in a browser for the interactive 3D
scene with all cars.

Usage (from the project root, using the mjlab venv):
    # scripted fleet, drive task
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\play_car.py
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\play_car.py --num-envs 32 --agent random

    # signature task with a trained checkpoint (hot-swappable in the viewer:
    # a checkpoint dropdown lists every model_*.pt in the same folder, so you
    # can watch progress WHILE train_car.py is still running)
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\play_car.py --task signature --checkpoint latest
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\play_car.py --task signature ^
        --checkpoint logs\\rsl_rl\\lego_car_signature\\<timestamp>\\model_299.pt

Scripted agents (no checkpoint needed):
    circle (default) - per-car constant differential-drive commands
    random           - uniform random wheel efforts
    zero             - cars sit still
"""

# Environment MUST be set before mjlab/warp import: force the CPU Warp
# backend (this machine's GPU is not CUDA) and UTF-8 stdio (mjlab prints
# emoji, which crashes on a GBK console).
import glob
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_latest_checkpoint(experiment: str) -> str:
  pattern = os.path.join(PROJECT_DIR, "logs", "rsl_rl", experiment, "*", "model_*.pt")
  candidates = glob.glob(pattern)
  if not candidates:
    raise SystemExit(f"No checkpoints found under {pattern}. Run train_car.py first.")
  return max(candidates, key=os.path.getmtime)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", choices=("drive", "signature"), default="drive")
  ap.add_argument("--num-envs", type=int, default=32)
  ap.add_argument("--agent", choices=("circle", "random", "zero"), default="circle",
                  help="Scripted agent (ignored when --checkpoint is given)")
  ap.add_argument("--checkpoint", type=str, default=None,
                  help="Path to a trained rsl-rl model_*.pt, or 'latest' to pick "
                       "the newest one for the task. Enables the viewer's "
                       "hot-swappable checkpoint dropdown.")
  ap.add_argument("--device", type=str, default="cpu")
  ap.add_argument("--trajectory", type=str, default=None,
                  help="Signature-task only: restrict every env to this one "
                       "trajectory file (target_trajectory_*.npz or "
                       "trajectory_*_paper.npz basename)")
  args = ap.parse_args()

  if args.trajectory:
    os.environ["LEGOCAR_TRAJECTORY"] = args.trajectory

  import __init__ as lego_car_tasks  # registers the tasks

  task_id = (lego_car_tasks.SIGNATURE_TASK_ID if args.task == "signature"
             else lego_car_tasks.TASK_ID)

  if args.checkpoint is not None:
    # Trained mode: reuse mjlab's own play pipeline, which wires up the viser
    # viewer AND the CheckpointManager (hot-swap dropdown over sibling *.pt).
    from mjlab.scripts.play import PlayConfig, run_play

    checkpoint = args.checkpoint
    if checkpoint == "latest":
      experiment = ("lego_car_signature" if args.task == "signature" else "lego_car")
      checkpoint = find_latest_checkpoint(experiment)
    print(f"Loading checkpoint: {checkpoint}")
    run_play(task_id, PlayConfig(
      agent="trained",
      checkpoint_file=checkpoint,
      num_envs=args.num_envs,
      device=args.device,
      viewer="viser",
      log_root=os.path.join(PROJECT_DIR, "logs", "rsl_rl"),
    ))
    return

  # Scripted mode: build the env directly and drive it with a simple policy.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.viewer import ViserPlayViewer

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = args.num_envs

  print(f"Building {args.num_envs} LEGO car envs on device '{args.device}' "
        f"(Warp CPU kernels compile on first run - allow a couple of minutes)...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  action_shape = env.unwrapped.action_space.shape  # (num_envs, 2)
  n = action_shape[0]

  if args.agent == "zero":
    def policy(obs):
      del obs
      return torch.zeros(action_shape, device=env.unwrapped.device)
  elif args.agent == "random":
    def policy(obs):
      del obs
      return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
  else:  # circle
    # Constant per-car (left, right) efforts: shared forward component plus
    # a per-car differential sweeping from left turns to right turns.
    base = 0.55
    diff = torch.linspace(-0.35, 0.35, n, device=env.unwrapped.device)
    efforts = torch.stack([base + diff, base - diff], dim=1)

    def policy(obs):
      del obs
      return efforts

  print("Starting viser viewer (browser UI at http://localhost:8080)...")
  ViserPlayViewer(env, policy).run()
  env.close()


if __name__ == "__main__":
  main()
