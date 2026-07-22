"""Train the LEGO car in mjlab on CPU (rsl-rl PPO over the batched
MuJoCo-Warp sim).

Checkpoints land in <project>/logs/rsl_rl/<experiment>/<timestamp>/model_*.pt
every `save_interval` iterations - point play_car.py --checkpoint at any of
them (or use --checkpoint latest) for hot-swappable viewing in the viser
browser viewer while training is still running.

Usage (from the project root):
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\train_car.py
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\train_car.py --num-envs 64 --max-iterations 500
    .venv-mjlab\\Scripts\\python.exe rl\\mjlab_port\\train_car.py --task drive
"""

# Must be set before mjlab/warp import: CPU Warp backend + UTF-8 stdio.
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", choices=("signature", "drive"), default="signature")
  ap.add_argument("--num-envs", type=int, default=32)
  ap.add_argument("--max-iterations", type=int, default=None,
                  help="Override the task's default PPO iteration count")
  ap.add_argument("--run-name", type=str, default=None,
                  help="Suffix for the log directory name")
  args = ap.parse_args()

  from mjlab.scripts.train import TrainConfig, launch_training
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import __init__ as lego_car_tasks  # registers the tasks

  task_id = (lego_car_tasks.SIGNATURE_TASK_ID if args.task == "signature"
             else lego_car_tasks.TASK_ID)

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(task_id)
  if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
  if args.run_name is not None:
    agent_cfg.run_name = args.run_name

  cfg = TrainConfig(
    env=env_cfg,
    agent=agent_cfg,
    gpu_ids=None,  # CPU
    log_root=os.path.join(PROJECT_DIR, "logs", "rsl_rl"),
  )
  print(f"Training {task_id}: {args.num_envs} envs, "
        f"{agent_cfg.max_iterations} iterations, CPU Warp backend")
  launch_training(task_id, cfg)


if __name__ == "__main__":
  main()
