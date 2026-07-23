"""Evaluate a trained mjlab signature policy: report the tip tracking error over
a batch of envs, optionally UNDER observation noise, so we can compare a nominal
vs a DR-trained checkpoint's robustness.

Set LEGOCAR_EVAL_NOISE=1 to inject the same observation noise the DR run trained
with (needs LEGOCAR_DR on, the default). Terminations are disabled here so the
metric reflects SUSTAINED tracking - envs that stray stay strayed instead of
resetting and hiding the failure.

Usage (from the project root, mjlab venv):
    # clean (no noise)
    python rl/mjlab_port/eval_signature.py --checkpoint logs/rsl_rl/lego_car_signature/<run>/model_299.pt
    # under observation noise
    LEGOCAR_EVAL_NOISE=1 python rl/mjlab_port/eval_signature.py --checkpoint <...>/model_299.pt
"""

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", required=True, help="Path to a model_*.pt")
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--steps", type=int, default=600, help="Control steps to roll out")
  ap.add_argument("--warmup", type=int, default=100,
                  help="Skip this many initial steps before measuring")
  ap.add_argument("--cpu", action="store_true")
  ap.add_argument("--gpu-id", type=int, default=0)
  args = ap.parse_args()

  if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  device = "cpu" if args.cpu else f"cuda:{args.gpu_id}"

  import torch
  from dataclasses import asdict

  # Same symbols run_play uses (all re-exported from mjlab.scripts.play).
  from mjlab.scripts.play import (
    ManagerBasedRlEnv,
    MjlabOnPolicyRunner,
    RslRlVecEnvWrapper,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
  )
  import __init__ as lego_car_tasks  # registers the tasks

  task_id = lego_car_tasks.SIGNATURE_TASK_ID
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = args.num_envs
  # Eval-only: disable terminations (sustained tracking - no resets hiding strays)
  # AND rewards. Rewards are unused here, and the finish/off_path reward terms
  # reference the now-removed termination terms (else KeyError at step). We only
  # read the command term's err_mm.
  env_cfg.terminations = {}
  env_cfg.rewards = {}
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.unwrapped.command_manager.get_term("signature_path")
  noise_on = os.environ.get("LEGOCAR_EVAL_NOISE", "0") not in ("0", "", "false", "False")

  def get_obs():
    res = env.get_observations()
    return res[0] if isinstance(res, (tuple, list)) else res

  obs = get_obs()
  errs = []
  with torch.inference_mode():
    for t in range(args.steps):
      actions = policy(obs)
      obs = env.step(actions)[0]
      if t >= args.warmup:
        errs.append(cmd.err_mm.detach().float().clone())

  E = torch.stack(errs).flatten()
  print("=" * 64)
  print(f"checkpoint : {args.checkpoint}")
  print(f"obs noise  : {'ON' if noise_on else 'OFF'}   "
        f"envs={args.num_envs}  steps={args.steps} (warmup {args.warmup})")
  print(f"tracking_err_mm  mean={E.mean():.3f}  median={E.median():.3f}  "
        f"p95={E.quantile(0.95):.3f}  max={E.max():.3f}")
  print(f"fraction strayed (>20mm) = {(E > 20).float().mean():.3f}")
  print("=" * 64)
  env.close()


if __name__ == "__main__":
  main()
