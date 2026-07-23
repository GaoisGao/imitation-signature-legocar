"""
train_rl.py - train a PPO policy on rl/signature_env.SignatureEnv with
CPU-parallel environments (SubprocVecEnv: one MuJoCo instance per process).

Optionally warm-starts the PPO policy from the behavior-cloning checkpoint
(models/bc_policy.pt): the BC net is the same 4 -> 64 -> 64 -> 2 MLP shape, so
its weights are copied into PPO's policy mean network, with the BC observation
normalizer folded into the first layer and the env's action scaling folded
into the last layer - the warm-started policy's deterministic output is
numerically identical to the BC policy's before any RL update.

Usage:
    py -3.13 rl/train_rl.py
    py -3.13 rl/train_rl.py --warm-start models/bc_policy.pt --domain-rand
    py -3.13 rl/train_rl.py --num-envs 12 --total-timesteps 2000000
    py -3.13 -m tensorboard --logdir rl/runs     # training curves
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

from signature_env import (ACTION_SCALE, OBS_SCALE, RL_DIR, PROJECT_DIR,
                           SignatureEnv)

import track_trajectory as tt  # signature_env put PROJECT_DIR on sys.path
import trajectory_io as tio

LEARNING_DIR = os.path.join(PROJECT_DIR, "learning")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")


def find_trajectory_files():
    return tio.find_trajectory_files(PROJECT_DIR)


def make_env_fn(path_worlds, args, rank: int):
    def _init():
        env = SignatureEnv(
            path_worlds, frame_skip=args.frame_skip,
            init_xy_noise=args.init_xy_noise_mm / 1000.0,
            init_yaw_noise=np.radians(args.init_yaw_noise_deg),
            domain_rand=args.domain_rand,
            w_progress=args.w_progress, w_track=args.w_track,
            err_gate_mm=args.err_gate_mm,
            w_action_rate=args.w_action_rate, w_time=args.w_time,
            completion_bonus=args.completion_bonus,
            off_path_penalty=args.off_path_penalty,
            off_path_limit_mm=args.off_path_limit_mm,
            obs_noise_std=args.obs_noise,
            max_time=args.max_time)
        env.reset(seed=args.seed + rank)
        return env
    return _init


def warm_start_from_bc(policy, bc_ckpt_path: str) -> None:
    """Copies the BC MLP into PPO's policy mean network, folding the BC
    observation normalizer into layer 1 and the action scaling into the
    output layer.

    BC computes  y = f((x - mean) / std)  on raw obs x and outputs raw (v, omega).
    The env feeds the policy  o = x / OBS_SCALE  and expects  a = y / ACTION_SCALE.
    Layer 1:  W1 @ ((x-mean)/std) + b1 = (W1 * OBS_SCALE/std) @ o + (b1 - W1 @ (mean/std))
    Output:   rows of (W3, b3) divide by ACTION_SCALE.
    """
    if LEARNING_DIR not in sys.path:
        sys.path.insert(0, LEARNING_DIR)
    from bc_model import load_policy

    bc_model, normalizer, config = load_policy(bc_ckpt_path)
    if config["hidden_size"] != 64:
        raise SystemExit(f"Warm start expects hidden_size=64 (net_arch [64, 64]), "
                         f"got {config['hidden_size']}")

    obs_scale = torch.tensor(OBS_SCALE, dtype=torch.float32)
    act_scale = torch.tensor(ACTION_SCALE, dtype=torch.float32)
    mean = torch.tensor(normalizer.mean, dtype=torch.float32)
    std = torch.tensor(normalizer.std, dtype=torch.float32)

    W1 = bc_model.net[0].weight.data
    b1 = bc_model.net[0].bias.data
    W3 = bc_model.net[4].weight.data
    b3 = bc_model.net[4].bias.data

    pi_net = policy.mlp_extractor.policy_net
    with torch.no_grad():
        pi_net[0].weight.copy_(W1 * (obs_scale / std))
        pi_net[0].bias.copy_(b1 - W1 @ (mean / std))
        pi_net[2].weight.copy_(bc_model.net[2].weight.data)
        pi_net[2].bias.copy_(bc_model.net[2].bias.data)
        policy.action_net.weight.copy_(W3 / act_scale[:, None])
        policy.action_net.bias.copy_(b3 / act_scale)


def quick_eval(model, path_world, args, out_png: str):
    """One deterministic episode from the exact nominal start, plotted with
    the same target-vs-tip comparison track_trajectory.py uses."""
    env = SignatureEnv([path_world], frame_skip=args.frame_skip,
                       init_xy_noise=0.0, init_yaw_noise=0.0, max_time=args.max_time)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    tip = env.tracker.tip_history_array()
    errors_mm = tt.plot_comparison(path_world, tip, out_png)
    status = "finished" if info.get("is_success") else "did NOT finish"
    print(f"Quick eval: {status} in {env.tracker.elapsed_time:.1f}s sim, "
          f"rms={np.sqrt(np.mean(errors_mm ** 2)):.2f}mm max={errors_mm.max():.2f}mm")
    print(f"Saved quick-eval plot to {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", type=str, nargs="+", default=None,
                    help="Trajectory .npz files to train on. Defaults to every "
                         "target_trajectory_*.npz / trajectory_*_paper.npz in the project.")
    ap.add_argument("--num-envs", type=int, default=8,
                    help="Parallel environment processes (default 8)")
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--frame-skip", type=int, default=10,
                    help="Physics steps per policy action (10 -> 50 Hz control)")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="Path to a BC checkpoint (models/bc_policy.pt) to initialize "
                         "the PPO policy mean network from")
    ap.add_argument("--domain-rand", action="store_true",
                    help="Randomize mass/friction/gear/damping at each episode reset")
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="Std of additive Gaussian observation noise during training "
                         "(0 = off; ~0.05 models the real camera/IMU/encoder sensing "
                         "gap so the policy is robust to it on hardware)")
    ap.add_argument("--init-xy-noise-mm", type=float, default=10.0)
    ap.add_argument("--init-yaw-noise-deg", type=float, default=15.0)
    ap.add_argument("--max-time", type=float, default=60.0)
    # reward weights (see SignatureEnv)
    ap.add_argument("--w-progress", type=float, default=2.0)
    ap.add_argument("--w-track", type=float, default=0.02,
                    help="Quadratic tracking penalty weight, per mm^2 of tip error")
    ap.add_argument("--err-gate-mm", type=float, default=3.0,
                    help="Accuracy gate width: progress reward is scaled by "
                         "exp(-(err/this)^2), so sloppy progress earns nothing")
    ap.add_argument("--w-action-rate", type=float, default=0.05)
    ap.add_argument("--w-time", type=float, default=0.05)
    ap.add_argument("--completion-bonus", type=float, default=30.0)
    ap.add_argument("--off-path-penalty", type=float, default=30.0)
    ap.add_argument("--off-path-limit-mm", type=float, default=20.0)
    # PPO hyperparameters
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024, help="Rollout steps per env per update")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=-1.0,
                    help="Initial exploration log-std in the [-1,1] action space "
                         "(-1.0 -> std 0.37; keep modest with --warm-start)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=os.path.join(MODEL_DIR, "rl_policy.zip"))
    ap.add_argument("--no-eval", action="store_true", help="Skip the post-training quick eval")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    trajectory_paths = args.trajectories or find_trajectory_files()
    if not trajectory_paths:
        raise SystemExit("No trajectory .npz files found. Record one first (webapp.py).")
    print(f"Training on {len(trajectory_paths)} trajectory file(s):")
    for p in trajectory_paths:
        print(f"  {os.path.basename(p)}")

    path_worlds = [tt.load_path_world(p) for p in trajectory_paths]

    env_fns = [make_env_fn(path_worlds, args, rank) for rank in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, info_keywords=("is_success",))

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                         activation_fn=torch.nn.ReLU,
                         log_std_init=args.log_std_init)
    model = PPO("MlpPolicy", vec_env, learning_rate=args.lr, n_steps=args.n_steps,
                batch_size=args.batch_size, gamma=args.gamma, ent_coef=args.ent_coef,
                policy_kwargs=policy_kwargs, seed=args.seed, verbose=1,
                tensorboard_log=os.path.join(RL_DIR, "runs"))

    if args.warm_start:
        if not os.path.exists(args.warm_start):
            raise SystemExit(f"No BC checkpoint at {args.warm_start}")
        warm_start_from_bc(model.policy, args.warm_start)
        print(f"Warm-started policy mean network from {args.warm_start}")

    print(f"PPO on {args.num_envs} parallel env(s), {args.total_timesteps} timesteps, "
          f"domain_rand={args.domain_rand}, obs_noise={args.obs_noise}")
    model.learn(total_timesteps=args.total_timesteps)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Saved RL policy to {args.output}")

    # Record the exact run configuration next to the model, so results stay
    # attributable to their settings (see rl/TRAINING_LOG.md).
    config_path = os.path.splitext(args.output)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved run config to {config_path}")
    vec_env.close()

    if not args.no_eval:
        out_png = os.path.join(RL_DIR, "rl_train_quick_eval.png")
        quick_eval(model, path_worlds[-1], args, out_png)


if __name__ == "__main__":
    main()
