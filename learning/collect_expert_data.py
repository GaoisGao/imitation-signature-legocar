"""
collect_expert_data.py - Step: run the pure-pursuit expert controller
(track_trajectory.SignatureTracker) in MuJoCo over one or more recorded
signature trajectories, logging (observation, action) pairs at every
control step to build a behavior-cloning training set.

Observation (see track_trajectory.SignatureTracker._build_observation):
    [dx_local, dy_local, dist_to_final, at_end_flag] - the pure-pursuit
    lookahead target in the chassis's local frame, remaining distance to
    the end of the path, and whether the lookahead has run off the end of
    the path (drives the expert's end-of-path slowdown).
Action (see track_trajectory.SignatureTracker._expert_action):
    [v, omega] - chassis-frame forward speed (m/s) and yaw rate (rad/s).
    This is the same slot evaluate_bc.py substitutes a trained BC policy
    into, and what on-robot deployment would convert to left/right wheel
    speeds (the same differential-drive step track_trajectory.py already
    does internally).

--episodes-per-traj N demonstrates each trajectory N times: episode 0
always starts from the exact nominal pose (the clean demonstration), and
episodes 1..N-1 start with a randomized initial pose error
(--init-xy-noise-mm / --init-yaw-noise-deg), so the expert's *recovery*
back onto the path gets recorded too. Without those recovery
demonstrations, a BC policy never sees an error state during training and
has no idea how to act in one (the classic BC distribution-shift problem);
with them, it learns corrective behavior. The saved .npz keeps a per-step
`episode_ids` array so episode boundaries survive concatenation (useful
for sequence models or RL later).

Usage:
    py -3.13 learning/collect_expert_data.py
    py -3.13 learning/collect_expert_data.py --trajectory target_trajectory_20260708_161115.npz
    py -3.13 learning/collect_expert_data.py --all --episodes-per-traj 5 --output datasets/expert_dataset.npz
"""

import argparse
import glob
import os

import numpy as np

import common
import track_trajectory as tt
import trajectory_io as tio


def collect_episode(path_world, args, xy_noise_m: float, yaw_noise_rad: float, seed):
    """Runs the expert once (with the given initial-pose noise) and returns
    (obs (N,4), act (N,2), finished: bool)."""
    tracker = tt.SignatureTracker(path_world, speed=args.speed, lookahead=args.lookahead,
                                   finish_tol=args.finish_tol, path_spacing=args.path_spacing,
                                   record=True, init_xy_noise=xy_noise_m,
                                   init_yaw_noise=yaw_noise_rad, seed=seed)
    dt = tracker.m.opt.timestep
    max_steps = int(args.max_time / dt)
    for _ in range(max_steps):
        if tracker.step():
            break

    obs = np.array(tracker.observations, dtype=np.float32)
    act = np.array(tracker.actions, dtype=np.float32)
    return obs, act, tracker.finished


def collect_trajectory(trajectory_path: str, args, seed_rng):
    """Demonstrates one trajectory --episodes-per-traj times (episode 0
    clean, the rest with randomized initial pose errors). Returns a list of
    (obs, act) per episode; episodes that time out are dropped, since a
    non-recovering rollout would put non-expert behavior in the dataset."""
    path_world = tt.load_path_world(trajectory_path, args.smooth_window, args.path_spacing)
    name = os.path.basename(trajectory_path)
    xy_noise_m = args.init_xy_noise_mm / 1000.0
    yaw_noise_rad = np.radians(args.init_yaw_noise_deg)

    episodes = []
    for ep in range(args.episodes_per_traj):
        clean = ep == 0
        obs, act, finished = collect_episode(
            path_world, args,
            xy_noise_m=0.0 if clean else xy_noise_m,
            yaw_noise_rad=0.0 if clean else yaw_noise_rad,
            seed=int(seed_rng.integers(2 ** 31)))
        tag = "clean" if clean else "noisy"
        if not finished:
            print(f"  {name} ep{ep} ({tag}): TIMED OUT after {len(obs)} steps - dropped")
            continue
        print(f"  {name} ep{ep} ({tag}): {len(obs)} steps recorded")
        episodes.append((obs, act))
    return episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=str, default=None,
                    help="A single target_trajectory_*.npz / trajectory_*_paper.npz. "
                         "Defaults to the most recently modified one in the project folder.")
    ap.add_argument("--all", action="store_true",
                    help="Collect from every trajectory .npz found in the project folder "
                         "instead of just one")
    ap.add_argument("--smooth-window", type=int, default=tt.DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--path-spacing", type=float, default=tt.DEFAULT_PATH_SPACING)
    ap.add_argument("--lookahead", type=float, default=tt.DEFAULT_LOOKAHEAD)
    ap.add_argument("--speed", type=float, default=tt.DEFAULT_SPEED)
    ap.add_argument("--finish-tol", type=float, default=tt.DEFAULT_FINISH_TOL)
    ap.add_argument("--max-time", type=float, default=60.0, help="Safety cutoff on simulated seconds")
    ap.add_argument("--episodes-per-traj", type=int, default=1,
                    help="Demonstrations per trajectory: episode 0 starts from the exact nominal "
                         "pose, the rest from randomized initial pose errors, so the expert's "
                         "recovery behavior gets into the dataset (default 1 = clean only)")
    ap.add_argument("--init-xy-noise-mm", type=float, default=10.0,
                    help="Radius (mm) of the uniform disk the noisy episodes' starting pencil-tip "
                         "position is drawn from, around the path's start point")
    ap.add_argument("--init-yaw-noise-deg", type=float, default=15.0,
                    help="Noisy episodes' starting heading error, uniform in +- this many degrees")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the initial-pose perturbations (reproducible datasets)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output .npz path. Defaults to datasets/<trajectory-name>_expert.npz "
                         "(single trajectory) or datasets/expert_dataset.npz (--all)")
    args = ap.parse_args()

    if args.all:
        trajectory_paths = tio.find_trajectory_files(common.PROJECT_DIR)
        if not trajectory_paths:
            raise SystemExit("No trajectory .npz files found in the project folder.")
    else:
        trajectory_path = args.trajectory or tt.find_latest_trajectory_file(common.PROJECT_DIR)
        if trajectory_path is None:
            raise SystemExit("No trajectory .npz found. Pass --trajectory explicitly.")
        trajectory_paths = [trajectory_path]

    print(f"Running pure-pursuit expert over {len(trajectory_paths)} trajectory file(s), "
          f"{args.episodes_per_traj} episode(s) each...")
    seed_rng = np.random.default_rng(args.seed)
    obs_list, act_list, source_list, episode_id_list = [], [], [], []
    episode_id = 0
    for path in trajectory_paths:
        for obs, act in collect_trajectory(path, args, seed_rng):
            obs_list.append(obs)
            act_list.append(act)
            source_list.extend([os.path.basename(path)] * len(obs))
            episode_id_list.append(np.full(len(obs), episode_id, dtype=np.int32))
            episode_id += 1

    if not obs_list:
        raise SystemExit("No episodes finished within --max-time; nothing to save.")

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    episode_ids = np.concatenate(episode_id_list, axis=0)

    default_name = ("expert_dataset.npz" if args.all else
                    f"{os.path.splitext(os.path.basename(trajectory_paths[0]))[0]}_expert.npz")
    output_path = args.output or os.path.join(common.DATASET_DIR, default_name)

    np.savez(output_path, observations=observations, actions=actions,
             sources=np.array(source_list), episode_ids=episode_ids)
    print(f"Saved {len(observations)} (observation, action) pairs "
          f"({episode_id} episodes) to {output_path}")


if __name__ == "__main__":
    main()
