"""
evaluate_bc.py - Step: run a trained BC policy inside the same MuJoCo
SignatureTracker the pure-pursuit expert was recorded from
(track_trajectory.py), substituting the policy for the expert's
pure-pursuit control law via SignatureTracker's `controller` hook, and
report pencil-tip tracking error and completion - optionally against the
expert on the same trajectory.

--episodes N runs N rollouts with randomized initial pose errors
(--init-xy-noise-mm / --init-yaw-noise-deg, same placement-noise model as
learning/collect_expert_data.py) and reports per-episode results plus a
summary: completion rate (finished within --max-time), tracking error, and
completion time. With --compare-expert, the expert runs the same episodes
from *identical* starting poses (same per-episode seeds), so the
comparison is paired - any difference is the controller, not the luck of
the perturbation draw. Set the noise flags to 0 to evaluate from the exact
nominal start (the default single-episode mode does this).

Usage:
    py -3.13 learning/evaluate_bc.py
    py -3.13 learning/evaluate_bc.py --trajectory target_trajectory_20260708_161115.npz --compare-expert
    py -3.13 learning/evaluate_bc.py --episodes 10 --compare-expert
    py -3.13 learning/evaluate_bc.py --view
"""

import argparse
import os

import numpy as np
import torch

import common
import track_trajectory as tt
from bc_model import load_policy

# BC evaluation outputs (trace .npz + comparison .png) live here, parallel to
# datasets/sim_traces (pure-pursuit sim) and datasets/closedloop_traces (robot).
BC_EVAL_DIR = os.path.join(common.PROJECT_DIR, "datasets", "bc_policy")


def make_bc_controller(model, normalizer):
    """Wraps a loaded BCPolicy as a track_trajectory.SignatureTracker
    `controller(obs) -> (v, omega)` callable."""
    def controller(obs: np.ndarray):
        obs_norm = normalizer.transform(obs[None, :]).astype(np.float32)
        with torch.no_grad():
            action = model(torch.from_numpy(obs_norm)).numpy()[0]
        return float(action[0]), float(action[1])
    return controller


def run_episode(path_world, args, controller, seed, view: bool = False):
    """One rollout from a (possibly perturbed) start. Returns a dict with
    the tip history, whether it finished within --max-time, elapsed
    simulated time, and error stats vs. the target path."""
    tracker = tt.SignatureTracker(path_world, speed=args.speed, lookahead=args.lookahead,
                                   finish_tol=args.finish_tol, path_spacing=args.path_spacing,
                                   controller=controller,
                                   init_xy_noise=args.init_xy_noise_mm / 1000.0,
                                   init_yaw_noise=np.radians(args.init_yaw_noise_deg),
                                   seed=seed)
    dt = tracker.m.opt.timestep
    max_steps = int(args.max_time / dt)

    viewer_ctx = None
    if view:
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(tracker.m, tracker.d)

    for _ in range(max_steps):
        finished = tracker.step()
        if viewer_ctx is not None:
            viewer_ctx.sync()
            if not viewer_ctx.is_running():
                break
        if finished:
            break

    if viewer_ctx is not None:
        viewer_ctx.close()

    tip = tracker.tip_history_array()
    errors_mm = tt.compute_tracking_error_mm(tip[:, :2], path_world)
    return {
        "tip": tip,
        "finished": tracker.finished,
        "time_s": tracker.elapsed_time,
        "rms_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
        "max_mm": float(errors_mm.max()),
    }


def evaluate(path_world, args, controller, seeds, label, view_first: bool = False):
    results = []
    for i, seed in enumerate(seeds):
        r = run_episode(path_world, args, controller, seed, view=view_first and i == 0)
        status = f"finished in {r['time_s']:.1f}s" if r["finished"] else "TIMED OUT"
        print(f"  {label} ep{i}: {status}, rms={r['rms_mm']:.2f}mm max={r['max_mm']:.2f}mm")
        results.append(r)
    return results


def summarize(results):
    n = len(results)
    completed = [r for r in results if r["finished"]]
    return {
        "completion": f"{len(completed)}/{n}",
        "mean_rms_mm": float(np.mean([r["rms_mm"] for r in results])),
        "mean_max_mm": float(np.mean([r["max_mm"] for r in results])),
        "mean_time_s": (float(np.mean([r["time_s"] for r in completed]))
                         if completed else float("nan")),
    }


def print_summary_table(rows):
    """rows: list of (label, summary dict) pairs."""
    print(f"\n{'':24s} {'completion':>10s} {'mean rms (mm)':>14s} {'mean max (mm)':>14s} {'mean time (s)':>14s}")
    for label, s in rows:
        print(f"{label:24s} {s['completion']:>10s} {s['mean_rms_mm']:>14.2f} "
              f"{s['mean_max_mm']:>14.2f} {s['mean_time_s']:>14.1f}")


def plot_episodes(target_world, bc_results, expert_results, out_png):
    """Overlays every episode's tip path: crimson = BC, orange dash-dot =
    expert (if given), dotted blue = target. With initial-pose noise, the
    fan of starting points converging onto the path shows recovery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6 * tt.PLANE_HEIGHT_MM / tt.PLANE_WIDTH_MM))
    ax.plot(target_world[:, 0] * 1000, target_world[:, 1] * 1000, ":", color="steelblue",
            linewidth=1.5, label="target")
    if expert_results is not None:
        for i, r in enumerate(expert_results):
            ax.plot(r["tip"][:, 0] * 1000, r["tip"][:, 1] * 1000, "-.", color="darkorange",
                    linewidth=1.2, alpha=0.7, label="pure-pursuit expert" if i == 0 else None)
    for i, r in enumerate(bc_results):
        ax.plot(r["tip"][:, 0] * 1000, r["tip"][:, 1] * 1000, "-", color="crimson",
                linewidth=1.0, alpha=0.7, label="BC policy" if i == 0 else None)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=os.path.join(common.MODEL_DIR, "bc_policy.pt"))
    ap.add_argument("--trajectory", type=str, default=None,
                    help="Path to a target_trajectory_*.npz or trajectory_*_paper.npz. "
                         "Defaults to the most recently modified one in the project folder.")
    ap.add_argument("--smooth-window", type=int, default=tt.DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--path-spacing", type=float, default=tt.DEFAULT_PATH_SPACING)
    ap.add_argument("--lookahead", type=float, default=tt.DEFAULT_LOOKAHEAD)
    ap.add_argument("--speed", type=float, default=tt.DEFAULT_SPEED)
    ap.add_argument("--finish-tol", type=float, default=tt.DEFAULT_FINISH_TOL)
    ap.add_argument("--max-time", type=float, default=60.0,
                    help="Simulated seconds before an episode counts as a timeout (not completed)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Rollouts per controller, each from a fresh randomized initial pose")
    ap.add_argument("--init-xy-noise-mm", type=float, default=0.0,
                    help="Radius (mm) of the uniform disk each episode's starting pencil-tip "
                         "position is drawn from (0 = exact nominal start)")
    ap.add_argument("--init-yaw-noise-deg", type=float, default=0.0,
                    help="Each episode's starting heading error, uniform in +- this many degrees")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the initial-pose draws. BC and the expert reuse the same "
                         "per-episode seeds, so --compare-expert is a paired comparison.")
    ap.add_argument("--compare-expert", action="store_true",
                    help="Also run the pure-pursuit expert on the same episodes (identical "
                         "starting poses) and report it alongside")
    ap.add_argument("--view", action="store_true",
                    help="Open a live MuJoCo viewer window for the first BC episode")
    ap.add_argument("--output", type=str, default=None, help="Where to save the comparison plot PNG")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No trained model at {args.model}. Run learning/train_bc.py first.")
    model, normalizer, config = load_policy(args.model)
    print(f"Loaded BC policy from {args.model} (hidden_size={config['hidden_size']})")

    trajectory_path = args.trajectory or tt.find_latest_trajectory_file(common.PROJECT_DIR)
    if trajectory_path is None:
        raise SystemExit("No trajectory .npz found. Pass --trajectory explicitly.")
    print(f"Loading trajectory: {trajectory_path}")
    if args.episodes > 1 or args.init_xy_noise_mm > 0 or args.init_yaw_noise_deg > 0:
        print(f"Evaluating {args.episodes} episode(s), initial pose noise: "
              f"xy<={args.init_xy_noise_mm}mm, yaw<=+-{args.init_yaw_noise_deg}deg")

    path_world = tt.load_path_world(trajectory_path, args.smooth_window, args.path_spacing)

    seed_rng = np.random.default_rng(args.seed)
    seeds = [int(seed_rng.integers(2 ** 31)) for _ in range(args.episodes)]

    bc_controller = make_bc_controller(model, normalizer)
    bc_results = evaluate(path_world, args, bc_controller, seeds, "BC", view_first=args.view)

    expert_results = None
    rows = [("BC policy", summarize(bc_results))]
    if args.compare_expert:
        expert_results = evaluate(path_world, args, None, seeds, "expert")
        rows.append(("pure-pursuit expert", summarize(expert_results)))
    print_summary_table(rows)

    timestamp = os.path.splitext(os.path.basename(trajectory_path))[0]
    os.makedirs(BC_EVAL_DIR, exist_ok=True)
    out_npz = os.path.join(BC_EVAL_DIR, f"bc_eval_{timestamp}.npz")
    np.savez(out_npz, actual_trajectory=bc_results[0]["tip"], target_world=path_world)
    print(f"\nSaved BC evaluation trajectory (episode 0) to {out_npz}")

    out_png = args.output or os.path.join(BC_EVAL_DIR, f"bc_eval_{timestamp}.png")
    plot_episodes(path_world, bc_results, expert_results, out_png)
    print(f"Saved comparison plot to {out_png}")


if __name__ == "__main__":
    main()
