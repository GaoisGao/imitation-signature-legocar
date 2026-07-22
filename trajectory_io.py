"""
trajectory_io.py - single source of truth for saving recorded trajectories.

Every trajectory captured in this project goes through save_trajectory(), for
BOTH kinds of recording the overhead camera now produces:

  * a human signature demonstration (red tip on the handheld pencil), and
  * a robot-execution trace during deployment (red dot on the LEGO car's pen),

because the camera tracks the same red tip in both cases.

save_trajectory() ALWAYS writes two files with the same basename:

  1. <base>.npz  - the trainable format. Key 'target_trajectory' is an (N, 3)
     float64 array of [x_mm, y_mm, t_seconds] in the paper frame - the exact
     layout track_trajectory.py / webapp.py / learning/ / rl/ already load
     (they read columns 0:3 of 'target_trajectory'). Extra keys (raw pixels,
     pen_down, metadata) are stored alongside without breaking that contract.

  2. <base>.png  - a visualization of the raw trajectory, for a quick eyeball
     check of what was captured, drawn in the paper mm frame.

Naming convention (so downstream globbing keeps working):
  * human demos    -> prefix 'target_trajectory'  (the reference path)
  * robot traces   -> prefix 'robot_trace'         (an executed/evaluation run)
"""

import glob
import json
import os
from datetime import datetime

import numpy as np

# Physical sheet size (mm), must match coordinate_plane.py / track_trajectory.py.
PLANE_WIDTH_MM = 199.0
PLANE_HEIGHT_MM = 137.0

# -- dataset layout (single source of truth) ------------------------------
# All recorded trajectories are organized under datasets/:
#   datasets/trajectories/  <- the trainable .npz files
#   datasets/plots/         <- the .png visualizations
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
TRAJ_DIR = os.path.join(DATASET_DIR, "trajectories")
PLOT_DIR = os.path.join(DATASET_DIR, "plots")

# Filename patterns for recorded trajectories (human demos + robot traces +
# the legacy coordinate_plane.py paper output).
TRAJ_PATTERNS = ("target_trajectory_*.npz", "robot_trace_*.npz", "trajectory_*_paper.npz")


def timestamp() -> str:
    """A filename-safe YYYYMMDD_HHMMSS stamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_trajectory_files(project_dir: str = PROJECT_DIR):
    """Every recorded-trajectory .npz, newest last. Searches
    datasets/trajectories/ first and the project root second (legacy
    location), so old files still resolve during/after the move."""
    dirs = [os.path.join(project_dir, "datasets", "trajectories"), project_dir]
    files = []
    for d in dirs:
        for pat in TRAJ_PATTERNS:
            files += glob.glob(os.path.join(d, pat))
    # De-dup while keeping mtime order.
    files = sorted(set(files), key=os.path.getmtime)
    return files


def find_latest_trajectory_file(project_dir: str = PROJECT_DIR):
    """The most recently modified recorded trajectory, or None."""
    files = find_trajectory_files(project_dir)
    return files[-1] if files else None


def plot_trajectory(xy_mm: np.ndarray, out_png: str, title: str = None,
                    pen_down: np.ndarray = None,
                    plane_w: float = PLANE_WIDTH_MM, plane_h: float = PLANE_HEIGHT_MM):
    """Render the raw (paper-frame) trajectory to a PNG for a quick eyeball check.

    Draws the pen path within the sheet rectangle, marks start (green) and end
    (black), and, if `pen_down` is given, only connects consecutive samples
    while the pen was down (pen-up gaps are left as breaks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.asarray(xy_mm, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * plane_h / plane_w))

    if pen_down is None:
        ax.plot(xy[:, 0], xy[:, 1], "-", linewidth=1.5, color="crimson")
    else:
        pen_down = np.asarray(pen_down).astype(bool)
        seg = []
        for i in range(len(xy)):
            if pen_down[i]:
                seg.append(xy[i])
            elif seg:
                s = np.array(seg)
                ax.plot(s[:, 0], s[:, 1], "-", linewidth=1.5, color="crimson")
                seg = []
        if seg:
            s = np.array(seg)
            ax.plot(s[:, 0], s[:, 1], "-", linewidth=1.5, color="crimson")

    if len(xy):
        ax.plot(xy[0, 0], xy[0, 1], "o", color="green", markersize=7, label="start")
        ax.plot(xy[-1, 0], xy[-1, 1], "s", color="black", markersize=6, label="end")
        ax.legend(loc="upper right", fontsize=8)

    # Sheet boundary for context.
    ax.plot([0, plane_w, plane_w, 0, 0], [0, 0, plane_h, plane_h, 0],
            "-", color="0.7", linewidth=1.0)
    ax.set_xlim(-5, plane_w + 5)
    ax.set_ylim(-5, plane_h + 5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_trajectory(xy_mm, t, prefix: str = "target_trajectory",
                    pixel_xy=None, pen_down=None, meta: dict = None,
                    ts: str = None, traj_dir: str = TRAJ_DIR, plot_dir: str = PLOT_DIR):
    """Save one recording as a trainable .npz + a .png (separate folders).

    The .npz goes to datasets/trajectories/ and the .png to datasets/plots/,
    both sharing the same basename.

    Args:
        xy_mm:    (N, 2) paper-frame coordinates in millimeters.
        t:        (N,) timestamps in seconds (t[0] == 0 at recording start).
        prefix:   'target_trajectory' for human demos, 'robot_trace' for
                  robot-execution recordings (keeps downstream globs working).
        pixel_xy: optional (N, 2) raw camera-pixel path, stored for debugging
                  / re-calibration. Not used by training loaders.
        pen_down: optional (N,) 0/1 pen-contact flags.
        meta:     optional JSON-serializable dict (camera, resample_hz, etc.).
        ts:       optional timestamp override (else generated now).
        traj_dir/plot_dir: destination folders (default datasets/trajectories,
                  datasets/plots).

    Returns (npz_path, png_path).
    """
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    ts = ts or timestamp()
    base = f"{prefix}_{ts}"
    npz_path = os.path.join(traj_dir, base + ".npz")
    png_path = os.path.join(plot_dir, base + ".png")

    xy = np.asarray(xy_mm, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = min(len(xy), len(t))
    xy, t = xy[:n], t[:n]

    # The canonical trainable array every downstream consumer reads.
    target_trajectory = np.column_stack([xy, t])  # (N, 3): x_mm, y_mm, t

    save_data = {"target_trajectory": target_trajectory}
    if pixel_xy is not None:
        save_data["pixel_trajectory"] = np.asarray(pixel_xy, dtype=np.float64).reshape(-1, 2)[:n]
    if pen_down is not None:
        save_data["pen_down"] = np.asarray(pen_down, dtype=np.int8).reshape(-1)[:n]
    if meta is not None:
        save_data["meta_json"] = np.array(json.dumps(meta))

    np.savez(npz_path, **save_data)

    plot_trajectory(xy, png_path, title=base,
                    pen_down=save_data.get("pen_down"))

    return npz_path, png_path
