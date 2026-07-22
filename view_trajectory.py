"""
Trajectory Viewer
=================
Loads a recorded trajectory .npz and inspects/plots it. Recordings now live in
datasets/trajectories/ (a .png is auto-saved to datasets/plots/ at record time;
this viewer is for interactive inspection / older files).

Usage:
    python view_trajectory.py [path_to_file.npz]
If no path is given, the most recent recorded trajectory is used.
"""

import os
import sys

import numpy as np

import trajectory_io as tio


def _extract_xy_t(data):
    """Return (x, y, t) from either the current 'target_trajectory' format or
    the legacy multi-recording ('traj_i' + 'count') pixel format."""
    if "target_trajectory" in data.files:
        tr = data["target_trajectory"]
        return tr[:, 0], tr[:, 1], tr[:, 2]
    keys = [k for k in data.files if k not in ("count", "meta_json", "pixel_trajectory", "pen_down")]
    tr = max((data[k] for k in keys), key=len)
    return tr[:, 0], tr[:, 1], tr[:, 2]


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = tio.find_latest_trajectory_file()
        if filepath is None:
            print("No recorded trajectory .npz found and none was specified.")
            return
        print(f"No file specified, using most recent: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    print("keys:", data.files)
    x, y, t = _extract_xy_t(data)
    print(f"points={len(x)}  x: {x.min():.1f}-{x.max():.1f}  "
          f"y: {y.min():.1f}-{y.max():.1f}  duration: {t[-1] - t[0]:.2f}s")

    try:
        import matplotlib.pyplot as plt
        plt.plot(x, y, marker=".", markersize=2, color="crimson")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(os.path.basename(filepath))
        plt.show()
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
