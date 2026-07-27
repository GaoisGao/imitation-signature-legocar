"""trace_bias.py - Split a closed-loop trace's tracking error into a SYSTEMATIC
part (a constant offset to one side of the path) and a RANDOM part (oscillation
about it).

Why this matters
----------------
RMS alone cannot tell those apart, but the fixes are completely different:

  * systematic (mean signed error != 0)  -> GEOMETRY CALIBRATION. The controller
    is tracking faithfully, but the thing being tracked is offset from where the
    code thinks the pencil tip is: --tip-offset-mm, --yaw-scale, wheel geometry.
    No amount of retraining removes it.
  * random (mean ~0, large std)          -> DYNAMICS. Lag, gain, speed. This is
    what controller/policy work actually improves.

A trace with mean signed error -3 mm and std 1 mm is a 3.2 mm RMS run that is
really a 1 mm controller with a 3 mm ruler error.

Sign convention: positive = tip is LEFT of the path direction of travel,
negative = RIGHT.

Usage:
    py -3.13 rl/deploy/trace_bias.py                       # every logged trace
    py -3.13 rl/deploy/trace_bias.py --glob '*1210*'       # a subset
"""

import argparse
import glob
import os

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(DEPLOY_DIR))
TRACE_DIR = os.path.join(PROJECT_DIR, "datasets", "closedloop_traces")


def signed_cross_track(tip_xy, path_xy):
    """Signed perpendicular distance from each tip point to the path polyline.

    Sign comes from the z-component of (path tangent) x (tip - nearest point):
    positive when the tip sits to the LEFT of the direction of travel."""
    tangents = np.gradient(path_xy, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    out = np.empty(len(tip_xy))
    for i, p in enumerate(tip_xy):
        d = np.linalg.norm(path_xy - p, axis=1)
        j = int(np.argmin(d))
        rel = p - path_xy[j]
        # cross(tangent, rel).z
        out[i] = tangents[j, 0] * rel[1] - tangents[j, 1] * rel[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="closedloop_log_*.npz",
                    help="Filename pattern inside datasets/closedloop_traces/")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(TRACE_DIR, args.glob)))
    if not files:
        raise SystemExit(f"No traces matching {args.glob} in {TRACE_DIR}")

    print(f"{'trace':<28}{'n':>5}{'rms':>7}{'bias':>8}{'std':>7}{'bias%':>7}  verdict")
    print("-" * 82)
    for f in files:
        d = np.load(f, allow_pickle=True)
        log, path_mm = d["log"], d["target_mm"]
        tip = log[:, 1:3]
        ok = np.isfinite(tip).all(axis=1)
        tip = tip[ok]
        if len(tip) < 20:
            continue

        sgn = signed_cross_track(tip, path_mm)
        bias, std = float(np.mean(sgn)), float(np.std(sgn))
        rms = float(np.sqrt(np.mean(sgn ** 2)))
        # Share of the squared error explained by the constant offset.
        frac = bias ** 2 / max(rms ** 2, 1e-9)

        if frac > 0.5:
            verdict = "SYSTEMATIC -> calibration"
        elif frac > 0.25:
            verdict = "mixed"
        else:
            verdict = "random -> dynamics"
        print(f"{os.path.basename(f)[:27]:<28}{len(tip):>5}{rms:>7.2f}"
              f"{bias:>8.2f}{std:>7.2f}{100*frac:>6.0f}%  {verdict}")

    print("\nbias = mean signed error (+ = tip LEFT of travel, - = RIGHT)")
    print("bias% = share of squared error from the constant offset alone")


if __name__ == "__main__":
    main()
