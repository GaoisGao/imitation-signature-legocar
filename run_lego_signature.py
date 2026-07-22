
"""
run_lego_signature.py - Step 5: drive the real LEGO Double Motor robot so its
pen tip follows a recorded signature trajectory, over Bluetooth via the LEGO
Education Python API (https://github.com/LEGO/LEGOEducation/blob/main/doublemotor.md).

This is the real-hardware counterpart to track_trajectory.py's MuJoCo
simulation. It reads the exact same trajectory files (webapp.py's
target_trajectory_*.npz, or coordinate_plane.py's trajectory_*_paper.npz)
and works directly in the paper mm frame (0,0)=ID0 marker corner,
(199,137)=ID3 marker corner - no MuJoCo world-frame offset needed here.

The Double Motor API (see lelib.py, already used elsewhere in this project)
only exposes coarse commands - turn in place N degrees, drive forward N
motor-shaft-degrees - and no absolute position sensor. So instead of
track_trajectory.py's continuous pure-pursuit + PI velocity loop, this script:

  1. Loads + smooths the recorded path, then resamples it to a uniform
     arc-length spacing suited for physical driving.
  2. Corrects for the pen trailing PEN_OFFSET_MM behind the driven wheel
     axle (same geometry as lego_car_with_pencil.xml: wheel axle at
     chassis x=-24mm, pencil at chassis x=-73mm -> pen trails the axle
     by 49mm), by projecting each waypoint back along the local path
     tangent onto the axle's path.
  3. Converts the resulting polyline into an alternating sequence of
     (turn_degrees, drive_mm) commands, merging near-collinear points so
     the robot doesn't stop-and-turn on every tiny wiggle.
  4. Replays the command sequence on the real Double Motor.

Known limitation: this is open-loop dead reckoning (no camera-in-the-loop
correction like webapp.py's live red-dot tracking) - wheel slip will make
long paths drift. Good for short signature-length paths; for anything
longer, re-running coordinate_plane.py's ArUco tracking on the robot itself
to close the loop would be the next step, not implemented here.

--dry-run prints the planned command sequence (and, with --plot, a preview
PNG of the original vs. axle-corrected path) without touching hardware -
always sanity-check this before connecting to the robot.

You MUST verify WHEEL_DIAMETER_MM and PEN_OFFSET_MM below against your
actual robot (the defaults are read off lego_car_with_pencil.xml, which is
supposed to model it), and set --card-serial to your Double Motor's
Bluetooth connection card.

Usage:
    py -3.13 run_lego_signature.py --dry-run --plot preview.png
    py -3.13 run_lego_signature.py --card-serial 1779
    py -3.13 run_lego_signature.py --trajectory target_trajectory_20260708_161115.npz --speed 25
"""

import argparse
import os

import numpy as np

import track_trajectory as tt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TUNE: measured from lego_car_with_pencil.xml (wheel radius 0.02m -> 40mm
# diameter; pencil body 49mm behind the wheel axle). Re-measure on the real
# robot if it's built differently than the simulated model.
WHEEL_DIAMETER_MM = 40.0
PEN_OFFSET_MM = 49.0
TRACK_WIDTH_MM = 72.0  # measured from lego_car_with_pencil.xml: wheel_left/right at y=-+36mm

DEFAULT_SEGMENT_SPACING_MM = 6.0   # arc-length spacing used to build turn/drive commands
DEFAULT_MIN_TURN_DEG = 3.0         # turns smaller than this get folded into the next drive
DEFAULT_SPEED = 30                 # % motor speed (see lelib.doubleMotor.set_speed)
DEFAULT_CARD_SERIAL = "2312"       # this robot's Double Motor connection card
DEFAULT_CARD_COLOR = "MAGENTA"     # this robot's Double Motor connection card


# -- path -> command conversion ----------------------------------------------

def apply_pen_offset(path_mm: np.ndarray, offset_mm: float, spacing_mm: float = 1.0) -> np.ndarray:
    """
    Returns the axle path that makes a pen trailing offset_mm behind the
    driven axle trace path_mm: the axle must be offset_mm of arc length
    *ahead* of the pen at all times, so axle[k] = path_mm[k + shift] (shift =
    offset_mm / spacing_mm samples, since path_mm is uniformly arc-length
    resampled at spacing_mm). The last `shift` axle points hold at the final
    target point, so the trailing pen still reaches the end of the path.

    This look-ahead approach stays exactly on the original curve, unlike
    extrapolating off the local tangent, which blows up whenever the path
    curves tighter than offset_mm (common in signature-scale loops/corners).
    No-op when offset_mm == 0.
    """
    if offset_mm == 0.0:
        return path_mm
    shift = min(int(round(offset_mm / spacing_mm)), len(path_mm) - 1)
    if shift <= 0:
        return path_mm
    hold = np.tile(path_mm[-1], (shift, 1))
    return np.vstack([path_mm[shift:], hold])


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def build_commands(path_mm: np.ndarray, min_turn_deg: float):
    """
    Converts a polyline into a sequence of straight-line run lengths (mm)
    separated by in-place turns (degrees). Returns (initial_heading_deg,
    runs, turns) where len(turns) == len(runs) - 1: drive runs[0], then for
    each (turn, run) pair, turn then drive run.
    """
    deltas = np.diff(path_mm, axis=0)
    seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
    seg_heading = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))

    keep = seg_len > 1e-6
    seg_len, seg_heading = seg_len[keep], seg_heading[keep]
    if len(seg_len) == 0:
        raise ValueError("Path has zero length after resampling.")

    runs = [float(seg_len[0])]
    turns = []
    heading = seg_heading[0]
    for length, h in zip(seg_len[1:], seg_heading[1:]):
        turn = _angle_diff_deg(h, heading)
        if abs(turn) < min_turn_deg:
            runs[-1] += length
            continue
        turns.append(turn)
        runs.append(float(length))
        heading = h

    return float(seg_heading[0]), runs, turns


def iter_steps(runs, turns):
    """Yields ("drive", mm) / ("turn", deg) steps in execution order."""
    yield ("drive", runs[0])
    for turn, run in zip(turns, runs[1:]):
        yield ("turn", turn)
        yield ("drive", run)


def mm_to_motor_degrees(distance_mm: float, wheel_diameter_mm: float) -> float:
    return distance_mm / (np.pi * wheel_diameter_mm) * 360.0


# TUNE: the Double Motor's left and right motors are mounted mirrored on the
# chassis - lelib.py's run_left/run_right both use
# MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE as "drive forward", so the two
# motors' raw position/encoder counts increase in *opposite* world-frame
# directions even though driving straight is a single shared "forward". This
# flips the sign of the left motor's reported position relative to the
# geometric (world-frame) wheel rotation computed below. Confirmed
# empirically with motor_dashboard.py: the right motor's actual-vs-expected
# position traces matched closely without any correction, while the left
# motor's actual trace was consistently mirrored (negated) relative to the
# uncorrected expected trace. Flip back to +1 if your build's left motor
# isn't mirrored.
LEFT_MOTOR_ENCODER_SIGN = -1
RIGHT_MOTOR_ENCODER_SIGN = 1


def wheel_deltas_for_step(kind: str, value: float, wheel_diameter_mm: float, track_width_mm: float):
    """
    Returns (left_delta_deg, right_delta_deg): the expected change in each
    wheel's *reported motor position* for one ("drive", mm) / ("turn", deg)
    step from iter_steps(), i.e. directly comparable to
    dm.motor[MOTOR_LEFT/RIGHT].position deltas. Assumes the same
    differential-drive pivot convention as lelib.py's turn_left/turn_right:
    a positive (left) turn pivots the right wheel forward and the left
    wheel backward (in world-frame terms), each by the arc length
    track_width_mm/2 * radians(turn) converted to motor-shaft degrees, then
    corrected to each motor's own encoder sign convention (see
    LEFT_MOTOR_ENCODER_SIGN above).
    """
    if kind == "drive":
        d = mm_to_motor_degrees(value, wheel_diameter_mm)
        left, right = d, d
    else:
        arc_mm = (track_width_mm / 2.0) * np.radians(abs(value))
        d = mm_to_motor_degrees(arc_mm, wheel_diameter_mm)
        left, right = (-d, d) if value > 0 else (d, -d)
    return left * LEFT_MOTOR_ENCODER_SIGN, right * RIGHT_MOTOR_ENCODER_SIGN


# -- preview plot --------------------------------------------------------------

def save_preview_plot(target_mm: np.ndarray, axle_mm: np.ndarray, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6 * tt.PLANE_HEIGHT_MM / tt.PLANE_WIDTH_MM))
    ax.plot(target_mm[:, 0], target_mm[:, 1], "--", color="steelblue", label="pen path (target)")
    ax.plot(axle_mm[:, 0], axle_mm[:, 1], "-", color="crimson", linewidth=1.2, label="axle path (driven)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -- hardware execution ----------------------------------------------------

def run_on_robot(runs, turns, args) -> None:
    from lelib import doubleMotor
    import legoeducation as le

    card_color = None
    if args.card_color:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}", None)
        if card_color is None:
            raise SystemExit(f"Unknown --card-color '{args.card_color}'. "
                              f"See lelib.py / constants.md for valid LEGO_COLOR_* names.")

    dm = doubleMotor()
    dm.connect(card_serial=args.card_serial, card_color=card_color)
    dm.set_speed(args.speed)

    try:
        for kind, value in iter_steps(runs, turns):
            if kind == "turn":
                if value > 0:
                    dm.turn_left(abs(value))
                else:
                    dm.turn_right(abs(value))
            else:
                motor_deg = mm_to_motor_degrees(value, args.wheel_diameter_mm)
                dm.movement_move_for_degrees(motor_deg, direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD)
    finally:
        dm.stop()
        dm.disconnect()


# -- CLI -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", type=str, default=None,
                    help="Path to a target_trajectory_*.npz or trajectory_*_paper.npz. "
                         "Defaults to the most recently modified one in this folder.")
    ap.add_argument("--wheel-diameter-mm", type=float, default=WHEEL_DIAMETER_MM)
    ap.add_argument("--pen-offset-mm", type=float, default=PEN_OFFSET_MM)
    ap.add_argument("--segment-spacing-mm", type=float, default=DEFAULT_SEGMENT_SPACING_MM,
                    help="Arc-length spacing (mm) used to build turn/drive commands")
    ap.add_argument("--min-turn-deg", type=float, default=DEFAULT_MIN_TURN_DEG,
                    help="Skip turns smaller than this; fold into the surrounding drive")
    ap.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="Motor speed, 0-100")
    ap.add_argument("--smooth-window", type=int, default=tt.DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("--card-serial", type=str, default=DEFAULT_CARD_SERIAL,
                    help="Bluetooth connection card serial for the Double Motor")
    ap.add_argument("--card-color", type=str, default=DEFAULT_CARD_COLOR,
                    help="Bluetooth connection card color, e.g. AZURE, RED, BLUE "
                         "(see lelib.py / constants.md).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned command sequence; don't connect to hardware")
    ap.add_argument("--plot", type=str, default=None,
                    help="Save a PNG preview of the target vs. axle-corrected path")
    args = ap.parse_args()

    trajectory_path = args.trajectory or tt.find_latest_trajectory_file(BASE_DIR)
    if trajectory_path is None:
        raise SystemExit("No trajectory .npz found. Pass --trajectory explicitly.")
    print(f"Loading trajectory: {trajectory_path}")

    xy_mm, _ = tt.load_trajectory_mm(trajectory_path)
    xy_mm = tt.smooth_xy(xy_mm, args.smooth_window)
    fine_path = tt.resample_by_arclength(xy_mm, spacing=1.0)
    axle_path = apply_pen_offset(fine_path, args.pen_offset_mm)
    coarse_path = tt.resample_by_arclength(axle_path, spacing=args.segment_spacing_mm)

    initial_heading, runs, turns = build_commands(coarse_path, args.min_turn_deg)
    total_mm = sum(runs)
    print(f"Path: {len(coarse_path)} waypoints -> {len(runs)} drive segments, "
          f"{len(turns)} turns, total distance ~{total_mm:.1f}mm")

    if args.plot:
        save_preview_plot(fine_path, axle_path, args.plot)
        print(f"Saved path preview to {args.plot}")

    print(f"Initial heading: {initial_heading:.1f} deg from +X axis "
          f"(aim the robot this way, at the axle path's start point, before it drives)")

    if args.dry_run:
        for kind, value in iter_steps(runs, turns):
            if kind == "turn":
                print(f"  turn {value:+.1f} deg")
            else:
                motor_deg = mm_to_motor_degrees(value, args.wheel_diameter_mm)
                print(f"  drive {value:.1f} mm ({motor_deg:.1f} motor-deg)")
        return

    color_note = f", color {args.card_color}" if args.card_color else ""
    print(f"Connecting to Double Motor (card serial {args.card_serial}{color_note})...")
    run_on_robot(runs, turns, args)
    print("Finished replaying the trajectory on the robot.")


if __name__ == "__main__":
    main()
