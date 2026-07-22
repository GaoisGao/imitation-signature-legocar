"""
coordinate_plane.py - Live 2D coordinate plane from 4 ArUco corner markers,
plus Step 2 (pixel -> paper coordinate conversion) of the signature pipeline.

Place the printed aruco_plane.png sheet in view of the webcam.
When all 4 markers (IDs 0-3) are visible, a perspective-corrected
coordinate grid is overlaid on the camera feed and a clean top-down
rectified view is shown in a second window.

Coordinate system, in millimeters (matches aruco_test.py --generate-plane):
  ID 0 -> (0,   0  )  origin / bottom-left
  ID 1 -> (199, 0  )  +X axis end   (ID0-ID1 horizontal distance = 199 mm)
  ID 2 -> (0,   137)  +Y axis end   (ID0-ID2 vertical distance   = 137 mm)
  ID 3 -> (199, 137)  opposite corner (ID2-ID3 = 199 mm, ID1-ID3 = 137 mm)

Step 2: Convert to Paper Coordinate Frame
  - Detects the 4 ArUco markers to build a homography mapping paper
    millimeter coordinates to camera pixel coordinates.
  - Loads a pixel trajectory (x, y, t) recorded by record_trajectory.py.
  - Inverts the homography to map each pixel point to paper mm coordinates.
  - Smooths the trajectory with a moving-average filter and resamples it
    to a uniform time step.
  - Saves the result to a new .npz file.

Press ESC to quit.
Usage:
  python coordinate_plane.py
  python coordinate_plane.py --camera 1              # use a different camera
  python coordinate_plane.py --dict 1                 # match the dict used to generate tags
  python coordinate_plane.py --trajectory trajectory_20260708_143720.npz
  python coordinate_plane.py --trajectory trajectory_20260708_143720.npz \\
                             --output paper_trajectory_20260708_143720.npz \\
                             --smooth-window 5 --resample-hz 60
"""

import argparse
import glob
import os
import cv2 as cv
import numpy as np

# Physical size of the printed sheet, in millimeters.
PLANE_WIDTH_MM  = 199.0   # horizontal distance: ID0-ID1 and ID2-ID3
PLANE_HEIGHT_MM = 137.0   # vertical distance:   ID0-ID2 and ID1-ID3

# World (x, y) assigned to each marker ID, in millimeters.
WORLD = {
    0: [0.0,            0.0],
    1: [PLANE_WIDTH_MM,  0.0],
    2: [0.0,             PLANE_HEIGHT_MM],
    3: [PLANE_WIDTH_MM,  PLANE_HEIGHT_MM],
}

GRID_SPACING_MM = 20    # grid line spacing along each axis, in mm
RECT_SCALE      = 3     # pixels per mm in the rectified top-down view


# -- homography helpers ---------------------------------------------------

def _marker_center(corners_for_one_marker: np.ndarray) -> np.ndarray:
    return corners_for_one_marker[0].mean(axis=0)


def _build_homography(ids: np.ndarray, corners: list):
    """
    Compute H that maps world mm coords -> image pixel coords.
    Returns (H, id_map) where id_map = {marker_id: corners_array}.
    H is None when fewer than 4 plane markers are visible.
    """
    # OpenCV <5 returns ids shaped (N, 1); OpenCV 5 returns a flat (N,) array.
    ids_flat = np.asarray(ids).flatten()
    id_map = {int(ids_flat[i]): corners[i] for i in range(len(ids_flat))}
    if not all(mid in id_map for mid in WORLD):
        return None, id_map

    src = np.float32([WORLD[mid]                   for mid in (0, 1, 2, 3)])
    dst = np.float32([_marker_center(id_map[mid])   for mid in (0, 1, 2, 3)])
    H, _ = cv.findHomography(src, dst)
    return H, id_map


def _project(H: np.ndarray, world_xy) -> np.ndarray:
    """Map a single world (x, y) mm point into image pixel coords."""
    pt = np.float32([[world_xy]])
    return cv.perspectiveTransform(pt, H)[0][0]


def pixels_to_paper(H: np.ndarray, pixel_points: np.ndarray) -> np.ndarray:
    """Map an (N, 2) array of image pixel coords to an (N, 2) array of paper mm coords."""
    H_inv = np.linalg.inv(H)
    pts = np.float32(pixel_points).reshape(-1, 1, 2)
    world = cv.perspectiveTransform(pts, H_inv)
    return world.reshape(-1, 2)


def _pt(arr) -> tuple:
    """Convert any array-like to a plain Python int tuple for OpenCV drawing."""
    return (int(arr[0]), int(arr[1]))


# -- trajectory conversion (Step 2) ----------------------------------------

def smooth_trajectory(xy: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving-average smoothing applied independently to x and y.
    Edges are padded by replicating the first/last sample (not zero-padded),
    so the smoothed endpoints aren't dragged toward (0, 0).
    """
    if window <= 1 or len(xy) < window:
        return xy
    half = window // 2
    kernel = np.ones(window) / window
    padded_x = np.pad(xy[:, 0], (half, half), mode='edge')
    padded_y = np.pad(xy[:, 1], (half, half), mode='edge')
    smoothed_x = np.convolve(padded_x, kernel, mode='valid')[:len(xy)]
    smoothed_y = np.convolve(padded_y, kernel, mode='valid')[:len(xy)]
    return np.column_stack([smoothed_x, smoothed_y])


def resample_trajectory(xy: np.ndarray, t: np.ndarray, hz: float):
    """Resample (x, y) over time to a uniform sample rate via linear interpolation."""
    if len(t) < 2:
        return xy, t
    t_uniform = np.arange(t[0], t[-1], 1.0 / hz)
    if len(t_uniform) == 0:
        t_uniform = t[:1]
    x_uniform = np.interp(t_uniform, t, xy[:, 0])
    y_uniform = np.interp(t_uniform, t, xy[:, 1])
    return np.column_stack([x_uniform, y_uniform]), t_uniform


def convert_trajectory_file(input_path: str, output_path: str, H: np.ndarray,
                             smooth_window: int, resample_hz: float):
    """
    Loads a pixel trajectory .npz (x, y, t) saved by record_trajectory.py,
    converts each recording to paper mm coordinates, smooths and resamples
    it, and saves the result to output_path. Returns (output_path, count).
    """
    data = np.load(input_path)
    save_data = {}
    count = 0
    for key in data.files:
        if key == 'count':
            continue
        traj = data[key]              # (N, 3): x_px, y_px, t
        pixel_xy = traj[:, :2]
        t = traj[:, 2]

        paper_xy = pixels_to_paper(H, pixel_xy)
        paper_xy = smooth_trajectory(paper_xy, smooth_window)
        paper_xy, t_resampled = resample_trajectory(paper_xy, t, resample_hz)

        save_data[key] = np.column_stack([paper_xy, t_resampled])
        count += 1

    save_data['count'] = np.array([count])
    np.savez(output_path, **save_data)
    return output_path, count


def _default_output_path(input_path: str) -> str:
    folder = os.path.dirname(input_path) or "."
    root, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(folder, f"{root}_paper{ext}")


def _find_latest_trajectory_file(folder: str):
    candidates = glob.glob(os.path.join(folder, "trajectory_*.npz"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# -- drawing helpers ---------------------------------------------------------

def _draw_grid_overlay(frame: np.ndarray, H: np.ndarray) -> None:
    """Overlay a perspective-correct coordinate grid onto the camera frame."""
    GRID_COLOR    = (200, 200, 200)
    BOUNDARY_COLOR = (255, 200, 0)
    X_COLOR       = (50,  50,  220)   # red-ish for X axis
    Y_COLOR       = (30, 180,  30)    # green for Y axis
    LABEL_COLOR   = (20,  20,   20)

    # Sheet boundary (the actual 199 x 137 mm rectangle)
    boundary_world = [[0, 0], [PLANE_WIDTH_MM, 0], [PLANE_WIDTH_MM, PLANE_HEIGHT_MM], [0, PLANE_HEIGHT_MM]]
    boundary_pts = np.array([_pt(_project(H, c)) for c in boundary_world], dtype=np.int32)
    cv.polylines(frame, [boundary_pts], True, BOUNDARY_COLOR, 2, cv.LINE_AA)

    xs = np.arange(0, PLANE_WIDTH_MM + 1e-6, GRID_SPACING_MM)
    ys = np.arange(0, PLANE_HEIGHT_MM + 1e-6, GRID_SPACING_MM)

    # Horizontal world lines (constant Y)
    for y in ys:
        p1 = _pt(_project(H, [0, y]))
        p2 = _pt(_project(H, [PLANE_WIDTH_MM, y]))
        col = X_COLOR if y == 0 else GRID_COLOR
        cv.line(frame, p1, p2, col, 2 if y == 0 else 1, cv.LINE_AA)
        if 0 < y < PLANE_HEIGHT_MM:
            lp = _pt(_project(H, [0, y]) + np.array([-30, 4]))
            cv.putText(frame, f"{int(y)}", lp,
                       cv.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLOR, 1, cv.LINE_AA)

    # Vertical world lines (constant X)
    for x in xs:
        p1 = _pt(_project(H, [x, 0]))
        p2 = _pt(_project(H, [x, PLANE_HEIGHT_MM]))
        col = Y_COLOR if x == 0 else GRID_COLOR
        cv.line(frame, p1, p2, col, 2 if x == 0 else 1, cv.LINE_AA)
        if 0 < x < PLANE_WIDTH_MM:
            lp = _pt(_project(H, [x, 0]) + np.array([-4, 18]))
            cv.putText(frame, f"{int(x)}", lp,
                       cv.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLOR, 1, cv.LINE_AA)

    # Axis arrows
    origin = _pt(_project(H, [0, 0]))
    x_tip  = _pt(_project(H, [PLANE_WIDTH_MM, 0]))
    y_tip  = _pt(_project(H, [0, PLANE_HEIGHT_MM]))
    cv.arrowedLine(frame, origin, x_tip, X_COLOR, 2, tipLength=0.05)
    cv.arrowedLine(frame, origin, y_tip, Y_COLOR, 2, tipLength=0.05)

    # Axis labels
    cv.putText(frame, "X", (x_tip[0] + 8, x_tip[1] + 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, X_COLOR, 2, cv.LINE_AA)
    cv.putText(frame, "Y", (y_tip[0] + 4, y_tip[1] - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, Y_COLOR, 2, cv.LINE_AA)

    # Origin dot + label
    cv.circle(frame, origin, 5, (0, 0, 0), -1)
    cv.putText(frame, "(0,0)", (origin[0] + 8, origin[1] - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA)


def _make_rectified_view(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Warp the camera frame into a clean top-down view of the coordinate plane.
    World origin (0,0) -> bottom-left of output; +Y goes up; +X goes right.
    Output size matches the sheet's real 199 x 137 mm aspect ratio.
    """
    W = int(round(PLANE_WIDTH_MM * RECT_SCALE))
    Hh = int(round(PLANE_HEIGHT_MM * RECT_SCALE))

    # Source: where each world corner sits in the camera image
    src_pts = np.float32([
        _project(H, [0,              0]),               # ID 0 -> output bottom-left
        _project(H, [PLANE_WIDTH_MM, 0]),                # ID 1 -> output bottom-right
        _project(H, [0,              PLANE_HEIGHT_MM]),  # ID 2 -> output top-left
        _project(H, [PLANE_WIDTH_MM, PLANE_HEIGHT_MM]),  # ID 3 -> output top-right
    ])
    # Destination: standard screen layout (Y flipped so +Y is up)
    dst_pts = np.float32([
        [0, Hh],    # bottom-left
        [W, Hh],    # bottom-right
        [0, 0],     # top-left
        [W, 0],     # top-right
    ])

    H_rect = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(frame, H_rect, (W, Hh))

    # Draw clean grid on the rectified image
    xs = np.arange(0, PLANE_WIDTH_MM + 1e-6, GRID_SPACING_MM)
    ys = np.arange(0, PLANE_HEIGHT_MM + 1e-6, GRID_SPACING_MM)
    for x in xs:
        p = int(round(x * RECT_SCALE))
        cv.line(warped, (p, 0), (p, Hh), (210, 210, 210), 1)   # vertical
        if x > 0:
            cv.putText(warped, f"{int(x)}", (p + 2, Hh - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv.LINE_AA)
    for y in ys:
        p = int(round(y * RECT_SCALE))
        cv.line(warped, (0, Hh - p), (W, Hh - p), (210, 210, 210), 1)  # horizontal
        if y > 0:
            cv.putText(warped, f"{int(y)}", (2, Hh - p - 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv.LINE_AA)

    # Bold axes
    cv.arrowedLine(warped, (0, Hh), (W, Hh), (50,  50, 220), 2, tipLength=0.05)   # X
    cv.arrowedLine(warped, (0, Hh), (0,  0), (30, 180,  30), 2, tipLength=0.05)   # Y
    cv.putText(warped, "X", (W - 18, Hh - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 220), 2, cv.LINE_AA)
    cv.putText(warped, "Y", (5, 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (30, 180, 30), 2, cv.LINE_AA)
    cv.putText(warped, "(0,0)", (4, Hh - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv.LINE_AA)

    return warped


# -- main loop -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict",   type=int, default=0,
                    help="ArUco dictionary id (default 0 = DICT_4X4_50)")
    ap.add_argument("--camera", type=int, default=2,
                    help="Camera device id (default 2, the USB overhead camera)")
    ap.add_argument("--trajectory", type=str, default=None,
                    help="Path to a pixel trajectory .npz saved by record_trajectory.py. "
                         "If omitted, the most recent trajectory_*.npz in this folder is used.")
    ap.add_argument("--output", type=str, default=None,
                    help="Where to save the converted paper-frame trajectory. "
                         "Defaults to '<trajectory>_paper.npz'.")
    ap.add_argument("--smooth-window", type=int, default=5,
                    help="Moving-average window size, in samples (default 5)")
    ap.add_argument("--resample-hz", type=float, default=60.0,
                    help="Uniform resampling rate in Hz (default 60)")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="Only show the live coordinate plane; skip trajectory conversion")
    args = ap.parse_args()

    trajectory_path = None if args.no_trajectory else args.trajectory
    if trajectory_path is None and not args.no_trajectory:
        trajectory_path = _find_latest_trajectory_file(os.path.dirname(os.path.abspath(__file__)))
        if trajectory_path:
            print(f"No --trajectory given, using most recent file: {trajectory_path}")

    output_path = None
    if trajectory_path:
        output_path = args.output or _default_output_path(trajectory_path)

    aruco_dict = cv.aruco.getPredefinedDictionary(args.dict)
    detector   = cv.aruco.ArucoDetector(aruco_dict, cv.aruco.DetectorParameters())

    cap = cv.VideoCapture(args.camera, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera id={args.camera}")

    print("Searching for ArUco markers ID 0-3 ... point camera at the sheet.")
    if trajectory_path:
        print(f"Will convert '{trajectory_path}' to paper coordinates once markers are locked.")
    print("Press ESC to quit.")

    converted = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners, ids, _ = detector.detectMarkers(frame)
        overlay = frame.copy()

        if ids is not None and len(ids) > 0:
            cv.aruco.drawDetectedMarkers(overlay, corners, ids)
            H, id_map = _build_homography(ids, corners)

            found_ids = sorted(int(i) for i in ids.flatten() if int(i) in WORLD)
            status = f"Markers found: {found_ids}  ({len(found_ids)}/4)"

            if H is not None:
                _draw_grid_overlay(overlay, H)
                rectified = _make_rectified_view(frame, H)
                cv.imshow("Top-down coordinate plane", rectified)
                status += "  |  LOCKED"
                text_color = (0, 220, 0)

                if trajectory_path and not converted:
                    try:
                        out, n = convert_trajectory_file(
                            trajectory_path, output_path, H,
                            args.smooth_window, args.resample_hz)
                        print(f"Converted {n} trajectories from '{trajectory_path}' "
                              f"to paper coordinates (mm) -> '{out}'")
                    except Exception as exc:
                        print(f"Trajectory conversion failed: {exc}")
                    converted = True
            else:
                text_color = (0, 180, 255)
        else:
            status = "No markers detected - aim camera at the sheet"
            text_color = (0, 0, 255)

        if trajectory_path:
            status += f"  |  trajectory: {'converted' if converted else 'waiting for lock'}"

        # Semi-transparent status bar
        bar = overlay[0:36, :]
        cv.rectangle(overlay, (0, 0), (overlay.shape[1], 36), (30, 30, 30), -1)
        cv.addWeighted(bar, 0.4, overlay[0:36, :], 0.6, 0, overlay[0:36, :])
        cv.putText(overlay, status, (10, 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv.LINE_AA)

        cv.imshow("Camera feed - ArUco coordinate plane", overlay)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
