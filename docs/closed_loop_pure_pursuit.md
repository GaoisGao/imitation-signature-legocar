# Running the closed-loop pure-pursuit controller (real robot)

How to make the real LEGO car trace a recorded signature using the
**closed-loop** pure-pursuit controller in [`drive_closed_loop.py`](../drive_closed_loop.py):
the overhead camera measures the pencil-tip position (red dot) and the built-in
IMU measures heading, and the controller steers the tip along the path in the
paper-millimetre frame. Validated at ~2.2 mm RMS tracking error.

This is the real-hardware, camera-in-the-loop counterpart to
[`track_trajectory.py`](../track_trajectory.py) (the same controller in MuJoCo
simulation) and [`run_lego_signature.py`](../run_lego_signature.py) (open-loop
dead reckoning, no feedback).

---

## This robot's confirmed settings

These are already baked in as defaults, listed here for reference:

| Setting | Value | Where |
| --- | --- | --- |
| Double Motor card serial / color | `2312` / `magenta` | `--card-serial` / `--card-color` |
| Recording color-sensor button card | `2312` / `magenta` | `--sensor-serial` |
| Motor wiring | invert both, **no** swap | baked into `drive_closed_loop.py` defaults |
| IMU yaw sign | `+1` (CCW positive) | `--yaw-sign` |
| IMU yaw scale | `0.1` (hub reports decidegrees) | `--yaw-scale` |
| Overhead camera id | `2` | `--camera` (try `0`/`1` if it fails) |

If you rewire the robot, change hubs, or move to a different build, re-run the
`motor-check` and `imu-check` steps below to reconfirm these.

---

## Prerequisites

- The printed ArUco calibration sheet (199 × 137 mm, markers ID 0–3) taped flat.
- Overhead camera looking straight down, whole sheet + margin in view, even
  lighting (no glare on the markers).
- A **red dot on the pencil tip** of the robot, and a **red-tipped pen** for
  signing. Only one strong red blob should be visible at a time.
- Python env set up (`py -3.13 -m pip install -r requirements.txt`).

---

## One-time calibration

Do these once per robot build (not every run). Results are baked into code or
saved to a file and reused automatically.

### 1. Motor speed calibration (deg/s per 100%)

Lift the car so **both wheels are off the ground**, then:

```bash
py -3.13 rl/deploy/openloop_deploy.py calibrate --card-serial 2312 --card-color magenta
```

Saves `rl/deploy/motor_calibration.json`, which `drive_closed_loop.py` reads
automatically. Re-run if you swap the battery or change motors (battery voltage
affects wheel speed). Override at runtime with `--degs-per-100pct` if needed.

### 2. Motor wiring check (forward / left / right)

On the floor with room to move:

```bash
py -3.13 drive_closed_loop.py motor-check --card-serial 2312 --card-color magenta
```

It drives forward, then turns left, then right, and prints what it expects.
This robot is already configured (invert both, no swap). Only if you see a
mismatch:

- FORWARD goes backward → add `--invert-left --invert-right` (already default here).
- FORWARD veers to one side → invert just that wheel (`--no-invert-left` / `--no-invert-right` to toggle).
- FORWARD ok but LEFT/RIGHT reversed → toggle the swap (`--swap-motors`).

### 3. IMU yaw sign check

```bash
py -3.13 drive_closed_loop.py imu-check --card-serial 2312 --card-color magenta
```

Rotate the robot **counter-clockwise** (viewed from above): the reading should
go **positive** and a 90° turn should read ~90 deg (raw ~900, since the hub
reports decidegrees). If it reads negative, add `--yaw-sign -1` to `drive`.

---

## Running a trace

### Step 1 — record a signature

```bash
py -3.13 record_trajectory.py --mode demo --camera 2 --sensor-serial 2312 --sensor-card-color magenta
```

Wait for the status bar to read `CALIBRATED` (all 4 markers locked), press the
sensor button (or `s`) to start, sign with the red pen, press again to stop and
save. Saved to `datasets/trajectories/target_trajectory_<timestamp>.npz` plus a
`.png` preview in `datasets/plots/`. (Keyboard fallback: `s` start/stop, `r`
reset, `q` quit; add `--no-sensor` to skip the button.)

### Step 2 — (optional) verify in simulation

```bash
py -3.13 track_trajectory.py --view
```

Runs the same controller in MuJoCo on your latest recording and reports the
tracking error, saving to `datasets/sim_traces/`. If it can't track it in sim,
fix the recording before going to hardware.

### Step 3 — preview the camera (no robot)

```bash
py -3.13 drive_closed_loop.py preview
```

Confirms the red tip is detected, `markers 4/4 LOCKED`, and the target path (plus
a green start dot and heading arrow) overlays correctly on the sheet. `ESC` quits.

### Step 4 — closed-loop drive

```bash
py -3.13 drive_closed_loop.py drive --card-serial 2312 --card-color magenta --speed 15 --lookahead 15
```

Place the robot anywhere on the sheet with the red tip visible (no manual aiming
needed). After the countdown it will automatically:

1. **Nudge forward** to measure its heading from the camera.
2. **Drive the tip to the path start.**
3. **Pause** (`--start-pause-s`, default 2 s).
4. **Trace** the signature (pure pursuit orients the chassis on its own).

`ESC` / `Ctrl+C` aborts. It auto-stops if the tip strays past `--off-path-limit`
(30 mm) or the red dot is lost for `--max-misses` ticks.

### Selecting which recording to trace

By default `preview` / `drive` use the **newest** recording. To pick another,
pass its path (see `datasets/plots/*.png` to eyeball shapes first):

```bash
py -3.13 drive_closed_loop.py drive --card-serial 2312 --card-color magenta \
    --trajectory datasets/trajectories/target_trajectory_20260720_162845.npz
```

---

## Outputs

Each drive saves to `datasets/closedloop_traces/`:

- `closedloop_log_<timestamp>.npz` — per-tick log (tip x/y, yaw, target, v/omega,
  wheel %, error).
- `closedloop_trace_<timestamp>.png` — target vs. actual tip trace with RMS / max
  error in the title.

---

## Tuning

| Flag | Default | Effect |
| --- | --- | --- |
| `--speed` | 30 mm/s | Tip speed. **Slower = more accurate** (more corrections per mm on the ~10 Hz loop). |
| `--lookahead` | 12 mm | How far ahead pure pursuit aims. **Smaller = hugs tight curves but can jitter; larger = smoother but cuts corners.** |
| `--start-pause-s` | 2 s | Pause after reaching the start, before tracing. |
| `--start-grace-s` | 2 s | Suppresses the off-path abort at the very start so the chassis can orient. |
| `--off-path-limit` | 30 mm | Abort if the tip strays this far from the path. |
| `--nudge-mm` | 20 mm | Forward distance used to measure heading; increase (e.g. 30) if the heading estimate is noisy. |
| `--manual-start` | off | Skip auto-approach: place the tip on the start facing the initial heading yourself. |

Good validated starting point: `--speed 15 --lookahead 15`. To sharpen peaks,
try `--speed 15 --lookahead 9`.

---

## Troubleshooting

- **Drives off straight, won't curve / halts early** → heading sign or scale is
  wrong. Re-run `imu-check` (CCW → positive; 90° → ~90 deg). Keep `--yaw-scale 0.1`.
- **Turns the wrong way** → motor wiring; re-run `motor-check` and adjust the
  `--swap-motors` / `--invert-*` flags.
- **"robot moved only N mm" during heading calibration** → wheels slipping or off
  the ground; ensure it's on the surface, or raise `--nudge-mm` / `--approach-speed`.
- **Red tip not detected** → remove other red objects; click the tip in
  `record_trajectory.py`'s window to retune HSV, or pass `--color`.
- **Markers won't lock** → improve lighting, flatten the sheet, ensure all four
  corners are visible and unoccluded at start.

Known limits: pen is always down (no lift), so the nudge + approach leave a short
mark before the trace — start near the origin to minimise it. Only the longest
recording in a file is used.
