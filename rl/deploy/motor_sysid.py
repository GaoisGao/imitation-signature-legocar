"""motor_sysid.py - System-identify the LEGO Double Motor's velocity response,
so the MuJoCo model (lego_car_with_pencil.xml + track_trajectory's PI loop) can
be tuned to match the REAL closed-loop dynamics instead of guessed physics.

Why this exists
---------------
The sim drives the wheels through a torque motor + a Python velocity PI loop
(track_trajectory.WheelVelocityPI) running at the 500 Hz physics rate. The real
robot is driven with `movement_move_tank(left%, right%)`, which is a SPEED
command regulated by the SPIKE hub's OWN internal PID - whose gains are baked
into firmware and are NOT exposed by legoeducation (there is no kp/ki getter).
So we cannot copy the hardware gains into the XML. Instead we MEASURE the real
closed-loop step response and fit an effective model:

    * k_ss   : steady-state wheel speed per 100% command   (deg/s per 100%)
    * tau    : first-order time constant of the speed loop  (s)
    * dead   : transport/BLE dead time before motion starts (s)

Those three numbers are what the sim must reproduce. `fit` prints XML/PI tuning
targets; the companion sim-side matcher (see rl/README) then picks
WheelVelocityPI KP/KI and wheel damping so the sim's step response overlays
these curves.

We characterize BOTH drive modes so the sim/deploy architecture choice is
data-driven, not assumed:
    * speed     -> movement_move_tank percents  (what drive_closed_loop.py uses;
                   exercises the firmware speed PID)
    * duty      -> motor_set_duty_cycle          (raw power, bypasses the
                   firmware PID - the closest analogue to the sim torque motor)

Usage (wheels OFF the ground for a clean no-load response):
    py -3.13 rl/deploy/motor_sysid.py step --card-serial 2312 --card-color magenta
    py -3.13 rl/deploy/motor_sysid.py step --card-serial 2312 --mode duty
    py -3.13 rl/deploy/motor_sysid.py fit                       # refit/plot a saved log
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(DEPLOY_DIR)
PROJECT_DIR = os.path.dirname(RL_DIR)
for p in (PROJECT_DIR, RL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

SYSID_DIR = os.path.join(DEPLOY_DIR, "sysid")

# Command levels (percent) to step through, both signs so we catch any
# asymmetry between drive directions. 0% between steps lets the wheel coast
# to rest so each step starts from a known state.
DEFAULT_LEVELS = [20, 40, 60, 80, 100, -40, -80]
SETTLE_S = 3.0          # hold each level this long: the firmware speed ramp can
                        # take >1s to settle at the default acceleration, so a
                        # short hold clips the steady state (seen 2026-07-23)
REST_S = 1.0            # coast-to-rest gap between levels
POLL_DT = 0.01          # target sampling period (s); actual rate is BLE-limited


# -- hardware ---------------------------------------------------------------

def _connect(args):
    from lelib import doubleMotor
    import legoeducation as le

    card_color = None
    if args.card_color:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}", None)
        if card_color is None:
            raise SystemExit(f"Unknown --card-color '{args.card_color}'")
    print(f"Connecting to Double Motor (card serial {args.card_serial})...")
    dm = doubleMotor()
    dm.connect(card_serial=args.card_serial, card_color=card_color)
    dm.motor_reset_relative_position()
    return dm, le


def _read(dm, le):
    """Snapshot both motors' live state from the background notification.
    Returns (left_pos_deg, right_pos_deg, left_speed, right_speed,
    left_power, right_power)."""
    ml, mr = dm.motor[le.MOTOR_LEFT], dm.motor[le.MOTOR_RIGHT]
    return (float(ml.position), float(mr.position),
            float(ml.speed), float(mr.speed),
            float(ml.power), float(mr.power))


def _command(dm, mode, pct):
    """Send `pct` to both sides in the chosen drive mode."""
    if mode == "speed":
        dm.movement_move_tank(float(pct), float(pct))
    else:  # duty
        import legoeducation as le
        dm.motor_set_duty_cycle(int(pct), motor=le.MOTOR_LEFT)
        dm.motor_set_duty_cycle(int(pct), motor=le.MOTOR_RIGHT)


def cmd_step(args):
    dm, le = _connect(args)
    if args.accel is not None:
        # The firmware default acceleration ramps speed gently (~0.3-0.5s rise);
        # set it high (100) to test how much of the lag is the ramp vs intrinsic.
        for m in (le.MOTOR_LEFT, le.MOTOR_RIGHT):
            dm.motor_set_acceleration(int(args.accel), int(args.accel), motor=m)
        print(f"Set motor acceleration/deceleration = {args.accel}")
    levels = [float(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS
    rows = []  # (t, level, l_pos, r_pos, l_spd, r_spd, l_pow, r_pow)
    print(f"Mode={args.mode}. Lift the wheels OFF the ground. Stepping "
          f"{levels} - {args.settle:.1f}s each. Ctrl+C to stop.")
    input("Press Enter when the wheels are free...")
    t0 = time.perf_counter()

    def sample(level, until):
        while time.perf_counter() < until:
            t = time.perf_counter() - t0
            rows.append((t, level, *_read(dm, le)))
            # keep a steady sample cadence without drift
            time.sleep(max(0.0, POLL_DT - ((time.perf_counter() - t0) - t)))

    try:
        _command(dm, args.mode, 0.0)
        sample(0.0, time.perf_counter() + REST_S)
        for lvl in levels:
            _command(dm, args.mode, lvl)
            sample(lvl, time.perf_counter() + args.settle)
            _command(dm, args.mode, 0.0)
            sample(0.0, time.perf_counter() + args.rest)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        dm.movement_stop()
        dm.disconnect()

    log = np.array(rows)
    os.makedirs(SYSID_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(SYSID_DIR, f"sysid_{args.mode}_{ts}.npz")
    np.savez(out, log=log, mode=args.mode, levels=np.array(levels),
             columns="t,level,l_pos,r_pos,l_spd,r_spd,l_pow,r_pow")
    print(f"Saved {len(log)} samples to {out}")
    fit_and_report(out)


# -- fitting ----------------------------------------------------------------

def _wheel_speed(spd):
    """The hub's own firmware speed estimate (l_spd/r_spd) is the clean response
    signal. Differentiating the BLE encoder position instead gives garbage - it
    updates sparsely, so np.gradient is mostly zero with occasional huge spikes
    (verified 2026-07-23). We fit the shape (tau/dead) on this channel; absolute
    deg/s-per-% calibration comes from openloop_deploy.py `calibrate`."""
    return np.asarray(spd, dtype=float)


def _fit_first_order(t, y, level):
    """Fit steady-state, first-order time constant tau, and dead time to one
    step segment (t re-based to 0 at the command edge). Returns a dict.

    tau is a least-squares fit, not a single 63% crossing: for a first-order
    rise y = y_ss (1 - exp(-(t-dead)/tau)), the transform z = -ln(1 - y/y_ss)
    is linear in (t-dead) with slope 1/tau, so a line through the (t, z) samples
    in the responsive band gives a far less noise-sensitive tau than one point."""
    y_ss = float(np.mean(y[int(0.7 * len(y)):]))          # last 30% = steady state
    if abs(y_ss) < 1e-6:
        return {"level": level, "y_ss": y_ss, "tau": None, "dead": None}
    s = np.sign(y_ss)
    # dead time: first sample exceeding 10% of steady state
    over = np.where(s * y >= 0.1 * s * y_ss)[0]
    dead = float(t[over[0]]) if len(over) else 0.0
    # least-squares tau over the 10%..90% band (linear region of the transform)
    frac = np.clip((s * y) / abs(y_ss), 0.0, 0.98)
    band = np.where((frac >= 0.1) & (frac <= 0.9) & (t > dead))[0]
    if len(band) >= 3:
        z = -np.log(1.0 - frac[band])
        slope = np.polyfit(t[band] - dead, z, 1)[0]
        tau = float(1.0 / slope) if slope > 1e-9 else None
    else:
        tgt = 0.632 * y_ss                                # fallback: 63% crossing
        reach = np.where(s * y >= s * tgt)[0]
        tau = float(t[reach[0]] - dead) if len(reach) else None
    return {"level": level, "y_ss": y_ss, "tau": tau, "dead": dead}


def fit_and_report(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    log = data["log"]
    mode = str(data["mode"])
    t, level = log[:, 0], log[:, 1]
    l_dps = _wheel_speed(log[:, 4])   # firmware speed channel (l_spd)
    r_dps = _wheel_speed(log[:, 5])   # firmware speed channel (r_spd)

    # split into per-level step segments (a segment = a contiguous run at one
    # nonzero command, re-based to its own start)
    segs, i = [], 0
    while i < len(level):
        j = i
        while j < len(level) and level[j] == level[i]:
            j += 1
        if level[i] != 0.0:
            segs.append((i, j))
        i = j

    print(f"\n=== system ID ({mode} mode), fit on the firmware speed channel ===")
    print(f"{'cmd%':>6} {'steady spd':>11} {'t63 (ms)':>9} {'dead (ms)':>10}")
    fits = []
    for (a, b) in segs:
        tt = t[a:b] - t[a]
        for label, dps in (("L", l_dps[a:b]), ("R", r_dps[a:b])):
            f = _fit_first_order(tt, dps, float(level[a]))
            f["side"] = label
            fits.append(f)
            tau_ms = f"{f['tau']*1000:.0f}" if f["tau"] else "n/a"
            dead_ms = f"{f['dead']*1000:.0f}" if f["dead"] is not None else "n/a"
            print(f"{f['level']:>5.0f}{label} {abs(f['y_ss']):>11.1f} {tau_ms:>9} {dead_ms:>10}")

    # Aggregate over steps that actually moved. Use |steady| and |level| so the
    # left wheel (which reads negative for a positive command - opposite encoder
    # sign to the right) doesn't cancel the right in the gain/gradient fits.
    good = [f for f in fits if f["tau"] is not None and abs(f["level"]) >= 40]
    cmds = np.array([abs(f["level"]) for f in good])
    yss = np.array([abs(f["y_ss"]) for f in good])
    k_ss = float(np.polyfit(cmds, yss, 1)[0] * 100.0) if len(good) >= 2 else float("nan")
    tau = float(np.median([f["tau"] for f in good])) if good else float("nan")
    dead = float(np.median([f["dead"] for f in good])) if good else float("nan")
    # Slew-rate check: if t63 grows with command amplitude, the response is
    # acceleration/ramp-limited, not first-order - fit tau is then amplitude
    # dependent and a slew-rate model fits the sim better than a single tau.
    hi = [f for f in good if abs(f["level"]) >= 80]
    lo = [f for f in good if abs(f["level"]) <= 40]
    if hi and lo:
        tau_hi = np.median([f["tau"] for f in hi]); tau_lo = np.median([f["tau"] for f in lo])
        if tau_hi > 1.5 * tau_lo:
            print(f"\nNOTE: t63 grows with amplitude ({tau_lo*1000:.0f}ms @<=40%% -> "
                  f"{tau_hi*1000:.0f}ms @>=80%%) => SLEW-RATE limited (firmware accel "
                  f"ramp), not a fixed lag. Re-run with --accel 100 to shrink it, and "
                  f"model it in sim as a wheel-speed slew limit, not just vel_lag_tau.")

    summary = {"mode": mode, "k_ss_dps_per_100pct": k_ss,
               "tau_s": tau, "dead_s": dead, "source": os.path.basename(npz_path)}
    print(f"\nk_ss = {k_ss:.1f} firmware-speed units per 100%   tau = {tau*1000:.0f} ms   "
          f"dead = {dead*1000:.0f} ms")
    print(f"Sim tuning target: set vel_lag_tau ~ {tau*1000:.0f}ms / vel_dead_time "
          f"~ {dead*1000:.0f}ms, verify with sim_match.py. (Absolute deg/s-per-% "
          "calibration is openloop_deploy.py `calibrate`, not this test.)")

    out_json = os.path.join(SYSID_DIR, f"sysid_fit_{mode}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved fit to {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(t, log[:, 1], color="gray", lw=1, label="command %")
        ax[0].set_ylabel("command (%)")
        ax[0].legend(loc="upper right")
        ax[1].plot(t, l_dps, color="crimson", lw=1, label="left wheel deg/s")
        ax[1].plot(t, r_dps, color="steelblue", lw=1, label="right wheel deg/s")
        ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("wheel speed (deg/s)")
        ax[1].legend(loc="upper right")
        ax[0].set_title(f"Motor sys-ID ({mode}): k_ss={k_ss:.0f} deg/s/100%, "
                        f"tau={tau*1000:.0f}ms, dead={dead*1000:.0f}ms")
        fig.tight_layout()
        png = os.path.join(SYSID_DIR, f"sysid_{mode}.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"Saved plot to {png}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")
    return summary


def cmd_fit(args):
    path = args.log
    if path is None:
        import glob
        logs = glob.glob(os.path.join(SYSID_DIR, "sysid_*.npz"))
        if not logs:
            raise SystemExit("No sysid_*.npz logs found. Run `step` first.")
        path = max(logs, key=os.path.getmtime)
        print(f"Using latest log: {path}")
    fit_and_report(path)


# -- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("step", help="Run speed/duty steps, log encoders, fit the response")
    p.add_argument("--card-serial", required=True)
    p.add_argument("--card-color", default=None)
    p.add_argument("--mode", choices=["speed", "duty"], default="speed",
                   help="speed = movement_move_tank (firmware PID); duty = raw power")
    p.add_argument("--levels", default=None,
                   help=f"Comma-separated command percents (default {DEFAULT_LEVELS})")
    p.add_argument("--settle", type=float, default=SETTLE_S,
                   help="Seconds to hold each command level")
    p.add_argument("--rest", type=float, default=REST_S,
                   help="Seconds at 0%% between levels")
    p.add_argument("--accel", type=float, default=None,
                   help="Set motor acceleration/deceleration (0-100) before the "
                        "test. Try 100 to remove the firmware's gentle default ramp "
                        "(the dominant lag); omit to measure the default behavior.")

    p = sub.add_parser("fit", help="Refit + replot a saved sysid log")
    p.add_argument("--log", default=None, help="sysid_*.npz (default: newest)")

    args = ap.parse_args()
    {"step": cmd_step, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
