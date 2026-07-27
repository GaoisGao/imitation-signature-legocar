"""sim_match.py - Verify the SIM wheel-speed step response matches the REAL
robot's, after setting SignatureTracker's vel_lag_tau / vel_dead_time.

Workflow to bridge the inner-loop gap (see rl/README):
  1. Measure the real speed loop:  rl/deploy/motor_sysid.py step ...
     -> writes rl/deploy/sysid/sysid_fit_speed.json  (k_ss, tau, dead)
  2. Verify the sim reproduces it:  rl/deploy/sim_match.py --from-fit
     -> drives a wheel-speed STEP in MuJoCo through the same PI loop plus the
        vel_lag_tau/vel_dead_time model, fits tau/dead the same way, and reports
        whether the sim now matches the robot.
  3. Train with the matched values:
        py -3.13 rl/train_rl.py --vel-lag-tau <tau> --vel-dead-time <dead> ...

Because vel_lag_tau is an explicit first-order filter on the wheel-speed target
(and the PI downstream is near-instantaneous), the sim response tau should come
out approximately equal to vel_lag_tau; this script confirms that end to end,
on the ground, under the same physics training uses.
"""

import argparse
import json
import os
import sys

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(DEPLOY_DIR)
PROJECT_DIR = os.path.dirname(RL_DIR)
for p in (PROJECT_DIR, RL_DIR, DEPLOY_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import track_trajectory as tt
from motor_sysid import _fit_first_order  # identical fit to the hardware side

SYSID_DIR = os.path.join(DEPLOY_DIR, "sysid")
FIT_JSON = os.path.join(SYSID_DIR, "sysid_fit_speed.json")


def sim_step_response(vel_lag_tau, vel_dead_time, v_target=tt.DEFAULT_SPEED,
                      hold_s=None, pre_s=0.3):
    # settle needs several time constants; scale the hold to tau (min 1.2s)
    if hold_s is None:
        hold_s = max(1.2, 8.0 * vel_lag_tau + 2.0 * vel_dead_time)
    """Step the wheel-speed command 0 -> (v_target/r) at t=0 and record the
    left wheel's angular speed (rad/s) vs time. The chassis is HELD FIXED and
    gravity is off, so the wheel spins against nothing but its own bearing (the
    sim analogue of the robot's wheels-lifted sysid test, where the body is
    clamped in a fixture). This isolates the speed loop (PI + the vel_lag_tau/
    vel_dead_time model + wheel inertia) so the sim tau is comparable to the one
    motor_sysid.py fits on the robot. Returns (t, wheel_dps) in deg/s."""
    import mujoco

    # long straight path so the run never reaches the end during the test
    xs = np.linspace(0.0, 0.6, 301)
    path = np.column_stack([xs, np.zeros_like(xs)])

    state = {"t": 0.0}

    def controller(_obs):
        # zero command during the pre-roll, then a hard step to v_target
        return (0.0 if state["t"] < pre_s else v_target), 0.0

    tr = tt.SignatureTracker(path, controller=controller,
                             vel_lag_tau=vel_lag_tau, vel_dead_time=vel_dead_time)
    dt = tr.m.opt.timestep

    tr.m.opt.gravity[:] = 0.0
    a = tr.chassis_qpos_adr
    chassis_pose = tr.d.qpos[a:a + 7].copy()   # freeze the chassis here
    tr.pi_left.integral = tr.pi_right.integral = 0.0
    tr._lag_omega = None
    tr._dead_buf = None
    n = int(round((pre_s + hold_s) / dt))
    t_list, w_list, edge = [], [], None
    for i in range(n):
        state["t"] = i * dt
        tr.step()
        # Clamp the chassis stationary (fixture): the wheel hinge dof keeps its
        # own velocity, but the free-joint dofs are zeroed so motor reaction
        # torque can't spin the body - otherwise the joint speed never settles.
        tr.d.qpos[a:a + 7] = chassis_pose
        tr.d.qvel[tr.chassis_dof_adr:tr.chassis_dof_adr + 6] = 0.0
        w = float(tr.d.qvel[tr.joint_left.dofadr[0]])   # rad/s
        t_list.append(state["t"])
        w_list.append(w)
        if edge is None and state["t"] >= pre_s:
            edge = i
    t = np.array(t_list)
    w_dps = np.degrees(np.array(w_list))
    # re-base to the step edge for the fit
    return t[edge:] - t[edge], w_dps[edge:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vel-lag-tau", type=float, default=None,
                    help="Lag time constant to test (s). Default: from the HW fit.")
    ap.add_argument("--vel-dead-time", type=float, default=None,
                    help="Dead time to test (s). Default: from the HW fit.")
    ap.add_argument("--from-fit", action="store_true",
                    help=f"Load target tau/dead from {os.path.relpath(FIT_JSON, PROJECT_DIR)}")
    ap.add_argument("--speed", type=float, default=tt.DEFAULT_SPEED,
                    help="Forward speed of the step (m/s)")
    args = ap.parse_args()

    hw = None
    if args.from_fit or args.vel_lag_tau is None:
        if not os.path.exists(FIT_JSON):
            raise SystemExit(f"No hardware fit at {FIT_JSON}. Run "
                             "`rl/deploy/motor_sysid.py step` first, or pass "
                             "--vel-lag-tau/--vel-dead-time explicitly.")
        with open(FIT_JSON) as f:
            hw = json.load(f)
        print(f"Hardware (speed mode): tau={hw['tau_s']*1000:.0f}ms  "
              f"dead={hw['dead_s']*1000:.0f}ms  k_ss={hw['k_ss_dps_per_100pct']:.0f} deg/s/100%")

    tau = args.vel_lag_tau if args.vel_lag_tau is not None else float(hw["tau_s"])
    dead = args.vel_dead_time if args.vel_dead_time is not None else float(hw["dead_s"])

    t, w_dps = sim_step_response(tau, dead, v_target=args.speed)
    fit = _fit_first_order(t, w_dps, level=100.0)
    sim_tau = fit["tau"]
    sim_dead = fit["dead"]
    print(f"\nSim with vel_lag_tau={tau*1000:.0f}ms, vel_dead_time={dead*1000:.0f}ms:")
    print(f"  measured sim tau  = {sim_tau*1000:.0f} ms" if sim_tau else "  tau: n/a")
    print(f"  measured sim dead = {sim_dead*1000:.0f} ms" if sim_dead is not None else "  dead: n/a")

    if hw and sim_tau:
        err = abs(sim_tau - hw["tau_s"]) * 1000.0
        verdict = "MATCH" if err < 15.0 else "MISMATCH - adjust vel_lag_tau"
        print(f"\n  |sim tau - hw tau| = {err:.0f} ms  -> {verdict}")
    print(f"\nTrain with:  py -3.13 rl/train_rl.py --vel-lag-tau {tau:.3f} "
          f"--vel-dead-time {dead:.3f} ...")


if __name__ == "__main__":
    main()
