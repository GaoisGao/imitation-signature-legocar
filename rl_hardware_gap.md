# RL sim-to-real gap on hardware — diagnosis

Why the RL (SB3 PPO) policies fail on the real robot while the classical
pure-pursuit controller and the BC policy deploy cleanly. Written before the
frequency-matched retrain, to record the evidence and the fix.

## Evidence (real robot, closed loop, same signature)

| Controller | speed scale | RMS (mm) | max (mm) | result |
| --- | --- | --- | --- | --- |
| pure pursuit | — | **1.8** | 4.2 | tracks well |
| BC policy | — | **2.0** | ~5 | tracks well |
| `rl_policy.zip` (old, pre-DR) | 1.0 | 67.8 | 108.8 | diverges (straight-line runaway) |
| `rl_dr_policy.zip` (obs-noise + physical DR) | 1.0 | 64.4 | 84.5 | diverges — *same failure mode* |
| `rl_dr_policy.zip` | 0.4 | **17.0** | 49.7 | much better, still far from usable |

Two different RL policies (pre-DR and DR) fail the **same way** at full speed,
and slowing to 0.4× cut RMS 64 → 17 mm. So the failure is **systematic to the RL
deployment**, not one bad checkpoint, and **speed/rate is the dominant factor**.

## Root cause: control-frequency mismatch (+ aggressive action space)

- The SB3 `SignatureEnv` trains at `frame_skip=10` → **50 Hz** control (one action
  every 20 ms). `drive_closed_loop.py` deploys over Bluetooth at **10 Hz** (one
  action every 100 ms). The policy learned a mapping calibrated to acting 5× more
  often than it does on hardware.
- The RL action space allows **v up to 60 mm/s** (2× the expert) and large omega.
  At 60 mm/s on the 10 Hz loop that is ~6 mm of travel per tick — a full
  lookahead — so a single held command overshoots, the next observation is far
  off, and the correction runs away.
- Slowing both v and omega by 0.4× (≈24 mm/s) largely removes the overshoot
  (64 → 17 mm), confirming the mechanism. It is not fully fixed because the
  policy's *mapping itself* was learned at 50 Hz and is out of distribution at
  10 Hz — scaling can't recover that; retraining at the deployment rate can.

## Why pure pursuit and BC transfer but RL does not

- **Pure pursuit** recomputes a corrective `(v, omega)` from the *current measured*
  tip/heading every tick. It is a feedback law: robust to control rate (just less
  precise when slower) and to disturbances by construction.
- **BC** imitates that feedback law's `(obs → action)` mapping, so it inherits the
  same rate-robustness, and it drives at the expert's gentle ~30 mm/s.
- **RL** optimized a per-step mapping against the 50 Hz sim dynamics and learned to
  exploit the faster action rate / higher speed. That mapping does not hold at
  10 Hz on real motors.

## Important: the DR we added does NOT address this

`--domain-rand` (mass/friction/gear/damping) and `--obs-noise` model the
**sensing and dynamics** gap — not the **control-frequency** gap. That is why the
DR policy failed identically to the non-DR one. **Frequency matching is the more
important fix here than sensor-noise DR.**

## Fix plan (next)

1. **Retrain at the deployment rate:** `--frame-skip 50` → 50 × 0.002 s = 0.1 s =
   **10 Hz**, matching `drive_closed_loop`. The policy then learns actions
   appropriate for a 100 ms hold.
2. Keep light DR: `--obs-noise 0.03` (4-dim obs is sensitive) and smoother actions
   `--w-action-rate 0.1`.
3. Consider capping the action space (`V_MAX`, `OMEGA_MAX` in `signature_env.py`)
   so the policy can't be aggressive.
4. Reward weights are tuned for 50 Hz (per-step); with 5× fewer steps per episode
   at `frame_skip=50` they may need retuning — expect a tuning pass.
5. Re-eval in sim, then deploy and compare to pure pursuit (1.8 mm) / BC (2.0 mm).

Command:
```bash
py -3.13 rl/train_rl.py --warm-start models/bc_policy.pt \
    --frame-skip 50 --obs-noise 0.03 --w-action-rate 0.1 \
    --output models/rl_dr_10hz.zip
```

## Takeaway

pure pursuit and BC are already good on hardware (1.8 / 2.0 mm). RL's value would
be robustness / higher speed, but it first has to *transfer* — and the biggest
transfer blocker here is the 50 Hz→10 Hz control-rate mismatch, not sensor noise.
Frequency-matched retraining is the prerequisite before any RL-vs-classical
comparison on the real robot is meaningful.
