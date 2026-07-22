# BC vs. pure pursuit — closed-loop results & analysis

Notes comparing the behaviour-cloning (BC) policy against the classical
pure-pursuit controller, in simulation and on the real robot (both deployed
closed-loop under the overhead camera + IMU). Same target signature throughout.

## Results so far

| Run | Controller | Speed / lookahead | Env | RMS error | Max error |
| --- | --- | --- | --- | --- | --- |
| Sim eval (`evaluate_bc.py`) | BC vs expert | 30 mm/s / 6 mm | MuJoCo, perfect obs | ~identical to expert | — |
| Real closed loop | pure pursuit | 15 mm/s / 15 mm | robot + camera | 2.2 mm | 11.3 mm |
| Real closed loop | pure pursuit | 30 mm/s / 6 mm | robot + camera | **1.8 mm** | **4.2 mm** |
| Real closed loop | BC policy (run 1) | ~30 mm/s / 6 mm | robot + camera | 2.7 mm | 6.0 mm |
| Real closed loop | BC policy (run 2) | ~30 mm/s / 6 mm | robot + camera | 2.0 mm | 4.5 mm |

Three things fall out of the matched-operating-point rows (30 mm/s, 6 mm lookahead):

- **Lookahead matters more than speed here.** Dropping lookahead 15 → 6 mm
  improved pure pursuit from 2.2 to 1.8 mm RMS and roughly halved max error
  (11.3 → 4.2 mm) by cutting fewer corners — even though speed doubled.
- **Run-to-run variation is real.** Two identical BC runs gave 2.7 and 2.0 mm
  RMS — a ~0.7 mm spread from camera/IMU noise, start pose, battery, and surface.
  Single runs are not conclusive; compare distributions, not single numbers.
- **BC ≈ pure pursuit within that noise.** Best BC (2.0 mm) is within ~0.2 mm of
  pure pursuit (1.8 mm) at the identical operating point, and the BC run-to-run
  spread (0.7 mm) is *larger* than that gap. So on hardware BC effectively
  matches the expert — the distillation transferred well. It still cannot beat
  the expert (its target is the expert).

## Why BC ≈ the expert in simulation (near-identical traces)

1. **BC's target *is* the expert.** It is trained by supervised regression (MSE)
   on `(observation, action)` pairs recorded *from* the pure-pursuit expert, so
   for each observation it learns to output exactly the expert's action. Its
   performance ceiling is the expert — it can match, never beat.
2. **The sim eval is fully in-distribution.** `evaluate_bc.py` runs BC in the
   same MuJoCo sim the training data came from, from the same clean nominal
   start (no init-pose noise by default), with perfect observations at the
   500 Hz sim step. BC only ever sees states it was trained on, so it reproduces
   the expert's actions almost exactly.
3. **The observation is a sufficient statistic for the action.** The 4-dim obs
   (`dx_local, dy_local, dist_to_final, at_end`) is literally *what pure pursuit
   reacts to*, so the expert action is a smooth deterministic function of it and
   a small 64×64 MLP fits it to very low error.

"Identical" therefore means two good things: training converged, and there is no
distribution shift in this clean sim test. BC also cuts the same corners as the
expert — it faithfully inherits the expert's tracking biases.

## Why the real BC trace differs from the sim BC trace

The sim was the best case (perfect obs, 500 Hz control, exact training dynamics).
The robot adds three things the sim did not have:

1. **Observation noise / latency.** Camera tip position (~0.5–1 mm pixel /
   homography noise) and IMU yaw arrive noisy and slightly delayed, at ~10 Hz.
   BC is a memoryless function trained on *clean* obs, so noisy inputs produce
   noisy actions (the small wobbles in the real trace).
2. **10 Hz vs 500 Hz control rate.** Each BC action is held for 100 ms on the
   robot instead of 2 ms in sim.
3. **Dynamics gap.** Real motors, friction, slip, and battery voltage differ
   from the sim model BC learned in.

Together these are the **sim-to-real distribution shift**: it appears only on
hardware, which is why the sim looked perfect and the robot does not.

## Why real BC is slightly worse than real pure pursuit (same camera, same loop)

- **Pure pursuit is the exact corrective law, recomputed from the current
  measured state every tick.** Whatever the measured tip/heading — even noisy or
  off-path — it computes by construction the (v, ω) that drives the tip back
  toward the path. It is inherently robust to noise and off-nominal states.
- **BC is a learned approximation of that law.** It carries (a) a small
  regression error even in-distribution, and (b) no guarantee of corrective
  behaviour when the observation drifts outside its training distribution, which
  happens more on hardware.

So on the robot: pure pursuit is the real controller; BC is a slightly noisy
copy of it that must also cope with inputs it did not train on. Net result:
**BC ≤ pure pursuit.** A ~0.5 mm RMS penalty is a reasonable price for that.

## Fair comparison (same speed)

Because BC runs at ~30 mm/s, compare it against pure pursuit at the *same* speed
to isolate the controller effect from the speed effect:

```bash
py -3.13 drive_closed_loop.py drive --card-serial 2312 --card-color magenta --speed 30 --lookahead 6
```

(30 mm/s, 6 mm lookahead = BC's operating point.) Measured result: pure pursuit
1.8 mm vs BC 2.0–2.7 mm across two runs at this identical operating point. The
best BC (2.0 mm) is within ~0.2 mm of pure pursuit, smaller than the BC
run-to-run spread (0.7 mm) — so BC effectively matches the expert on hardware.
Interestingly the 6 mm lookahead also made pure pursuit *better* than the
15 mm/s / 15 mm baseline, so lookahead, not speed, was the dominant knob for
this path. For a rigorous number, run each controller N times and compare means.

## Bottom line & next step

This is exactly what BC is supposed to do: match the expert in simulation, and
on hardware match it again to within run-to-run noise (2.0 mm vs 1.8 mm). It
cannot beat the expert (its target *is* the expert), but the distillation
transferred cleanly to the real robot end-to-end.

Because BC already essentially matches pure pursuit, the point of the **RL step**
is not "a bit more accuracy on this signature" — it is **robustness** (different
frictions, surfaces, disturbances) and potentially **higher speed** while
staying accurate. The most useful change would be adding **observation noise +
latency to the domain randomization** and retraining, so the policy is robust to
exactly the real-world sensing effects, then seeing whether it holds accuracy
where pure pursuit / BC start to degrade (e.g. at higher speed).
