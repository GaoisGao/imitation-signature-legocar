# RL training log

Run-by-run record: what was trained, what went wrong, what changed between runs.
Referenced from `rl/README.md`, `rl/signature_env.py`, and `rl/train_rl.py` as the
place where the reward weights and hyperparameter defaults are justified.

**Scope:** SB3 PPO on the deployable `(v, ω)` action space (`rl/train_rl.py`). The
mjlab port is a separate sim-only research track — its results live in
`RL_PROGRESS.md`.

**Provenance note.** Runs 1–2 predate TensorBoard retention; they are reconstructed
from the code comments they produced (`signature_env.py:26-35`, `:206-211`,
`README.md:213-217`) and no metrics survive. Runs 3–6 are measured from
`rl/runs/PPO_1..PPO_4`. Where a number is reconstructed rather than measured it is
marked *(no log)*.

---

## Summary

| # | TB dir | Model | Ctrl Hz | Plant | log_std | success | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | *(no log)* | — | 50 | ideal | -1.0 | — | linear track penalty → corner-cutting at full speed |
| 2 | *(no log)* | — | 50 | ideal | -1.0 | — | per-step error penalty → sprint-and-smear |
| 3 | `PPO_1` | `rl_dr_policy.zip` | 50 | ideal | -1.0 | **0.95** | works in sim; **diverges on hardware** |
| 4 | `PPO_2` | `rl_dr_10hz.zip` | 10 | ideal | -1.0 | **0.96** | best sim run ever; **fails under lag** |
| 5 | `PPO_3` | `rl_10hz_lag.zip` | 10 | lag | -1.0 | **0.00** | dead on arrival — never finished one episode |
| 6 | `PPO_4` | `rl_10hz_lag_v2.zip` | 10 | lag | **-2.5** | **0.89** | first to finish under lag (6/7, 4.68 mm) |
| 7 | `PPO_5` | `rl_A_best.zip` | 10 | lag | -2.5 | 0.88 | speed cap + reward fix → **6/7, 3.54 mm**; **1.94 mm on hardware** at `--policy-omega-scale 0.25` |
| 8 | `PPO_6` | `rl_B_best.zip` | 10 | lag+delay | -2.5 | 0.79 | obs-delay: mechanism works, estimate was too high (negative result) |
| 9 | `PPO_7` | `rl_C.zip` | 10 | lag | -2.5 | *running* | `omega_max` capped at training instead of clamped at deploy |

"Plant" = whether `vel_lag_tau`/`vel_dead_time` were active (the sysid-measured
wheel speed-loop lag, τ=0.479 s, dead=0.063 s from
`rl/deploy/sysid/sysid_fit_speed.json`).

---

## Runs 1–2 — reward shaping failures *(no log)*

Both trained at 50 Hz on the ideal plant. Neither produced a usable policy, but
they are the reason three of the current defaults exist.

**Run 1 — linear tracking penalty.** With the tracking penalty linear in error,
cutting corners at full speed was reward-optimal: the error paid for the shortcut
was less than the progress gained. **Fix:** the penalty is now **quadratic**
(`w_track * err_mm ** 2`), so large excursions cost superlinearly.

**Run 2 — per-step error penalty is not speed-invariant.** A plain per-step error
penalty has an episode total that *shrinks the faster the car goes* — fewer steps
means fewer penalties — so the policy learned to sprint and smear. **Fix:** the
**accuracy gate**, `exp(-(err/err_gate_mm)²)`, multiplies the progress reward.
Progress earns nothing unless the tip is actually on its local stretch of path,
which makes the objective speed-invariant.

**Standing constraint from these two runs:** accurate tracing must stay
**net-positive per step**. If staying on the path pays worse than
`-off_path_penalty`, the optimal policy is to dive off the path immediately and end
the episode cheaply. Check this whenever the weights are retuned.

---

## Run 3 (`PPO_1`) → `rl_dr_policy.zip` — 50 Hz, ideal plant

`--frame-skip 10 --domain-rand --obs-noise 0.05 --warm-start bc`, 2M steps.

| metric | value |
| --- | --- |
| success_rate | 0.95 final, 1.00 @139k |
| ep_rew_mean | 234 |
| ep_len_mean | 528 |

Reward weights `w_track 0.02 / w_time 0.05 / w_action_rate 0.05`. Clean sim result.

**Hardware: diverged** (64.4 mm RMS at speed scale 1.0; 17.0 mm at 0.4×). Root
cause was *not* DR but the **50 Hz-sim → 10 Hz-hardware control-frequency
mismatch** — see `rl_hardware_gap.md`.

## Run 4 (`PPO_2`) → `rl_dr_10hz.zip` — 10 Hz, ideal plant

The frequency-matched retrain: `--frame-skip 50`, DR off, `--obs-noise 0.03`.

| metric | value |
| --- | --- |
| success_rate | 0.96 final, 1.00 @114k |
| ep_rew_mean | **375** (best of any run) |
| ep_len_mean | 111 |

Fastest learner in the set — 0.99 success by 100k steps.

**But:** evaluated deterministically under the sysid-matched lag it **aborts
off-path at 11.1 s** (8.10 mm RMS, 16.07 mm max). Fixing the control *frequency*
was necessary but not sufficient; the wheel speed-loop *lag* was still unmodelled.

## Run 5 (`PPO_3`) → `rl_10hz_lag.zip` — 10 Hz + lag, log_std -1.0 — **FAILED**

First run with the lag plant active (`--vel-lag-tau 0.48 --vel-dead-time 0.063`),
plus rescaled per-second weights (`w_track 0.10 / w_time 0.25 / w_action_rate 0.01`).

| metric | value |
| --- | --- |
| success_rate | **0.00** for all 2M steps (best 0.02 @32k) |
| ep_rew_mean | -75 |
| ep_len_mean | **3** |

**Dead on arrival.** `ep_len_mean = 3` means episodes ended after 3 control steps
— 0.3 s. It never completed a single signature in 2M steps, and the saved
checkpoint aborts at 0.4 s in deterministic eval.

**Root cause: exploration noise did not scale with control period.** `log_std_init
-1.0` (σ≈0.37) was carried over from the 50 Hz runs. At 10 Hz each sampled action
is held for **100 ms**, so a σ=0.37 kick throws the tip off-path before the next
correction can land. A sim survival sweep put the usable ceiling near σ≈0.1
(log_std ≈ -2.3) under the 480 ms wheel lag.

**Fix:** `--log-std-init` default changed **-1.0 → -2.5** (σ≈0.08). Scale by
≈ln(5)=1.6 per 5× change in control frequency.

## Run 6 (`PPO_4`) → `rl_10hz_lag_v2.zip` — 10 Hz + lag, log_std -2.5

Identical to Run 5 except `log_std_init -2.5`, DR on, `--obs-noise 0.03`.
2026-07-27, 2M steps in 50 min (~680-1000 fps, 8 envs).

| step | 50k | 250k | 688k | 1368k | 2007k (final) |
| --- | --- | --- | --- | --- | --- |
| success_rate | 0.72 | 0.72 | 0.84 | **0.89** *(best)* | 0.79 |
| ep_rew_mean | -231 | -120 | 13 | 18 | -29 |
| ep_len_mean | 277 | 277 | 212 | 215 | 220 |

Final `train/std` collapsed to **0.021** — the policy drove its own exploration far
below even the -2.5 init (σ=0.08), confirming that low action noise is what this
plant rewards.

**The exploration-noise hypothesis is confirmed.** Same plant, same weights, same
warm start as Run 5 — only `log_std_init` changed — and success went **0.00 →
0.89**. Run 5's failure was exploration noise, not an unlearnable plant.

### Deterministic eval — the first policy to finish under lag

`evaluate_rl.py --from-fit --frame-skip 50`, all 7 recorded signatures, against
pure pursuit on the identical plant:

| | finished | mean RMS (finished) |
| --- | --- | --- |
| `rl_10hz_lag_v2` | **6 / 7** | 4.68 mm |
| pure pursuit | **6 / 7** | **2.55 mm** |
| all earlier checkpoints | 0 / 7 | — (all abort off-path) |

Per-trajectory RMS for v2: 3.19 / 5.35 / 5.28 / 5.22 / 5.48 / 3.57 mm.

**Both controllers fail on the same trajectory** (`..._20260720_162845`), and
neither fails by going off-path — both time out while tracking well (v2: 2.98 mm
RMS at timeout; pure pursuit: 1.13 mm). That signature is hard for the *plant*, not
for the policy — worth investigating separately (likely a tight feature the lagged
speed loop cannot turn through at the commanded speed).

So RL is now **functional but ~1.8× worse than the classical controller** under the
same conditions. It has not earned a hardware slot on accuracy; the bar is pure
pursuit 1.8 mm / BC 2.0 mm on the real robot.

### Hardware deployment (2026-07-27) — the lag work paid off

`drive_closed_loop.py --policy models/rl_10hz_lag_v2.zip --motor-accel 100`,
trajectory `..._20260722_160100`, two runs per speed scale. Sysid re-measured at
accel=100 first: **τ=0.481 s, dead=0.0627 s** — matching the 0.479 the policy
trained on, so the plant is confirmed consistent.

| speed scale | max speed | RMS (run 1 / run 2) | max err | completed? |
| --- | --- | --- | --- | --- |
| 0.4 | 24 mm/s | **3.2 / 3.2 mm** | 6.4 / 6.1 | yes |
| 0.5 | 30 mm/s | **3.5 / 3.3 mm** | 8.7 / 8.5 | yes |
| 0.8 | 48 mm/s | 4.9 / 6.4 mm | 13.6 / 13.6 | **no — aborts mid-trace** |
| 1.0 | 60 mm/s | 9.1 / 7.0 mm | 19.0 / 18.7 | **no — aborts mid-trace** |

Compare the previous deployment (Run 3, before the lag work): **64.4 mm RMS,
diverged**. This is a ~20× improvement and the first RL policy to complete the
signature on the real robot.

**Sim predicted hardware accurately.** Deterministic sim on this trajectory gave
3.57 mm; hardware at scale 0.5 gave 3.3-3.5 mm. Modelling the wheel speed loop
turned the sim from non-predictive into predictive — that is the main result of
this whole line of work.

**Speed is the remaining limit.** The policy works at ≤30 mm/s and breaks above it.
`V_MAX = 0.06` gave it 2× headroom over the expert and it learned to use that
headroom, but the extra speed is not realisable on hardware even with the lag
modelled. Both failures abort in the same region (the tight bottom loop of the S).
Something speed-dependent is still unmodelled — **observation latency (camera +
BLE) is the prime suspect, and the env models none.**

**Systematic residual, visible at every scale:** on the long right-hand sweep the
tip runs consistently ~3-5 mm *below* the target rather than oscillating about it.
A steady bias like that is calibration (tip offset / yaw scale), not dynamics —
worth chasing before more training, since it is a floor on RMS that no retrain
will remove.

**Two caveats:**

1. **It plateaus ~10 points below the ideal-plant runs** (0.85–0.89 vs 0.95–0.96).
   Most of the gain arrived by 50k steps (0.72); the remaining 1.6M bought ~0.15.
   The lag plant is genuinely harder, and this may be near its ceiling at these
   weights.
2. **Reward was negative while success was already ~0.75.** Early episodes ran ~285
   steps; at `w_time 0.25` that is ~-70 of time penalty alone, swamping the +30
   completion bonus. The policy was succeeding *and being paid negatively for it*.
   Reward only crossed zero around 1.1M steps, as episode length fell 285 → 205.
   **Suggested next tune:** raise `completion_bonus` or cut `w_time` for the lag
   regime, so finishing is unambiguously net-positive.

## Run 7 (`PPO_5`) → `rl_A.zip` / `rl_A_best.zip` — speed cap + reward rebalance

2026-07-27, 2M steps in ~50 min. Three changes from Run 6, chosen from that run's
two diagnosed weaknesses (hardware could not realize the trained speed; reward was
negative at 0.79 success). Everything else identical — same warm start, same
`log_std_init -2.5`, same DR, same `w_track 0.10` / `w_action_rate 0.01`.

| parameter | Run 6 | Run 7 | why |
| --- | --- | --- | --- |
| `v_max` | 0.060 | **0.035** | Hardware aborted above ~30 mm/s. The 2× headroom over the expert was unusable, so half the action range was fantasy. Capping puts the whole range inside the achievable envelope instead of relying on `--policy-speed-scale` at deploy time. |
| `completion_bonus` | 30 | **100** | — |
| `w_time` | 0.25 | **0.10** | Together: at ~220-step episodes `w_time 0.25` cost ~-55 against a +30 bonus, so the policy was **succeeding and being paid negatively for it**. Now ~-22 against +100. |
| `early_stop_at` | — | **800k** | Run 6 peaked 0.89 @1368k and decayed to 0.79. Saves the best-success checkpoint separately. |

`vel_lag_tau` also went 0.48 → **0.481** (sysid re-measured at accel=100; the
change is negligible but keeps the config honest).

| step | 50k | 250k | 688k | 1368k | 1974k *(best)* | 2007k (final) |
| --- | --- | --- | --- | --- | --- | --- |
| success_rate | 0.66 | 0.66 | 0.76 | 0.77 | **0.88** | 0.80 |
| ep_rew_mean | -199 | 28 | 146 | 184 | **259** | 189 |
| ep_len_mean | 306 | 316 | 274 | 266 | 218 | 250 |

### Result: 24% more accurate

`evaluate_rl.py --from-fit --frame-skip 50`, all 7 signatures:

| | finished | mean RMS |
| --- | --- | --- |
| Run 6 `rl_10hz_lag_v2` | 6/7 | 4.68 mm |
| **Run 7 `rl_A_best`** | **6/7** | **3.54 mm** |
| Run 7 `rl_A` (final) | 6/7 | 3.56 mm |
| pure pursuit | 6/7 | **2.55 mm** |

Per-trajectory (`rl_A_best`): 2.22 / 3.93 / 4.52 / 3.26 / 4.52 / 2.79 mm.
Gap to pure pursuit narrowed from **1.8× to 1.4×**.

**What actually moved.** `success_rate` barely changed (0.88 vs 0.89) — the entire
gain is accuracy. Two mechanisms:

1. **The speed cap did the heavy lifting.** Episodes went from 9-18 s to 15-25 s:
   the policy trades speed for precision, which is the right trade when the metric
   is tracking error. This is the single most effective knob found so far.
2. **The reward rebalance fixed the incentive.** `ep_rew_mean` -29 → **+260** at
   comparable success. Finishing is now unambiguously profitable.

**Early stopping earned nothing this run** — best (1974k) landed at the very end,
so `rl_A_best` and `rl_A` are within noise (3.54 vs 3.56 mm). Keep it anyway; it
would have saved Run 6's 0.89 peak.

**`..._162845` still does not finish** — but its RMS dropped to **1.65 mm**, the
best single number of any policy on any trajectory. It tracks beautifully and runs
out of time. Pure pursuit fails identically (1.13 mm, timeout). Confirms a plant
limitation, not a policy one.

### Gotcha: `_best.zip` checkpoints need their own config

`v_max` is resolved from the `_config.json` next to the model file. `rl_A_best.zip`
looked for `rl_A_best_config.json`, which did not exist (the run writes
`rl_A_config.json`), so it **silently fell back to the module default 0.060 and
evaluated the policy 1.7× too fast** — reporting 4.36 mm instead of 3.54 mm.

Fixed both ways: `v_max_from_config()` now falls back to the parent stem for
`_best` checkpoints, and `train_rl.py` writes a sibling config next to the best
checkpoint. **Always check the printed `v_max=...` line matches the run config.**

## Run 8 (`PPO_6`) → `rl_B.zip` — observation delay (negative result)

Run 7's config plus `--obs-delay 1`, justified by a measured **0.201 s (2.0 control
ticks)** total loop delay — cross-correlating commanded ω against IMU yaw rate over
8 hardware traces, corr 0.94-0.97, unanimous. Actuation dead time (0.063 s) is
already modelled, so ~1 tick was attributed to sensing.

Evaluated naively, B looks 40% worse than A (4.97 vs 3.54 mm mean). **That
comparison is invalid** — it is the cross-plant error this document already warns
about. The 2×2 on `..._160100`:

| policy \ plant | no obs-delay | obs-delay 1 |
| --- | --- | --- |
| **A** (trained without) | 2.79 | 4.89 |
| **B** (trained with) | 2.74 | **4.55** |

Read down the columns: **within each plant B ≥ A**. Delay-robustness training
worked. The delay itself costs ~75% accuracy for both policies, dwarfing any
policy difference.

**But the premise was wrong.** A, trained with no delay, predicted hardware well
(3.54 mm sim → 3.3-4.0 mm real). If the robot truly had a full tick of sensing
delay, A should have done markedly worse on the bench than its clean-plant sim
said. It did not. The 0.201 s measurement is command→IMU-yaw, which lumps in
actuation dynamics already modelled as `vel_lag_tau`; subtracting only the dead
time over-attributed the remainder to sensing. **Real sensing delay is well under
one tick — leave `--obs-delay 0`** until something measures it directly.

Kept as a negative result: the mechanism works if ever needed, the estimate did not.

---

## Hardware ω sweep on Run 7 (2026-07-27) — and the v_max/omega_max coupling bug

**Capping `v_max` without capping `OMEGA_MAX` is a bug.** Run 7 halved `v_max`
0.06 → 0.035 and left `OMEGA_MAX` at 10.0, which **doubled the ω:v ratio available
during training**. The policy learned to turn twice as hard per unit distance as
the hardware can track. On the robot at full scale it oscillated to **7.0 mm RMS
and aborted off-path** — worse than the policy it replaced, despite being 24%
better in sim.

`--policy-speed-scale 0.5` recovered stability but scales v *and* ω, halving an
already-capped linear speed: 3.9 mm in 18 s. The fix was a new deploy flag,
`--policy-omega-scale`, damping rotation alone at full speed:

| ω-scale | rms | bias | std | bias share | mean \|ω\| | ω chatter | mean \|v\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 7.00 | -1.67 | 6.79 | 6% | 0.79 | 0.490 | 12.4 (**abort**) |
| 0.5 | 3.16 | -2.00 | 2.44 | 40% | 0.25 | 0.087 | 12.8 |
| 0.4 | 2.87 | -1.93 | 2.13 | 45% | 0.23 | 0.057 | 11.6 |
| 0.3 | 2.31 | -1.48 | 1.76 | 42% | 0.17 | 0.030 | 11.0 |
| **0.25** | **1.94** | **-1.07** | **1.62** | **30%** | 0.15 | **0.022** | 10.3 |
| pure pursuit | **1.53** | +0.47 | 1.34 | 11% | 0.29 | 0.030 | 17.2 |

(chatter = mean |ω step-to-step change|, the oscillation measure.)

**1.94 mm is the best RL result to date** — within 27% of the calibrated classical
controller. Rotational smoothness is now *better* than pure pursuit's (0.022 vs
0.030).

**What the residual is.** Bias stays negative (tip inside the curve) and shrinks as
ω is damped, while pure pursuit on the same calibrated robot sits at **+0.47 mm**.
So the offset is the policy's, not the robot's, and it is an artifact of clamping:
the policy requests a turn, receives 25% of it, under-turns, rides inside. Strip
the bias and RL's random component is 1.62 vs pure pursuit's 1.34 — nearly equal.
**The bias is essentially the whole remaining gap, and it exists only because ω is
clamped after training rather than during it.**

Hence `--omega-max` (Run 9). Deploy-time damping has hit its ceiling: chatter is
already below the classical controller's, so there is nothing left for it to fix.

### Calibration paid off, and moved the bar

The systematic right-offset was real and is gone. Pure pursuit before: **1.76 mm
RMS, bias -1.58 (81% systematic)**. After: **1.53 mm, bias +0.47 (11%)**. Note the
bar moved *away* — calibration helps the classical controller more, because RL's
error is dominated by policy behaviour rather than geometry.

### Rule

`v_max` and `omega_max` are a **pair**. Changing one alone changes the curvature
the policy can command per unit distance, which is a different control problem.
`scales_from_config()` now returns both together so they cannot drift apart.

---

## Metric guidance (learned the hard way in Run 6)

**Do not compare `ep_rew_mean` across plants.** Ideal-plant runs finish in ~110
steps; lag-plant runs take ~285. The per-step `w_time` and `w_track` penalties
accumulate over that longer horizon, so identical behaviour scores far lower under
lag. During Run 6 the reward curve (-120 at 250k, vs `PPO_1`'s +140) suggested the
run was tracking Run 5's failure and should be killed — while `success_rate` showed
0.72 versus Run 5's 0.00. **`success_rate` is the cross-plant comparable metric;**
reward is only comparable within a fixed plant and weight set.

Also expect an initial dip below BC in short runs: PPO's value network starts
random and exploration noise perturbs the exact warm start. Judge over hundreds of
thousands of steps, not the first few updates.

## Evaluation protocol

`rollout/success_rate` is measured with **stochastic** actions, randomized initial
pose, and DR active. Deployment is **deterministic** from a fixed start, which
usually scores better. Always confirm with a deterministic eval on the measured
plant before trusting a checkpoint:

```bash
py -3.13 rl/evaluate_rl.py --model models/<name>.zip --from-fit --frame-skip <fs>
```

`--from-fit` pulls τ/dead from `rl/deploy/sysid/sysid_fit_speed.json`. **The default
plant is ideal and flatters a policy badly** — every checkpoint through Run 5
finishes on the ideal plant and aborts off-path under the measured lag:

| Policy | fs | ideal plant | sysid lag plant |
| --- | --- | --- | --- |
| `rl_10hz_lag` | 50 | abort @0.7s (11.53 rms) | abort @0.4s (12.78 rms) |
| `rl_dr_10hz` | 50 | finished 4.1s (1.28 rms) | abort @11.1s (8.10 rms) |
| `rl_dr_policy` | 10 | finished 3.7s (3.48 rms) | abort @1.9s (9.11 rms) |
| `rl_policy` | 10 | finished 3.2s (2.92 rms) | abort @3.2s (9.21 rms) |
| `rl_10hz_lag_v2` | 50 | — | **finished 13.4s (3.57 rms)** |
| `rl_A_best` | 50 | — | **finished 17.8s (2.79 rms)** |
| pure pursuit *(baseline)* | — | finished 13.4s (**1.75** rms) | finished 16.3s (**2.59** rms) |

Trajectory `target_trajectory_20260722_160100.npz`, deterministic, fixed start.
Pure pursuit degrades gracefully under lag and still finishes — the plant model is
sound, so aborts are a policy failure, not a broken sim.

**Bar to beat on hardware:** pure pursuit **1.8 mm** / BC **2.0 mm** RMS
(30 mm/s, 6 mm lookahead) — see `bc_vs_pure_pursuit.md`.
