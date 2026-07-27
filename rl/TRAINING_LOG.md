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
| 7 | `PPO_5` | `rl_A_best.zip` | 10 | lag | -2.5 | 0.88 | speed cap + reward fix → 6/7, 3.54 mm sim. **Deployment of record:** `--policy-omega-scale 0.2` → **1.4-1.7 mm, matches pure pursuit** (but 2x slower) |
| 8 | `PPO_6` | `rl_B_best.zip` | 10 | lag+delay | -2.5 | 0.79 | obs-delay: mechanism works, estimate was too high (negative result) |
| 9 | `PPO_7` | `rl_C_best.zip` | 10 | lag | -2.5 | 0.85 | `omega_max` capped at training → best in sim (2.59 mm) but **LOSES on hardware** (3.1 vs 2.15 mm) |

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

## Run 9 (`PPO_7`) → `rl_C_best.zip` — omega capped at training

**Best in sim, beaten on hardware.** See the hardware subsection below before
reusing this configuration.

Run 7's config plus `--omega-max 2.5` (= 10.0 x the best deploy clamp, 0.25).
One variable changed. No `--obs-delay` (Run 8 showed the estimate was too high).

| step | 50k | 250k | 827k | 1572k | 2007k (final) |
| --- | --- | --- | --- | --- | --- |
| success_rate | 0.34 | 0.67 | 0.77 | 0.73 | **0.85** |
| ep_rew_mean | -397 | 181 | 253 | 234 | **286** |
| ep_len_mean | 543 | 362 | 294 | 319 | 270 |

Slowest starter of any run (0.34 @50k, ep_len 543 - a tight omega budget makes the
task genuinely harder early) but the strongest finisher: it was still improving at
2M, unlike Runs 6 and 7 which peaked mid-run and decayed.

### Result

| | finished | mean RMS | mean max-err |
| --- | --- | --- | --- |
| Run 6 `rl_10hz_lag_v2` | 6/7 | 4.68 mm | - |
| Run 7 `rl_A_best` | 6/7 | 3.54 mm | 9.07 mm |
| **Run 9 `rl_C_best`** | **6/7** | **2.59 mm** | **5.65 mm** |
| Run 9 `rl_C` (final) | 6/7 | 3.01 mm | 6.81 mm |
| **pure pursuit** | 6/7 | **2.55 mm** | - |

Per-trajectory (`rl_C_best`): 1.89 / 2.98 / 3.16 / 2.84 / 2.76 / 1.93 mm.

In **sim**, the first RL policy to match the classical controller (2.59 vs 2.55).
27% better than Run 7, max error nearly halved (9.07 -> 5.65 mm). **This ranking
did not survive contact with the robot** - see below.

In sim it looked like in-training capping beats deploy-time clamping: Run 7 needed
throttling to 25% of its commanded omega and still carried a -1.07 mm under-turn
bias, while Run 9 learned inside a 2.5 rad/s budget. **On hardware the opposite
held** - the clamp also suppresses action noise, which the cap does not.

**Early stopping earned its keep**: `rl_C_best` (827k, the first rollout to touch
0.85) evaluates at **2.59 mm** vs the final checkpoint's **3.01 mm** - a 14% gain
that would have been silently lost. Note the final rollout *also* reads 0.85
success with a higher reward (286 vs 253), yet traces 16% worse. **success_rate and
ep_rew_mean did not identify the better policy here** - only the deterministic eval
did. Evaluate both checkpoints, every time.

**Cost:** episodes run ~305 steps vs Run 7's ~220 (17.6 s vs 12.5 s on
`..._160100`). A tighter omega budget means wider turns, so it trades time for
accuracy. Correct trade when the metric is tracking error, but it is why
`success_rate` (0.85) sits below Run 7's 0.88 - not a regression.

`..._162845` still times out, now at 2.49 mm. Unchanged conclusion: plant limit.

### Hardware: Run 9 LOSES to Run 7. The sim ranking inverted.

Same robot, same battery charge, same trajectory, two trials each. Run 7 deployed
with `--policy-omega-scale 0.25`; Run 9 with `--policy-speed-scale 0.65` to match
linear speed (Run 9 unthrottled runs ~15.5 mm/s vs Run 7's 10.3).

| | drive rms | bias | **std** | mean \|omega\| | **chatter** | mean \|v\| | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Run 7 @ omega-scale 0.25** | **2.2 / 2.1** | -1.26 | **1.62** | 0.155 | **0.022** | 10.5 | 14.1 s |
| Run 9 @ speed-scale 0.65 | 3.2 / 3.1 | -1.37 | 2.75 | 0.165 | 0.033 | 9.6 | 15.2 s |
| Run 9 unthrottled | 3.1 / 2.9 / 3.0 | -1.43 | 2.58 | 0.25 | 0.078 | 15.5 | 9.8 s |
| pure pursuit | 1.5 | +0.47 | 1.34 | 0.29 | 0.030 | 17.2 | 7.8 s |

(bias/std/chatter from `trace_bias.py`, which uses signed cross-track error and
reads ~0.1 mm below the drive script's nearest-point rms.)

**Sim said Run 9 by 27% (2.59 vs 3.54 mm). Hardware says Run 7 by 33%.** Run 7 wins
on accuracy, smoothness AND speed at once - there is no hidden trade.

**Why: a deploy-time clamp attenuates the policy's action noise; a training-time
cap does not.** Bias is nearly identical (-1.26 vs -1.37), so the in-training cap
*did* fix the under-turn bias as intended. But Run 9 is 70% noisier (std 2.75 vs
1.62, chatter 0.033 vs 0.022) and that swamps the gain. Run 7's raw policy commands
~0.6 mean |omega|; multiplying by 0.25 shrinks its *jitter* 4x along with its
magnitude. Run 9 learned to use its 2.5 rad/s budget, so its small rapid
corrections arrive at full amplitude with nothing suppressing them.

> **Capping the action space bounds what the policy CAN command. Clamping the
> output low-passes what it DOES command. Only the second buys smoothness** - and
> on this plant, with its 0.48 s wheel lag, smoothness is what pays.

Both use the same angular authority (mean |omega| ~0.16). The entire difference is
how jittery that usage is.

**Deployment configuration of record:** `rl_A_best.zip` with
`--policy-omega-scale 0.25` - **2.1-2.2 mm**, vs calibrated pure pursuit's 1.5 mm.

**Implication for the next run:** the lever is `--w-action-rate` (currently 0.01,
cut from 0.05 during the 10 Hz reweighting) - the term that directly penalizes
jittery actions, asking the policy to produce the smoothness the clamp currently
supplies for free. NOT `omega_max`: Run 9 already has less angular authority than
Run 7 and is still rougher, so it is not the binding constraint.

---

## Generalization + the omega floor (2026-07-27, one session, one battery charge)

All hardware testing before this used a single signature (`..._160100`). This
session swept `--policy-omega-scale` on Run 7 across **two unseen signatures**,
with pure pursuit as the control. All numbers from `trace_bias.py` (signed
cross-track; reads ~0.1 mm below the drive script's nearest-point rms).

**`..._111912` (177 pt)**

| controller | rms | bias | chatter | mean \|v\| | time |
| --- | --- | --- | --- | --- | --- |
| Run 7 @ omega0.25 | 1.65 | -0.95 | 0.023 | 10.6 | 18.2 s |
| **Run 7 @ omega0.2** | **1.30** | -0.44 | 0.018 | 9.5 | 20.5 s |
| Run 7 @ omega0.15 | 1.38 | -0.45 | 0.011 | 8.0 | 25.5 s |
| pure pursuit | **1.30** | +0.12 | 0.047 | **18.9** | **9.1 s** |

**`..._140212` (106 pt)**

| controller | rms | bias | chatter | mean \|v\| | time |
| --- | --- | --- | --- | --- | --- |
| Run 7 @ omega0.25 | 2.01 | -0.70 | 0.022 | 8.3 | 12.9 s |
| Run 7 @ omega0.2 | 1.62 | -0.56 | 0.014 | 7.1 | 15.3 s |
| **Run 7 @ omega0.15** | **1.45** | -0.62 | 0.009 | 4.9 | 25.6 s |
| pure pursuit | 1.52 | -1.10 | 0.030 | **16.2** | **6.3 s** |

**Mean over both signatures:** omega0.25 **1.83** | omega0.2 **1.46** |
omega0.15 **1.42** | pure pursuit **1.41**.

### Three findings

**1. It generalizes.** Run 7 @ omega0.25 scored 2.1-2.2 mm on the signature every
earlier test used, and 1.65 / 2.01 mm on two it had never traced. Not overfit.

**2. RL matches the classical controller on accuracy.** At omega0.2, 1.46 mm mean vs
pure pursuit's 1.41 - a tie, on unseen signatures. From 64.4 mm and diverging this
morning.

**3. RL is far SMOOTHER than pure pursuit.** Chatter 0.009-0.018 vs 0.030-0.047.
The classical controller is the jittery one now.

### The floor is at omega-scale 0.2. Stop there.

Accuracy plateaus between 0.2 and 0.15 (1.46 -> 1.42, inside run-to-run spread)
while **time keeps climbing**: 17.9 s -> 25.6 s mean. On the 106-pt signature 0.15
crawls at 4.9 mm/s, 4x slower than pure pursuit for the same 1.5 mm. Below 0.2 you
are buying nothing and paying in throughput.

**Deployment of record: `rl_A_best.zip --policy-omega-scale 0.2`.**

### The remaining gap is throughput, not accuracy

| | mean rms | mean time |
| --- | --- | --- |
| pure pursuit | 1.41 mm | **7.7 s** |
| Run 7 @ omega0.2 | 1.46 mm | 17.9 s (**2.3x slower**) |

Clamping omega does two things at once: it suppresses the policy's angular jitter
*and* caps cornering rate, which forces slow traversal. We want the first without
the second.

**This reframes Run D.** Training for smoothness is no longer about accuracy - the
clamp already delivers more smoothness than pure pursuit. The goal is omega0.2's
accuracy **without** its speed penalty: a policy intrinsically smooth enough to
need no clamp, free to use its full omega budget. That is `--w-action-rate` up
(0.01 -> 0.08) with `omega_max` left at the Run 7 value and **no deploy clamp**.
Run 9 already established that lowering `omega_max` is the wrong lever.

---

## Where things stand (end of 2026-07-27)

**Best deployable configuration**

```bash
py -3.13 drive_closed_loop.py drive --card-serial 2312 --card-color magenta   --trajectory <signature.npz> --motor-accel 100   --policy models/rl_A_best.zip --policy-omega-scale 0.2
```
~1.46 mm mean across three signatures, matching pure pursuit's 1.41, at ~2.3x the
traversal time.

**The day, in one line each**

| fix | effect |
| --- | --- |
| Model the wheel speed loop (tau=0.481 s, dead=0.063 s) | sim became predictive; 64.4 mm -> 3.2 mm |
| Scale exploration noise to control period (log_std -1.0 -> -2.5) | 0.00 -> 0.89 success |
| Cap `v_max`, rebalance reward (bonus 100, w_time 0.10) | 4.68 -> 3.54 mm sim |
| Fix geometry calibration | pure pursuit 1.76 -> 1.53 mm, bias -1.58 -> +0.47 |
| Damp omega at deploy (`--policy-omega-scale 0.2`) | 3.2 -> 1.46 mm hardware |

**Next: Run D — intrinsic smoothness, to recover the 2.3x speed**

```bash
py -3.13 rl/train_rl.py --warm-start models/bc_policy.pt --domain-rand --obs-noise 0.03   --vel-lag-tau 0.481 --vel-dead-time 0.0627   --v-max 0.035 --w-action-rate 0.08 --completion-bonus 100 --w-time 0.10   --early-stop-at 800000 --output models/rl_D.zip
```
`--omega-max` omitted on purpose (Run 9). Evaluate `rl_D_best` AND `rl_D` - Run 9
showed success_rate and reward do not identify the better checkpoint.

**Success criterion:** ~1.5 mm at pure-pursuit speed (16-19 mm/s), deployed with no
flags. If it needs a clamp to reach 1.5 mm it has not beaten Run 7 @ omega0.2, and
the honest conclusion is that this architecture needs the clamp.

**Open threads**

- `..._162845` times out for **both** RL and pure pursuit while tracking accurately
  (2.49 / 1.13 mm). A plant limit, never diagnosed.
- Sim does not model whatever penalizes angular jitter on the real robot - the one
  place the lag-matched sim is still not predictive (it ranked Run 9 over Run 7;
  hardware disagreed).
- Battery charge shifts traversal speed measurably. Compare runs within a session.

---

## Parameter changes, Run 7 (A) vs Run 9 (C)

Everything not listed is identical: `frame_skip 50`, `log_std_init -2.5`,
`--domain-rand`, `--obs-noise 0.03`, `vel_lag_tau 0.481`, `vel_dead_time 0.0627`,
`w_progress 2.0`, `w_track 0.10`, `err_gate_mm 3.0`, `w_action_rate 0.01`,
`off_path_penalty 30`, `off_path_limit_mm 20`, warm start from BC, 2M steps.

| parameter | Run 7 (A) | Run 9 (C) |
| --- | --- | --- |
| `--v-max` | 0.035 | 0.035 |
| `--omega-max` | 10.0 *(module default, uncapped)* | **2.5** |
| `--completion-bonus` | 100 | 100 |
| `--w-time` | 0.10 | 0.10 |
| `--early-stop-at` | 800k | 800k |

**One variable changed: `omega_max`.**

| | best success | sim mean RMS | sim mean max-err | hardware rms | deploy flags needed |
| --- | --- | --- | --- | --- | --- |
| Run 7 (A) | 0.88 | 3.54 mm | 9.07 mm | **2.1-2.2 mm** | `--policy-omega-scale 0.25` |
| Run 9 (C) | 0.85 | **2.59 mm** | **5.65 mm** | 3.1-3.2 mm | none |

**Run 9 is better in sim and worse on hardware.** The sim models the wheel speed
loop but not whatever penalizes angular jitter on the real robot, so it cannot see
the cost of Run 9's noisier commands. This is the one place the lag-matched sim is
still not predictive - worth remembering before trusting a sim ranking again.

Secondary observation: Run 9 unthrottled traces the signature in **9.8 s at 3.0 mm**
versus Run 7's **14.1 s at 2.15 mm**. If traversal speed ever matters more than
accuracy, Run 9 is the better policy - it is 44% faster for 40% more error.

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
