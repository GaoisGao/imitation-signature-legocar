# RL phase — progress tracker

**Last updated:** 2026-07-23

**One-line status:** mjlab (GPU) DR validated in sim — obs-noise DR tightens the
tail under noise. Pivoted to the deployable **SB3 (v,ω)** policy for the real
robot; it **fails on hardware** (diverges, ~64 mm) due to a **50 Hz-sim → 10 Hz-
hardware control-frequency mismatch** (see [rl_hardware_gap.md](rl_hardware_gap.md)),
NOT the DR. **Now retraining at `--frame-skip 50` (10 Hz-matched).** Classical
baselines still lead on hardware: pure pursuit 1.8 mm, BC 2.0 mm.

---

## Environment (desktop)

- Linux desktop `zhenkai-gao-G457`, **RTX 5070 Ti 16 GB**, driver 595.71, CUDA 13.2.
- Repo cloned at `~/Desktop/imitation-signature-legocar`.
- mjlab venv: **`.venv-mjlab`** — Warp 1.15, torch 2.13.0+cu130, mjlab. All verified
  on the Blackwell `sm_120` GPU.
- **Workflow:** Claude edits on the Windows laptop and pushes to GitHub; you
  `git pull` and run on the desktop. Git is the sync channel.

## Start-of-day resume checklist

1. `ssh <user>@<desktop-ip>` → `cd ~/Desktop/imitation-signature-legocar` → `git pull`
2. Stop auto-suspend (once): `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`
3. Stop auto-reboot (once): edit `/etc/apt/apt.conf.d/50unattended-upgrades`,
   set `Unattended-Upgrade::Automatic-Reboot "false";`
   *(the desktop rebooted mid-training on 07-22 from unattended-upgrades, not sleep)*
4. Always train inside tmux: `tmux new -s rl` → `source .venv-mjlab/bin/activate`
   *(detach `Ctrl-b d`, reattach `tmux attach -t rl`)*

## Done

- [x] **GPU enabled** in `rl/mjlab_port/train_car.py` (commit `64e7aa9`): default GPU,
  `--cpu` fallback, `--gpu-id`.
- [x] **v1 observation-noise DR** in `rl/mjlab_port/signature_env_cfg.py` (commit
  `5e53618`): additive Gaussian noise on the actor obs — signature (0.05, ≈camera
  tip), base_lin_vel (0.005), base_ang_vel (0.02, ≈IMU), wheel_vel (0.1, ≈encoder).
  Applied **only in training** (actor group `enable_corruption=not play`). Toggle
  with `LEGOCAR_DR=0`.
- [x] GPU throughput ~270–280k steps/s at 4096 envs; 300 iters ≈ 3.5 min.
- [x] `dr_obsnoise` 300-iter run → **tracking_err_mm 0.86, off_path 0%, finished ~72**, converged (action std → 0.12).

## Results

Sim tracking error (mjlab, 300 iters, 4096 envs, 2026-07-22/23). The metric is
computed from the TRUE tip state, so the DR row is tracking accuracy *while acting
on noisy observations*.

| Run (log dir) | Obs noise | tracking_err_mm | Notes |
| --- | --- | --- | --- |
| `2026-07-23_00-00-21_nominal` | OFF | **0.778** | clean baseline (best) |
| `2026-07-22_23-21-21_nominal` | OFF | 0.915 | earlier nominal (run-to-run spread) |
| `2026-07-22_23-46-21_dr_obsnoise` | ON | **0.861** | v1 obs-noise DR, off_path 0% |

**Takeaway:** the DR policy (0.861 mm, *with* noisy obs) sits inside the nominal
run-to-run spread (0.778–0.915 mm, *clean* obs) — so observation-noise DR costs
essentially nothing in nominal accuracy.

### Eval under noise (2026-07-23) — nominal vs DR robustness

`eval_signature.py`, 1024 envs, terminations off (sustained tracking), model_299.

| Checkpoint | clean mean | noise mean | noise p95 | noise max |
| --- | --- | --- | --- | --- |
| `nominal` | **0.582** | 1.393 | 3.441 | 15.998 |
| `dr_obsnoise` | 0.806 | **1.313** | **2.696** | **13.682** |

**Takeaway:** nominal wins on clean obs (0.58 vs 0.81 — DR trades a little clean
accuracy). Under noise, DR degrades *less* (Δ+0.51 vs Δ+0.81) and clearly wins on
the **tail** (p95 2.70 vs 3.44, max 13.7 vs 16.0). So obs-noise DR gives modest,
mainly worst-case robustness at this noise level. Latency + physical DR (v2)
should widen the gap — they model the dominant real-world effects (BLE+camera
lag, friction/motor) that obs noise alone doesn't.

### SB3 (v,ω) policy on the real robot (2026-07-23)

The deployable path. Trained `rl/train_rl.py --warm-start bc --domain-rand
--obs-noise 0.05` → sim quick-eval 3.6 mm (jittery). On hardware via
`drive_closed_loop.py --policy`:

| Policy | speed scale | HW RMS | HW max | result |
| --- | --- | --- | --- | --- |
| `rl_policy.zip` (old, pre-DR) | 1.0 | 67.8 | 108.8 | diverges |
| `rl_dr_policy.zip` (DR) | 1.0 | 64.4 | 84.5 | diverges (same mode) |
| `rl_dr_policy.zip` (DR) | 0.4 | 17.0 | 49.7 | much better, still bad |

Both RL policies diverge the same way; slowing 0.4× cut RMS 64→17. **Root cause:
50 Hz-sim → 10 Hz-hardware control-frequency mismatch (+ aggressive 60 mm/s action),
not the DR.** Full analysis in [rl_hardware_gap.md](rl_hardware_gap.md).
**Superseded** by the wheel speed-loop lag work below.

### Sim-to-real gap 2: wheel speed-loop lag (2026-07-27)

Frequency matching alone was not enough. `rl/deploy/motor_sysid.py` measured the
real speed loop at **τ=0.479 s, dead=0.063 s** (`sysid/sysid_fit_speed.json`);
`track_trajectory.py` now models it (`vel_lag_tau`/`vel_dead_time`) and
`evaluate_rl.py --from-fit` evaluates on it. Under that plant **every checkpoint
through `rl_10hz_lag` aborts off-path**, while pure pursuit still finishes
(1.75 mm ideal → 2.59 mm lagged) — so the plant model is sound and the aborts are
a policy failure.

Second fix: **`log_std_init` -1.0 → -2.5**. At 10 Hz an action is held 100 ms, so
σ=0.37 throws the tip off-path before the next correction. `rl_10hz_lag` (σ=0.37)
scored **0.00 success across all 2M steps**; the identical run at σ=0.08 reaches
**0.85**.

Full run-by-run record, reward-shaping history, and the deterministic
under-lag eval table: **[rl/TRAINING_LOG.md](rl/TRAINING_LOG.md)**.

Hardware baselines (the bar to beat): pure pursuit **1.8 mm** / BC **2.0 mm**
(30 mm/s, 6 mm lookahead). See `bc_vs_pure_pursuit.md`. (The mjlab sim numbers
above are NOT directly comparable — different action space.)

## Next steps (in order)

1. ~~Record nominal vs DR (sim).~~ **DONE** — DR free in nominal accuracy.
2. ~~Eval-under-noise (mjlab).~~ **DONE** — obs-noise DR tightens the tail under noise.
3. ~~Port DR to SB3, train, deploy on robot.~~ **DONE, but the policy diverges on
   hardware** — control-frequency mismatch (see the SB3 hardware section above).
4. ~~Frequency-matched SB3 retrain (`--frame-skip 50`) → `rl_dr_10hz.zip`.~~
   **DONE, still fails** — 0.96 success in sim but aborts at 11.1 s under the
   measured wheel lag. Frequency was necessary, not sufficient.
5. ~~Measure the real speed loop + model it in sim.~~ **DONE** — `motor_sysid.py`
   → τ=0.479 s / dead=0.063 s; `track_trajectory.py` reproduces it.
6. ~~Lag-matched retrain at `--log-std-init -2.5`~~ → **`models/rl_10hz_lag_v2.zip`,
   the first RL policy to finish under the measured lag** (6/7 signatures,
   4.68 mm mean RMS; success 0.00 → 0.89). But pure pursuit on the identical plant
   is 6/7 at **2.55 mm** — RL is still ~1.8× worse and has not earned a hardware slot.
7. ~~Deploy `rl_10hz_lag_v2` on hardware.~~ **DONE — it works.** 3.2-3.5 mm RMS,
   completes the signature at speed scale 0.4-0.5 (≤30 mm/s), vs **64.4 mm and
   diverging** before the lag work. Sim predicted it (3.57 mm sim → 3.3-3.5 mm real).
   Fails above 30 mm/s. Full table in [rl/TRAINING_LOG.md](rl/TRAINING_LOG.md).
8. ~~Run A: cap `V_MAX` 0.06 → 0.035, `completion_bonus` 30 → 100, `w_time`
   0.25 → 0.10, early-stop.~~ **DONE — 4.68 → 3.54 mm (24% better), 6/7 finishing.**
   Gap to pure pursuit down from 1.8× to 1.4×. The speed cap was the dominant
   effect. Full analysis: [rl/TRAINING_LOG.md](rl/TRAINING_LOG.md) Run 7.
9. **← NEXT: Run B — observation latency.** Measured at **2.0 control ticks total
   loop delay** (0.201 s) by cross-correlating commanded ω against IMU yaw rate
   across 8 hardware traces (corr 0.94-0.97, unanimous). Actuation dead time
   (0.063 s) is already modelled, so ~1 tick of additional *sensing* delay:
   train Run A's config plus `--obs-delay 1`. Caveat: that measurement uses the
   IMU path; the camera path feeding dx/dy may be slower, so sweep 2 if 1 helps.
10. **Calibration (no retrain, biggest single win available).** `rl/deploy/trace_bias.py`
   splits a trace's error into constant offset vs oscillation. Every hardware run is
   offset to the RIGHT of travel: pure pursuit bias **-1.58 mm of its 1.76 mm RMS
   (81%)**, RL ~-1.9 mm of 3.2 mm (~35%). Same sign under both controllers at every
   speed → geometry, not policy. Fixing it takes pure pursuit to ~0.8 mm and RL to
   ~2.5 mm for zero training. Bisect: motor asymmetry → IMU yaw bias → lateral tip
   offset.
8. Investigate `target_trajectory_20260720_162845` — **both** RL and pure pursuit
   time out on it while tracking accurately (2.98 / 1.13 mm). A plant limitation,
   not a policy failure.
5. If still short of the classical baselines: cap `V_MAX`/`OMEGA_MAX`, or accept
   that pure pursuit / BC are the better controllers for this task and use RL only
   to study robustness in sim.
6. Optional (mjlab research track): v2 DR = latency (`delay_*`) + physical
   (`dr.effort_limits`/`geom_friction`/`body_mass`) — study, not deployable.

## Key facts / gotchas

- **mjlab port = sim research only.** Its action is raw wheel **efforts** and its obs
  includes `base_lin_vel` — neither exists on the real Double Motor. Use mjlab (GPU
  speed + viser viewer) to develop/validate the DR recipe fast; the hardware-deployable
  policy is the **SB3 (v,ω)** one.
- **DR toggle:** `LEGOCAR_DR=0` → no observation noise (clean baseline).
- **Checkpoints:** `logs/rsl_rl/<experiment>/<timestamp>/model_*.pt`, every 25 iters.
- **Watch it draw (viser):** `python rl/mjlab_port/play_car.py --task signature --checkpoint latest`
  → `http://localhost:8080` (over SSH: `ssh -L 8080:localhost:8080 <user>@<ip>`).

## Commands

```bash
# DR on (default)
python rl/mjlab_port/train_car.py --task signature --num-envs 4096 --run-name dr_obsnoise
# nominal (no noise) A/B baseline
LEGOCAR_DR=0 python rl/mjlab_port/train_car.py --task signature --num-envs 4096 --run-name nominal
# CPU fallback: add --cpu ;  TensorBoard: python -m tensorboard.main --logdir logs/rsl_rl
```

## mjlab API reference (so we skip re-introspecting)

- **Noise** (`mjlab.utils.noise`): `GaussianNoiseCfg(operation, mean, std)`,
  `UniformNoiseCfg(operation, n_min, n_max)`. `operation` e.g. `"add"`.
- **Obs latency** (`ObservationTermCfg` fields): `delay_min_lag`, `delay_max_lag`,
  `delay_hold_prob`, `delay_per_env`, `delay_update_period`, `delay_per_env_phase`.
- **Physical DR** (`mjlab.envs.mdp.dr`) — each takes `env, env_ids, ...` and an
  `asset_cfg=SceneEntityCfg(name="lego_car", ...)` (default name is `"robot"`, so it
  **must be overridden** to our entity `"lego_car"`):
  - `body_mass(ranges, distribution="uniform", operation="scale")`
  - `geom_friction(ranges, ...)`, `dof_damping(ranges)`, `joint_damping(ranges)`
  - `effort_limits(effort_limit_range: tuple, operation="scale")`  ← motor strength
  - `encoder_bias(bias_range: tuple)`
  - `pd_gains(kp_range, kd_range)` (our motors are effort actuators — may not apply)
  - `ranges` is a `Ranges` type — confirm its shape before first use.
- **Events**: `EventTermCfg(func, params, mode, interval_range_s)` from
  `mjlab.managers.event_manager`; `mode` = `"reset"` / `"interval"` / `"startup"`.
  Disturbances: `mdp.push_by_setting_velocity`, `mdp.apply_external_force_torque`.
- **Scene**: entity name = `"lego_car"`; wheel joints `"joint_left"`, `"joint_right"`.

## Tomorrow's starting prompt (paste to a fresh Claude session)

> Continuing the **lego-signature-car** RL phase on my GPU desktop (mjlab). Read
> `RL_PROGRESS.md` — it has the full state, results, and mjlab API reference.
> Current: obs-noise DR (v1) = **0.861 mm** (noisy obs) vs nominal **0.778 mm**
> (clean) — DR is free in nominal accuracy. Next: (1) build **eval-under-noise** to
> prove DR holds where nominal degrades, (2) v2 DR (latency + physical params).
> **Propose a plan before changing code.**
