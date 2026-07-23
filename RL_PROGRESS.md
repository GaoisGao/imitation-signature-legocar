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
**In progress:** frequency-matched retrain `--frame-skip 50`.

Hardware baselines (the bar to beat): pure pursuit **1.8 mm** / BC **2.0 mm**
(30 mm/s, 6 mm lookahead). See `bc_vs_pure_pursuit.md`. (The mjlab sim numbers
above are NOT directly comparable — different action space.)

## Next steps (in order)

1. ~~Record nominal vs DR (sim).~~ **DONE** — DR free in nominal accuracy.
2. ~~Eval-under-noise (mjlab).~~ **DONE** — obs-noise DR tightens the tail under noise.
3. ~~Port DR to SB3, train, deploy on robot.~~ **DONE, but the policy diverges on
   hardware** — control-frequency mismatch (see the SB3 hardware section above).
4. **← CURRENT: frequency-matched SB3 retrain** — `--frame-skip 50` (10 Hz), light
   DR (`--obs-noise 0.03`), action smoothing (`--w-action-rate 0.1`), warm-start
   from BC → `models/rl_dr_10hz.zip`. Sim quick-eval, then deploy vs pure pursuit
   (1.8) / BC (2.0). May need reward retuning (5× fewer steps per episode).
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
