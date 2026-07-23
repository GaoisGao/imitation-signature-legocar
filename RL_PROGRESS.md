# RL phase — progress tracker

**Last updated:** 2026-07-23

**One-line status:** mjlab GPU training works (~270–280k steps/s). v1 obs-noise DR
trained to **0.861 mm**, matching the clean nominal baseline (0.778–0.915 mm) — so
DR is essentially free in nominal accuracy. Next: **eval-under-noise** (does DR hold
where nominal degrades?), then v2 DR (latency + physical params).

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

Hardware baselines (reference only — **NOT directly comparable**: sim vs. real, and
mjlab uses a wheel-effort action, not (v,ω)): pure pursuit **1.8 mm**, BC **1.9 mm**
(both at 30 mm/s, 6 mm lookahead). See `bc_vs_pure_pursuit.md`.

## Next steps (in order)

1. ~~Record the nominal result and compare to DR.~~ **DONE:** nominal 0.778–0.915 mm
   (clean) vs DR 0.861 mm (noisy) → DR is essentially free in nominal accuracy.
2. **Build eval-under-noise** *(Claude)*: force `enable_corruption` ON in a play/eval
   run so we can measure nominal-vs-DR tracking **under noise** — the real robustness
   test. Expect nominal to degrade, DR to hold.
3. **v2 domain randomization** *(Claude)*: latency (`delay_*` obs fields) + physical
   DR via `events` (`dr.effort_limits` motor strength, `dr.geom_friction`,
   `dr.body_mass`, `dr.dof_damping`, mid-episode `push_by_setting_velocity`).
4. **Later, at the work site:** train the deployable **SB3 (v,ω)** policy
   (`rl/train_rl.py`) with matching DR and compare on the real robot vs BC / pure
   pursuit (`drive_closed_loop.py --policy`).

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
