# RL policy for signature tracing

How the signature-copying task becomes a reinforcement-learning problem, building
directly on the existing pipeline (Phase 1 camera trajectory -> Phase 2 pure-pursuit
expert -> Phase 3 behavior cloning). Answers to the questions in NOTES.md are woven
in below.

## The MDP formulation (NOTES Question 1)

The starting point is `track_trajectory.SignatureTracker`: it already *is* the
environment. It owns the MuJoCo model, exposes a `controller(obs) -> (v, omega)`
hook (the exact slot `learning/evaluate_bc.py` plugs the BC policy into), and steps
the physics. Turning it into an RL problem means wrapping it as a `gymnasium.Env`
and adding a reward — nothing about the physics or the interfaces changes.

| MDP piece | Definition | Where it comes from |
|---|---|---|
| Observation | `[dx_local, dy_local, dist_to_final, at_end]` (optionally + wheel speeds, previous action, next-k path points in the local frame) | `SignatureTracker._build_observation` — identical to what BC trained on |
| Action | `(v, omega)` chassis-frame command, passed through the same PI wheel-velocity inner loop | `SignatureTracker.step` — identical to the expert/BC interface |
| Reward (per step) | `- w_track * d(tip, path)^2 + w_prog * (arc-length progress) - w_rate * ||a_t - a_{t-1}||^2 - w_time` + completion bonus at the end | `compute_tracking_error_mm` already computes the tracking term |
| Episode | One signature; reset places the car at the (perturbed) path start; terminate on `finish_tol` or `max_time` | `_reset_pose` + `init_xy_noise` / `init_yaw_noise` already implement the reset randomization |
| Domain randomization | At each reset, jitter the XML's `TUNE:` parameters: masses, wheel/paper/tip friction, motor gear, add actuator delay/noise | Not yet implemented — this is the piece NOTES marks as planned |

The crucial design choice was already made in Phase 2: because the observation is
*task-relative* (lookahead target in the chassis frame) rather than absolute world
position, one policy generalizes across signatures. Train on many trajectories
(all recorded `target_trajectory_*.npz` plus procedurally generated scribbles) with
each parallel env holding a different one.

Why RL can beat BC here: BC's ceiling is the pure-pursuit expert (its loss is
imitation of expert *actions*), while RL's objective is the actual task metric
(tracking error of the *tip*), so it can exceed the expert — and mid-episode
disturbances + domain randomization during training give it the robustness BC
lacks by construction.

## How BC and pure pursuit bootstrap RL (NOTES Question 2)

Three standard bridges, in increasing order of coupling:

1. **Warm start (recommended first):** initialize the PPO policy's mean network
   from `models/bc_policy.pt`. `learning/bc_model.BCPolicy` is a 4 -> 64 -> 64 -> 2
   MLP — make SB3's `policy_kwargs=dict(net_arch=[64, 64])` match, copy the
   weights, and reuse the saved observation `Normalizer` (as a `VecNormalize`
   wrapper or baked into the env). PPO then starts from a policy that can already
   trace, so exploration refines instead of flailing.
2. **Residual RL:** `action = pure_pursuit(obs) + policy(obs)`. The policy only
   learns a correction; the expert guarantees a sane baseline from step one.
   Smallest exploration burden, but the deployed controller keeps the expert
   in the loop.
3. **Expert-anchored regularization / DAgger:** add a penalty for straying from
   the expert's action early in training (annealed to zero), or interleave
   expert relabeling. More machinery, usually unnecessary if 1. works.

Pure pursuit additionally remains the *evaluation baseline*: every RL result is
reported side by side with the expert via the same paired-seed protocol
`learning/evaluate_bc.py --compare-expert` already uses.

## Verifying sim-to-real transfer (NOTES Question 3)

The Phase 1 camera rig is the measurement instrument — the loop closes with no new
hardware:

1. Deploy the policy on the real car (the `(v, omega)` -> wheel-speed conversion is
   the same differential-drive math `SignatureTracker.step` does; `motor_dashboard.py`
   already drives the real Double Motor).
2. Put a red marker on the real pencil, record the drawn signature with
   `webapp.py` / `record_trajectory.py` + `coordinate_plane.py`, exactly like a
   human demonstration.
3. Compute `compute_tracking_error_mm` of the real trace against the same target
   path, and compare the sim-vs-real error distributions (same plots as
   `bc_eval_*.png`). `track_trajectory.simulate_reference` also gives simulated
   yaw / wheel-angle profiles to compare against the robot's IMU + encoder logs.

If real error >> sim error, widen domain randomization on the parameters that
plausibly differ (friction, motor gain, latency) and retrain.

## Expected performance ordering (NOTES Question 4)

- **Pure pursuit (+ PI inner loop):** near-optimal in the nominal sim (it has the
  exact model); degrades under model mismatch or disturbances it was never
  designed for; no learning needed. The PI wheel loop is the "PID" of this
  project and stays underneath every method.
- **BC:** matches the expert on-distribution; degrades off-distribution
  (initial-pose recovery data helps, but only for start-pose error).
- **RL (PPO):** can exceed the expert on tracking error and tolerate mid-run
  pushes / parameter shifts if trained with disturbances + domain randomization;
  costs training time and reward tuning.
- Other candidates worth a row in the comparison table: DAgger (fixes BC's
  distribution shift cheaply), residual RL (best sample efficiency), MPC
  (strong but needs the model online).

## Environment-setup improvements (NOTES Question 5)

- Pen-lift detection (track marker height or pen-down pressure proxy) so
  multi-stroke signatures don't become one connected line.
- Keep the human timing profile (Phase 1 records `t`, but the tracker resamples
  by arc length at constant speed) if reproducing dynamics matters.
- Camera intrinsic/distortion calibration on top of the homography for accuracy
  at the sheet edges.
- Feed upcoming path curvature into the observation so the policy can anticipate
  tight cursive loops instead of reacting at the lookahead.

## Environment / dependencies

Installed and verified on this machine (Python 3.13, 16-core CPU):

```
py -3.13 -m pip install gymnasium stable-baselines3 tensorboard
```

- `gymnasium` — the Env API + vectorization
- `stable-baselines3` — PPO/SAC with `SubprocVecEnv` (one MuJoCo instance per
  process = CPU-parallel training; ~12-16 workers on this machine)
- `tensorboard` — training curves (`py -3.13 -m tensorboard --logdir rl/runs`)
- already present: `mujoco`, `torch` (CPU), `numpy`, `matplotlib`

**About mjlab:** mjlab's *performant* parallel training requires an NVIDIA CUDA
GPU (it is built on MuJoCo Warp), which this machine (Intel Arc) lacks — hence
the SB3 CPU-parallel setup above, which is fully adequate for this small model
(4-dim obs, one small car). However, **empirically verified on this machine
(2026-07-15)**: mjlab 1.5.1 installs and runs its evaluation/play path on native
Windows with pure-CPU Warp, including the browser-based viser viewer at
`http://localhost:8080`. Requirements: `CUDA_VISIBLE_DEVICES=""` (forces the
CPU Warp backend), `PYTHONUTF8=1` (the demo prints emoji, which crashes on a
GBK console), and `--device cpu`. Two caveats: mjlab has **no training-time
visualization** (per its FAQ — the viser viewer is for evaluation/play; they
recommend W&B video logging for training), and Windows support is officially
"preliminary". Seeing the LEGO car in that viewer requires porting the env to
mjlab's API (robot cfg from lego_car_with_pencil.xml + a manager-based task).
On a Linux + NVIDIA machine (lab workstation / cluster), the full mjlab
training route is:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --package signature-rl && cd signature-rl
uv add mjlab
uv run demo          # verifies the GPU pipeline end to end
```

The MDP above ports directly: the observation/action/reward definitions are
framework-agnostic, and `lego_car_with_pencil.xml` loads unchanged.

## File layout

```
rl/
  README.md            (this file)
  TRAINING_LOG.md      run-by-run record: what was trained, what went wrong,
                       what changed between runs
  signature_env.py     gymnasium.Env wrapping SignatureTracker: [-1,1]^2 action
                       scaling, fixed obs scaling, frame skip (50 Hz control),
                       reward, episode logic, random trajectory per reset,
                       optional domain randomization. Also make_sb3_controller()
                       to plug a trained SB3 policy into the SignatureTracker
                       controller hook for evaluation/deployment.
  train_rl.py          SB3 PPO + SubprocVecEnv (CPU-parallel), --warm-start
                       models/bc_policy.pt, --domain-rand, TensorBoard logging,
                       post-training quick-eval plot
  evaluate_rl.py       (planned) same protocol as learning/evaluate_bc.py:
                       paired seeds, --compare-expert, tracking-error plots
  deploy/              real-robot deployment. openloop_deploy.py: replay the
                       trained SB3 policy's wheel-speed tape on the physical
                       Double Motor with no sensing (subcommands: tape / jog /
                       calibrate / drive / compare) to measure the raw
                       sim-to-real dynamics gap. Closed-loop (camera-in-the-
                       loop) deployment is the next step after this baseline.
  mjlab_port/          the car as mjlab tasks, browser viser viewer at
                       http://localhost:8080, CPU-only capable:
                       - Mjlab-LegoCar-Drive: scripted visualization fleet
                         (play_car.py, default)
                       - Mjlab-LegoCar-Signature: the full signature-tracing
                         MDP (signature_mdp.py: per-env recorded paths as a
                         command term, accuracy-gated progress reward, local-
                         window error, off-path/finish terminations - same
                         Run-3 semantics as signature_env.py, but the action
                         is the two wheel efforts directly, no PI loop)
                       Train:  .venv-mjlab\Scripts\python.exe rl\mjlab_port\train_car.py
                       Watch:  .venv-mjlab\Scripts\python.exe rl\mjlab_port\play_car.py
                                 --task signature --checkpoint latest
                       The trained-mode viewer has a checkpoint dropdown that
                       hot-swaps any model_*.pt from the run folder, so you
                       can watch the policy improve WHILE training runs.
                       Honest caveat: CPU-Warp throughput here (~85 env-steps/s
                       at 8 envs) is far below the SB3 stack (~2000+/s), so
                       serious training still belongs on SB3 locally or mjlab
                       on an NVIDIA machine; this port's value is the viewer
                       and GPU-readiness.
```

## Training

```
# full run: warm-start from BC, domain randomization, 8 parallel envs, 2M steps
py -3.13 rl/train_rl.py --warm-start models/bc_policy.pt --domain-rand

# monitor training curves (ep_rew_mean, success_rate)
py -3.13 -m tensorboard.main --logdir rl/runs
```

Saves `models/rl_policy.zip` and a quick-eval plot `rl/rl_train_quick_eval.png`
(one deterministic episode from the nominal start). All reward weights, noise
levels, and PPO hyperparameters are CLI flags — see `py -3.13 rl/train_rl.py -h`.

Notes:
- The warm start is exact: the PPO policy's deterministic output initially
  equals the BC policy's (verified to ~1e-7), with the BC observation
  normalizer folded into the first layer and action scaling into the last.
- Expect an initial *dip* below BC performance in short runs: PPO's value
  network starts random and exploration noise (log_std_init -1.0, std ~0.37)
  perturbs the warm start before the policy improves past it. Judge runs by
  `success_rate` and `ep_rew_mean` over hundreds of thousands of steps, not
  the first few updates.
- Reward-weight caution: the defaults are the product of two failed balances —
  see rl/TRAINING_LOG.md. Keep tracking-well net-positive per step (or the
  policy learns to dive off the path to end the episode cheaply), and keep the
  tracking penalty quadratic (a linear penalty made corner-cutting at full
  speed reward-optimal in Run 1).
- Throughput on this machine: ~1600 policy steps/s with 4 envs (~16k physics
  steps/s); 2M timesteps is roughly a 15-25 minute run with 8 envs.
