# imitation-signature-legocar

A LEGO Education differential-drive car that copies a human's handwritten
**signature**. A person draws a signature under an overhead camera; the recorded
path is followed in a MuJoCo simulation by a classical controller, whose
behaviour is then distilled into a behaviour-cloning (BC) policy and, finally, a
reinforcement-learning (RL) policy.

> **Status:** initial documentation snapshot pushed to preserve the working
> version before further testing and modification. Trained policies and a sample
> signature are included so each phase runs out-of-the-box.

## Closed-loop demo (real robot, overhead camera + IMU)

All three controllers tracing the **same signature** in one session — same robot,
same calibration, same battery charge. Closed-loop throughout: camera-measured
pencil-tip position plus IMU heading, in the paper mm frame.

| Pure pursuit | Behaviour cloning | Reinforcement learning |
| --- | --- | --- |
| ![pure pursuit closed-loop trace](datasets/closedloop_traces/demo_pp.gif) | ![BC policy closed-loop trace](datasets/closedloop_traces/demo_bc.gif) | ![RL policy closed-loop trace](datasets/closedloop_traces/demo_rl.gif) |
| ![pure pursuit trace vs target](datasets/closedloop_traces/demo_pp.png) | ![BC policy trace vs target](datasets/closedloop_traces/demo_bc.png) | ![RL policy trace vs target](datasets/closedloop_traces/demo_rl.png) |
| **1.6 mm** RMS · 3.7 mm max | **1.7 mm** RMS · 4.1 mm max | **2.0 mm** RMS · 4.7 mm max |

## The three controllers compared

| | RMS | max | time | mean speed | ω chatter | systematic bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Pure pursuit** (classical) | **1.6 mm** | **3.7 mm** | **8.1 s** | **18.5 mm/s** | 0.043 | -0.39 mm |
| **Behaviour cloning** | 1.7 mm | 4.1 mm | 8.1 s | 18.8 mm/s | 0.038 | **-0.04 mm** |
| **Reinforcement learning** | 2.0 mm | 4.7 mm | 15.6 s | 9.1 mm/s | **0.015** | -0.99 mm |

(ω chatter = mean step-to-step change in commanded angular rate — the oscillation
measure from `rl/deploy/trace_bias.py`. Bias = mean signed cross-track error;
negative means riding inside the curve.)

**Pure pursuit is still the one to beat.** It has an analytic model of the
geometry, so it needs no data and generalizes to any path for free. Best accuracy
and fastest traversal.

**BC matches it, which is the point.** Within 0.1 mm and identical speed — the
distillation transferred to hardware essentially losslessly. It cannot *exceed*
pure pursuit, because pure pursuit is its supervision target. What it buys is a
4→64→64→2 MLP that runs anywhere, with no lookahead geometry to tune.

**RL is close but slower, and gets there differently.** It is the only controller
never shown the expert's actions — it learned from reward alone in simulation, and
had to cross the sim-to-real gap on its own. Its commands are **2.5× smoother**
than either classical method (0.015 vs 0.038-0.043), but it trades speed for
accuracy: 15.6 s against 8.1 s. Its residual is dominated by a -0.99 mm inward
bias (corner-cutting), where BC's is nearly zero.

The honest summary: **classical control wins on this task.** The signature-tracing
problem is exactly what pure pursuit was designed for — a known path, good state
feedback, no contact dynamics. The learned policies are worth building because they
extend to settings where no such model exists, and RL closing to within 0.4 mm of a
purpose-built analytic controller, from reward alone, is the result worth reporting.

Getting RL there took three fixes, each worth roughly an order of magnitude — the
robot's wheel speed loop has a **0.48 s lag** the simulator did not model,
exploration noise must scale with **control period**, and the speed and angular
ceilings must be capped **together**. First hardware deployment: 64.4 mm and
diverging. See [rl/TRAINING_LOG.md](rl/TRAINING_LOG.md) for the run-by-run record.

Reproduce (recording the newest signature is picked up automatically):

```bash
py -3.13 drive_closed_loop.py drive --card-serial 2312 --card-color magenta   --trajectory datasets/trajectories/target_trajectory_20260722_160100.npz   --motor-accel 100 --speed 30 --lookahead 6                                    # pure pursuit
  #  ... --policy models/bc_policy.pt                                           # BC
  #  ... --policy models/rl_A_best.zip --policy-omega-scale 0.2                 # RL
```

## The pipeline

1. **Record** — track a red pen tip under an overhead camera and map camera
   pixels to paper-millimetre coordinates via ArUco-marker homography.
2. **Classical control** — follow the recorded path in a MuJoCo sim with a
   pure-pursuit controller (with feedback linearization for the pencil tip, which
   trails 73 mm behind the chassis).
3. **Behaviour cloning** — distill the controller into a small MLP policy
   (4 → 64 → 64 → 2).
4. **Reinforcement learning** — train a PPO policy (Stable-Baselines3),
   warm-started from the BC policy.

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
py -3.13 -m pip install -r requirements.txt
```

`mjlab`, `rsl-rl-lib`, and the LEGO `legoeducation` hardware API are optional
(commented in `requirements.txt`) — needed only for the mjlab RL port and
real-robot deployment.

## Quick start

The trained `models/bc_policy.pt` and `models/rl_policy.zip` are included, so the
evaluate steps run immediately without retraining.

```bash
# Phase 1 — record a signature (needs a camera + printed ArUco sheet)
py -3.13 webapp.py

# Core — follow a signature with the classical controller (pure pursuit)
py -3.13 track_trajectory.py --trajectory datasets/trajectories/target_trajectory_20260710_111912.npz

# Phase 2/3 — collect expert data, then train & evaluate behaviour cloning
py -3.13 learning/collect_expert_data.py --all --episodes-per-traj 5 --output datasets/expert_dataset.npz
py -3.13 learning/train_bc.py --dataset datasets/expert_dataset.npz --epochs 300
py -3.13 learning/evaluate_bc.py --trajectory datasets/trajectories/target_trajectory_20260710_111912.npz --compare-expert

# Phase 4 — train & evaluate RL (PPO, warm-started from BC)
py -3.13 rl/train_rl.py --warm-start models/bc_policy.pt --domain-rand
py -3.13 rl/evaluate_rl.py --trajectory datasets/trajectories/target_trajectory_20260710_111912.npz --view
```

## Layout

| Path | Role |
| --- | --- |
| `track_trajectory.py` | **Core:** MuJoCo sim + pure-pursuit controller (imported everywhere) |
| `lego_car_with_pencil.xml` | **Core:** simulated car + paper model (199 × 137 mm) |
| `record_trajectory.py`, `coordinate_plane.py`, `webapp.py` | Phase 1: camera tracking + ArUco homography + integrated capture |
| `run_lego_signature.py` | Real-robot drive, **open-loop** dead reckoning from a trajectory |
| `drive_closed_loop.py` | Real-robot drive, **closed-loop** (overhead-camera tip position + IMU heading), pure pursuit in paper mm — see [docs/closed_loop_pure_pursuit.md](docs/closed_loop_pure_pursuit.md) |
| `lelib.py`, `motor_dashboard.py` | LEGO hardware wrapper, motor-tuning dashboard |
| `view_trajectory.py`, `trajectory_io.py` | Trajectory plotting and `.npz` IO |
| `learning/` | Phases 2–3: behaviour cloning (model, data collection, training, eval) |
| `rl/` | Phase 4: Gymnasium env, PPO training, evaluation, deploy, mjlab port |
| `models/` | Trained policies (`bc_policy.pt`, `rl_policy.zip`) |
| `datasets/trajectories/`, `datasets/plots/` | Raw recordings: trainable `.npz` + `.png` visualizations |
| `datasets/sim_traces/` | `track_trajectory.py` sim outputs: traced path `.npz` + verification `.png` |
| `datasets/closedloop_traces/` | `drive_closed_loop.py` outputs: per-tick log `.npz` + target-vs-actual trace `.png` |
| `datasets/bc_policy/` | `learning/evaluate_bc.py` outputs: BC eval trace `.npz` + BC-vs-expert `.png` |

## Key design notes

- **Observation (4-dim, task-relative):** `[dx_local, dy_local, dist_to_final, at_end]`
  — chosen so a policy generalizes across signatures.
- **Action (2-dim):** `[v (m/s), omega (rad/s)]` chassis command; a PI wheel-velocity
  inner loop turns this into motor torques.
- **Paper frame == world frame:** the paper geom is centered at the world origin
  and matches the printed 199 × 137 mm ArUco sheet, so paper-mm trajectories map
  directly.
- Timing is discarded: paths are resampled by arc length and followed at constant
  speed — the pipeline copies the signature's *shape*, not the demonstrator's speed.

## Known scope limits

- No pen-lift detection (a multi-stroke signature becomes one connected line).
- Only the longest recording in a file is used downstream.
- Domain randomization currently covers only initial-pose noise — no mid-episode
  disturbances, sensor noise, or physical-parameter perturbation yet.
