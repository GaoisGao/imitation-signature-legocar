# imitation-signature-legocar

A LEGO Education differential-drive car that copies a human's handwritten
**signature**. A person draws a signature under an overhead camera; the recorded
path is followed in a MuJoCo simulation by a classical controller, whose
behaviour is then distilled into a behaviour-cloning (BC) policy and, finally, a
reinforcement-learning (RL) policy.

> **Status:** initial documentation snapshot pushed to preserve the working
> version before further testing and modification. Trained policies and a sample
> signature are included so each phase runs out-of-the-box.

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
| `drive_closed_loop.py` | Real-robot drive, **closed-loop** (overhead-camera tip position + IMU heading), pure pursuit in paper mm |
| `lelib.py`, `motor_dashboard.py` | LEGO hardware wrapper, motor-tuning dashboard |
| `view_trajectory.py`, `trajectory_io.py` | Trajectory plotting and `.npz` IO |
| `learning/` | Phases 2–3: behaviour cloning (model, data collection, training, eval) |
| `rl/` | Phase 4: Gymnasium env, PPO training, evaluation, deploy, mjlab port |
| `models/` | Trained policies (`bc_policy.pt`, `rl_policy.zip`) |
| `datasets/trajectories/`, `datasets/plots/` | Raw recordings: trainable `.npz` + `.png` visualizations |
| `datasets/sim_traces/` | `track_trajectory.py` sim outputs: traced path `.npz` + verification `.png` |
| `datasets/closedloop_traces/` | `drive_closed_loop.py` outputs: per-tick log `.npz` + target-vs-actual trace `.png` |

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
