# RL phase — home workflow (mjlab / GPU / sim-only)

Guide for the RL phase on the **NVIDIA GPU desktop at home**, where there is
**no robot and no overhead camera** — so the focus is entirely on
**training and evaluation in simulation** (mjlab on the GPU). Hardware
validation is deferred to the work-site rig later.

Pure-pursuit and BC are done and validated on hardware; see
[bc_vs_pure_pursuit.md](bc_vs_pure_pursuit.md) and
[rl/README.md](rl/README.md) (the full MDP formulation + file map).

## 1. Get the code

```bash
git clone https://github.com/GaoisGao/imitation-signature-legocar   # or: git pull
```

## 2. What you can / can't do at home

- **Can:** train in **mjlab on the GPU** (MuJoCo-Warp, many parallel envs),
  watch the policy live in the **viser browser viewer** (`http://localhost:8080`,
  hot-swaps checkpoints mid-training), evaluate in sim, iterate on reward and
  domain randomization.
- **Can also:** SB3 CPU-parallel training (`rl/train_rl.py`) — this is the path
  that yields the **deployable `(v, omega)` policy** that plugs into
  `drive_closed_loop.py --policy` later. Fast even without the GPU.
- **Can't:** record new signatures (no camera) or run closed-loop hardware
  deployment. Use the **sample trajectories already committed** in
  `datasets/trajectories/` — no camera needed to train.

## 3. mjlab setup (the GPU path)

mjlab needs an NVIDIA CUDA GPU (it is built on MuJoCo Warp). The port lives in
`rl/mjlab_port/` and registers two tasks:

- **`Mjlab-LegoCar-Signature`** — the full signature-tracing MDP
  (`signature_mdp.py`): per-env recorded paths as a command term, accuracy-gated
  progress reward, off-path/finish terminations. **Action = the two wheel
  efforts directly (no PI loop)** — note this differs from the SB3 env's
  `(v, omega)` action.
- **`Mjlab-LegoCar-Drive`** — a scripted visualization fleet (`play_car.py`).

Install mjlab in its own env (cross-platform uv route, per `rl/README.md`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --package signature-rl && cd signature-rl
uv add mjlab
uv run demo          # verifies the GPU pipeline end to end
```

(Or a `.venv-mjlab` venv with mjlab installed, as the Windows port used.)
Best supported on **Linux + NVIDIA**; Windows + NVIDIA may work but mjlab's
Windows support is "preliminary".

> **⚠️ First task at home:** `rl/mjlab_port/train_car.py` is currently **forced
> to CPU** (`os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` and
> `gpu_ids=None`) because the work machine had no CUDA GPU. On your NVIDIA
> desktop, enabling the GPU (drop the CPU-forcing env var, set `gpu_ids`) is what
> unlocks mjlab's real throughput. Ask Claude to make this change first and
> verify env-steps/s jumps far above the ~85/s CPU-Warp figure.

## 4. mjlab train / watch commands

```bash
# Train (raise --num-envs on the GPU; 64+ is reasonable)
python rl/mjlab_port/train_car.py --num-envs 64 --max-iterations 500

# Watch live in the browser (hot-swaps checkpoints while training runs)
python rl/mjlab_port/play_car.py --task signature --checkpoint latest
#   -> viser viewer at http://localhost:8080
```

Checkpoints land in `logs/rsl_rl/<experiment>/<timestamp>/model_*.pt`.
(mjlab has no training-time visualization; use the viser viewer on checkpoints,
or W&B video logging.)

## 5. Starting prompt to paste to Claude

A fresh session on the desktop won't have this machine's memory, but the repo
carries the context. Paste something like:

> I'm continuing the **lego-signature-car** project on my **NVIDIA GPU desktop**,
> starting the **RL phase — simulation only** (no robot/camera here). I want to
> train in **mjlab on the GPU**. Please read `rl/README.md`,
> `rl/mjlab_port/train_car.py`, `rl/mjlab_port/signature_mdp.py`,
> `rl/mjlab_port/signature_env_cfg.py`, and `rl/TRAINING_LOG.md`, then **propose a
> plan before changing code**. First priority: the mjlab port is currently forced
> to CPU (`CUDA_VISIBLE_DEVICES=""`, `gpu_ids=None`) — enable the NVIDIA GPU and
> verify throughput. Then I want to **add observation noise + latency (plus the
> physical-parameter jitter) to the domain randomization** — the piece
> `rl/README.md` marks as "not yet implemented" — so the policy is robust to the
> real closed-loop sensing gap that hurt BC (see `bc_vs_pure_pursuit.md`).
> Baselines to beat: pure pursuit 1.8 mm rms, BC 1.9 mm (matched at 30 mm/s /
> 6 mm lookahead).

## 6. Goals achievable at home (sim only)

1. **Enable the GPU** in the mjlab port and confirm throughput.
2. **Add domain randomization**: observation noise + control latency, plus the
   XML's physical-parameter jitter (mass, friction, motor gear, wheel damping) —
   the item flagged "not yet implemented" in `rl/README.md`.
3. **Train** a robust signature policy; watch it improve in the viser viewer.
4. **Evaluate in sim** against the pure-pursuit / BC baselines (paired-seed
   protocol, as `learning/evaluate_bc.py --compare-expert` does).
5. **Checkpoint** good models under `logs/` (and export a deployable one).
6. **Deferred to the work site:** closed-loop hardware sim-to-real validation.

## 7. Two RL stacks — pick per goal

| | mjlab port (`rl/mjlab_port/`) | SB3 (`rl/train_rl.py`) |
| --- | --- | --- |
| Backend | rsl-rl PPO on MuJoCo-Warp (GPU) | SB3 PPO, SubprocVecEnv (CPU-parallel) |
| Action | two wheel efforts (no PI loop) | `(v, omega)` (same as expert/BC) |
| Strength | GPU scale + live viser viewer | fast on CPU; **deployable** policy |
| Deploy hook | not directly `--policy`-compatible | plugs into `drive_closed_loop.py --policy` |
| Warm-start | — | `--warm-start models/bc_policy.pt` |

At home the GPU makes **mjlab** the natural experimentation platform. If you also
want a policy you can later run on the robot, do an **SB3** run too (it produces
the `(v, omega)` policy `drive_closed_loop.py --policy models/rl_policy.zip`
expects). Doing both is reasonable.

## 8. Current state (baselines to beat)

| Controller | Env | Speed / lookahead | RMS | Max |
| --- | --- | --- | --- | --- |
| pure pursuit | robot + camera | 30 mm/s / 6 mm | 1.8 mm | 4.2 mm |
| BC policy | robot + camera | ~30 mm/s / 6 mm | 1.9–2.7 mm (run-to-run) | 4.0–6.0 mm |
| BC vs expert | MuJoCo (perfect obs) | 30 mm/s / 6 mm | ~identical to expert | — |

Why RL (given BC already matches pure pursuit): RL optimizes the *actual task
metric* (tip tracking error), not imitation of the expert's actions, so with
disturbances + domain randomization it can exceed the expert and stay accurate
under model/sensor mismatch and at higher speed.

---

*Hardware config (work-site only, irrelevant to sim training): Double Motor card
`2312`/`magenta`, motors invert-both/no-swap, IMU yaw scale 0.1 — all baked into
`drive_closed_loop.py` defaults.*
