"""
common.py - shared paths, observation/action layout, dataset loading, and
feature normalization used across the behavior-cloning pipeline
(collect_expert_data.py -> train_bc.py -> evaluate_bc.py).

Importing this module adds the project root to sys.path, so the other
learning/ scripts can `import track_trajectory as tt` even though they live
in a subfolder.
"""

import glob
import os
import sys

import numpy as np

LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LEARNING_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Must match track_trajectory.SignatureTracker._build_observation /
# _expert_action: obs = [dx_local, dy_local, dist_to_final, at_end_flag],
# action = [v (m/s), omega (rad/s)].
OBS_DIM = 4
ACT_DIM = 2


def find_dataset_files(pattern: str = "*expert*.npz"):
    return sorted(glob.glob(os.path.join(DATASET_DIR, pattern)))


def load_datasets(paths):
    """Concatenates observations/actions from one or more
    collect_expert_data.py .npz files into a single (obs, actions) pair."""
    obs_list, act_list = [], []
    for p in paths:
        data = np.load(p)
        obs_list.append(data["observations"])
        act_list.append(data["actions"])
    if not obs_list:
        raise ValueError("No dataset files given.")
    return np.concatenate(obs_list, axis=0), np.concatenate(act_list, axis=0)


class Normalizer:
    """Per-feature standardization (zero mean, unit std), fit on the
    training observations and saved alongside the policy so evaluate_bc.py
    (and eventually on-robot deployment) sees identical input scaling."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
