"""
bc_model.py - the PyTorch behavior-cloning policy: a small MLP mapping the
pure-pursuit-style observation (see common.OBS_DIM) to a chassis-frame
(v, omega) action (see common.ACT_DIM). train_bc.py fits it; evaluate_bc.py
(and eventually on-robot deployment) loads it back with save_policy /
load_policy, which bundle in the observation normalization stats so
inference always sees the same input scaling training did.
"""

import torch
import torch.nn as nn

import common


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = common.OBS_DIM, act_dim: int = common.ACT_DIM,
                 hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def save_policy(path: str, model: BCPolicy, normalizer: common.Normalizer, config: dict) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "obs_mean": normalizer.mean,
        "obs_std": normalizer.std,
        "config": config,
    }, path)


def load_policy(path: str, device: str = "cpu"):
    """Returns (model, normalizer, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = BCPolicy(obs_dim=config["obs_dim"], act_dim=config["act_dim"],
                      hidden_size=config["hidden_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = common.Normalizer(ckpt["obs_mean"], ckpt["obs_std"])
    return model, normalizer, config
