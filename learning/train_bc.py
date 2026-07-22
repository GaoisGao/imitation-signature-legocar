"""
train_bc.py - Step: train a PyTorch behavior-cloning policy (bc_model.BCPolicy)
to imitate the pure-pursuit expert's (observation -> action) pairs collected
by collect_expert_data.py.

Usage:
    py -3.13 -m pip install torch
    py -3.13 learning/train_bc.py
    py -3.13 learning/train_bc.py --dataset datasets/expert_dataset.npz --epochs 300
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import common
from bc_model import BCPolicy, save_policy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, nargs="+", default=None,
                    help="One or more collect_expert_data.py .npz files. Defaults to every "
                         "datasets/*expert*.npz found.")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=os.path.join(common.MODEL_DIR, "bc_policy.pt"))
    args = ap.parse_args()

    dataset_paths = args.dataset or common.find_dataset_files()
    if not dataset_paths:
        raise SystemExit("No dataset files found. Run learning/collect_expert_data.py first.")
    print(f"Loading {len(dataset_paths)} dataset file(s): "
          f"{[os.path.basename(p) for p in dataset_paths]}")

    observations, actions = common.load_datasets(dataset_paths)
    print(f"Total (observation, action) pairs: {len(observations)}")

    normalizer = common.Normalizer.fit(observations)
    obs_norm = normalizer.transform(observations)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n = len(obs_norm)
    perm = rng.permutation(n)
    n_val = int(n * args.val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train = torch.tensor(obs_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(actions[train_idx], dtype=torch.float32)
    x_val = torch.tensor(obs_norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(actions[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                               batch_size=args.batch_size, shuffle=True)

    model = BCPolicy(obs_dim=common.OBS_DIM, act_dim=common.ACT_DIM, hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    log_every = max(1, args.epochs // 20)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(x_train)

        if len(x_val) > 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
        else:
            val_loss = float("nan")

        if epoch % log_every == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    config = {"obs_dim": common.OBS_DIM, "act_dim": common.ACT_DIM, "hidden_size": args.hidden_size}
    save_policy(args.output, model, normalizer, config)
    print(f"Saved trained BC policy to {args.output}")


if __name__ == "__main__":
    main()
