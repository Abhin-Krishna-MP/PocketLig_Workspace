"""Train/evaluate the affinity GNN. Resumable: saves last.pt every epoch and continues from it.

Usage:
  python -m pocketlig.models.train --processed data/processed --splits data/splits.csv --out runs/gnn_s0 --seed 0
  python -m pocketlig.models.train --processed data/processed --splits data/splits.csv --out runs/overfit --overfit_n 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ..eval.affinity_metrics import compute_metrics
from .gnn import AffinityGNN


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        ps.append(model(batch).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
        ids += list(batch.pdb_id)
    return np.concatenate(ys), np.concatenate(ps), ids


def fit(train_set, val_set, out_dir: Path, epochs=100, lr=1e-3, hidden=128, layers=4, batch_size=32,
        seed=0, device="cpu", patience=20, weight_decay=1e-5, dropout=0.1, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = AffinityGNN(hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    state = {"epoch": 0, "best_val_rmse": float("inf"), "bad": 0, "history": []}
    last = out_dir / "last.pt"
    if last.exists():  # resume after a timeout
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        log(f"resumed from epoch {state['epoch']}")
    loss_fn = torch.nn.MSELoss()
    while state["epoch"] < epochs and state["bad"] < patience:
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        yv, pv, _ = predict(model, val_loader, device)
        val_rmse = float(np.sqrt(np.mean((yv - pv) ** 2)))
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_mse": total / n, "val_rmse": val_rmse})
        if val_rmse < state["best_val_rmse"]:
            state["best_val_rmse"], state["bad"] = val_rmse, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, last)
        log(f"epoch {state['epoch']} train_mse {total / n:.4f} val_rmse {val_rmse:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out_dir / "history.json").write_text(json.dumps(state["history"], indent=1))
    return state


def load_split(processed: Path, splits_csv: str):
    df = pd.read_csv(splits_csv)
    sets = {}
    for name in ("train", "val", "test"):
        ids = df.loc[df["split"] == name, "pdb_id"].tolist()
        sets[name] = [torch.load(processed / f"{i}.pt", weights_only=False) for i in ids if (processed / f"{i}.pt").exists()]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--overfit_n", type=int, default=0, help="train+val on the first N train graphs (bug check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sets = load_split(Path(args.processed), args.splits)
    out = Path(args.out)
    if args.overfit_n:
        small = sets["train"][: args.overfit_n]
        state = fit(small, small, out, epochs=args.epochs, lr=args.lr, hidden=args.hidden, layers=args.layers,
                    batch_size=args.batch_size, seed=args.seed, device=args.device, patience=10**9, dropout=0.0)
        print("final train rmse (should be < 0.1):", state["history"][-1]["val_rmse"])
        return
    state = fit(sets["train"], sets["val"], out, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                layers=args.layers, batch_size=args.batch_size, seed=args.seed, device=args.device, dropout=args.dropout)
    model = AffinityGNN(hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(args.device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    results = {"seed": args.seed, "best_val_rmse": state["best_val_rmse"]}
    for name in ("val", "test"):
        y, p, ids = predict(model, DataLoader(sets[name], batch_size=args.batch_size), args.device)
        results[name] = compute_metrics(y, p)
        pd.DataFrame({"pdb_id": ids, "y": y, "pred": p}).to_csv(out / f"pred_{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
