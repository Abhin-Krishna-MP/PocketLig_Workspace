"""Evaluation driver for 5-seed GNN models on Test and CASF-2016 benchmarks.

Computes metrics (Pearson R, Spearman rho, RMSE, MAE, SD) for all seeds,
computes 5-seed distribution (mean and std), evaluates the 5-seed ensemble,
and produces results/gnn_summary.csv comparing against baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from pocketlig.eval.affinity_metrics import compute_metrics
from pocketlig.models.gnn import AffinityGNN
from pocketlig.models.train import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_gnn")


def load_dataset(processed_dir: Path, pids: list[str]) -> list:
    data = []
    missing = []
    for pid in pids:
        p = processed_dir / f"{pid}.pt"
        if p.exists():
            data.append(torch.load(p, weights_only=False))
        else:
            missing.append(pid)
    if missing:
        logger.warning(f"Missing {len(missing)} graphs in {processed_dir}: e.g. {missing[:3]}")
    return data


def evaluate_model_on_dataset(model: AffinityGNN, dataset: list, device: str, batch_size: int = 16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, ids = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, ids, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5 GNN seeds on Test and CASF-2016.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Base directory with gnn_s<seed> folders")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"), help="Path to processed graph tensors")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"), help="Path to splits.csv")
    parser.add_argument("--casf-table", type=Path, default=Path("data/casf2016_table.csv"), help="Path to CASF-2016 table")
    parser.add_argument("--out-csv", type=Path, default=Path("results/gnn_summary.csv"), help="Path to output summary CSV")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Test IDs
    splits_df = pd.read_csv(args.splits)
    test_ids = splits_df.loc[splits_df["split"] == "test", "pdb_id"].tolist()
    logger.info(f"Loading {len(test_ids)} Test graphs...")
    test_data = load_dataset(args.processed, test_ids)

    # 2. Load CASF-2016 IDs
    casf_df = pd.read_csv(args.casf_table, header=None, names=["pdb_id", "year", "pK"])
    casf_ids = casf_df["pdb_id"].tolist()
    logger.info(f"Loading {len(casf_ids)} CASF-2016 graphs...")
    casf_data = load_dataset(args.processed, casf_ids)

    seeds = [0, 1, 2, 3, 4]
    test_preds_by_seed = {}
    casf_preds_by_seed = {}

    rows = []

    for s in seeds:
        seed_dir = args.runs_dir / f"gnn_s{s}"
        ckpt_path = seed_dir / "best.pt"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found for seed {s}: {ckpt_path}")
            continue

        model = AffinityGNN(hidden=128, layers=4, dropout=0.1).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=False))
        model.eval()

        # Evaluate on Test
        y_test, p_test, ids_test, m_test = evaluate_model_on_dataset(model, test_data, args.device)
        test_preds_by_seed[s] = (y_test, p_test, ids_test)
        pd.DataFrame({"pdb_id": ids_test, "y_true": y_test, "y_pred": p_test}).to_csv(
            seed_dir / "pred_test.csv", index=False
        )

        # Evaluate on CASF-2016
        y_casf, p_casf, ids_casf, m_casf = evaluate_model_on_dataset(model, casf_data, args.device)
        casf_preds_by_seed[s] = (y_casf, p_casf, ids_casf)
        pd.DataFrame({"pdb_id": ids_casf, "y_true": y_casf, "y_pred": p_casf}).to_csv(
            seed_dir / "pred_casf2016.csv", index=False
        )

        # Update seed's metrics.json
        metrics_json_path = seed_dir / "metrics.json"
        existing_metrics = {}
        if metrics_json_path.exists():
            try:
                existing_metrics = json.loads(metrics_json_path.read_text())
            except Exception:
                pass
        existing_metrics["test"] = m_test
        existing_metrics["casf2016"] = m_casf
        metrics_json_path.write_text(json.dumps(existing_metrics, indent=2))

        logger.info(
            f"Seed {s}: Test R={m_test['pearson_r']:.4f}, RMSE={m_test['rmse']:.4f} | "
            f"CASF-2016 R={m_casf['pearson_r']:.4f}, RMSE={m_casf['rmse']:.4f}"
        )

        rows.append({
            "model": f"gnn_seed_{s}",
            "split": "test",
            **m_test
        })
        rows.append({
            "model": f"gnn_seed_{s}",
            "split": "casf2016",
            **m_casf
        })

    # Compute Statistics across seeds
    df_rows = pd.DataFrame(rows)
    for split_name in ["test", "casf2016"]:
        sub = df_rows[df_rows["split"] == split_name]
        mean_row = {"model": "gnn_5seed_mean", "split": split_name, "n": int(sub["n"].iloc[0])}
        std_row = {"model": "gnn_5seed_std", "split": split_name, "n": int(sub["n"].iloc[0])}
        for metric in ["pearson_r", "spearman_rho", "rmse", "mae", "sd"]:
            mean_row[metric] = float(sub[metric].mean())
            std_row[metric] = float(sub[metric].std())
        rows.append(mean_row)
        rows.append(std_row)

    # Compute 5-Seed Ensemble Predictions
    # 1. Test Ensemble
    ens_test_preds = np.mean([p for _, p, _ in test_preds_by_seed.values()], axis=0)
    y_test_ref = next(iter(test_preds_by_seed.values()))[0]
    ids_test_ref = next(iter(test_preds_by_seed.values()))[2]
    m_test_ens = compute_metrics(y_test_ref, ens_test_preds)
    pd.DataFrame({"pdb_id": ids_test_ref, "y_true": y_test_ref, "y_pred": ens_test_preds}).to_csv(
        Path("results/preds_gnn_ensemble_test.csv"), index=False
    )
    rows.append({"model": "gnn_ensemble_5seed", "split": "test", **m_test_ens})

    # 2. CASF-2016 Ensemble
    ens_casf_preds = np.mean([p for _, p, _ in casf_preds_by_seed.values()], axis=0)
    y_casf_ref = next(iter(casf_preds_by_seed.values()))[0]
    ids_casf_ref = next(iter(casf_preds_by_seed.values()))[2]
    m_casf_ens = compute_metrics(y_casf_ref, ens_casf_preds)
    pd.DataFrame({"pdb_id": ids_casf_ref, "y_true": y_casf_ref, "y_pred": ens_casf_preds}).to_csv(
        Path("results/preds_gnn_ensemble_casf2016.csv"), index=False
    )
    rows.append({"model": "gnn_ensemble_5seed", "split": "casf2016", **m_casf_ens})

    # Add Baselines from results/baselines.csv for Direct Comparison
    baselines_csv = Path("results/baselines.csv")
    if baselines_csv.exists():
        b_df = pd.read_csv(baselines_csv)
        for _, b_row in b_df[b_df["split"].isin(["test", "casf2016"])].iterrows():
            rows.append(b_row.to_dict())

    summary_df = pd.DataFrame(rows)
    cols = ["model", "split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "sd"]
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    summary_df.to_csv(args.out_csv, index=False)
    logger.info(f"Saved complete evaluation summary to {args.out_csv}")

    # Print summary table
    print("\n" + "=" * 80)
    print("AFFINITY BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    pivot = summary_df.pivot(index="model", columns="split", values=["pearson_r", "rmse", "sd"])
    print(pivot.to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
