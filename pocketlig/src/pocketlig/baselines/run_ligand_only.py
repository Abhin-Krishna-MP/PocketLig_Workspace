"""Driver script for the ligand-only baseline (RDKit descriptors -> GBR).

Evaluates on val, test, and CASF-2016 core set, reporting Pearson R, Spearman rho,
RMSE, MAE, and SD via pocketlig.eval.affinity_metrics.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from pocketlig.baselines.ligand_only import featurize
from pocketlig.data.pocket import load_ligand
from pocketlig.eval.affinity_metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ligand_only")


def get_ligand_paths(pdb_id: str, refined_dir: Path, casf_dir: Path) -> tuple[Path | None, Path | None]:
    for base in [refined_dir / pdb_id, casf_dir / pdb_id]:
        sdf = base / f"{pdb_id}_ligand.sdf"
        mol2 = base / f"{pdb_id}_ligand.mol2"
        if sdf.exists() or mol2.exists():
            return sdf, mol2
    return None, None


def featurize_dataset(df: pd.DataFrame, refined_dir: Path, casf_dir: Path, desc: str = "") -> tuple[list[str], np.ndarray, np.ndarray]:
    ids, feats, ys = [], [], []
    failed = []
    for _, row in df.iterrows():
        pid = row["pdb_id"]
        y = float(row["pK"])
        sdf, mol2 = get_ligand_paths(pid, refined_dir, casf_dir)
        try:
            mol = load_ligand(sdf, mol2)
            f = featurize(mol)
            ids.append(pid)
            feats.append(f)
            ys.append(y)
        except Exception as e:
            failed.append((pid, str(e)))

    if failed:
        logger.warning(f"{desc}: {len(failed)} molecules failed to featurize (e.g. {failed[0][0]}: {failed[0][1]})")
    logger.info(f"{desc}: successfully featurized {len(ids)}/{len(df)} molecules")
    return ids, np.stack(feats) if feats else np.empty((0,)), np.asarray(ys, dtype=float)


def update_baselines_csv(baselines_csv: Path, new_rows: list[dict]) -> None:
    cols = ["model", "split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "sd"]
    if baselines_csv.exists():
        existing_df = pd.read_csv(baselines_csv)
        # Drop rows for the same model and splits if they exist
        new_splits = {(r["model"], r["split"]) for r in new_rows}
        keep_mask = ~existing_df.apply(lambda r: (r["model"], r["split"]) in new_splits, axis=1)
        combined = pd.concat([existing_df[keep_mask], pd.DataFrame(new_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(new_rows)
    # Ensure column ordering
    ordered_cols = [c for c in cols if c in combined.columns] + [c for c in combined.columns if c not in cols]
    combined[ordered_cols].to_csv(baselines_csv, index=False)
    logger.info(f"Updated summary table at {baselines_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ligand-only baseline.")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"), help="Path to splits.csv")
    parser.add_argument("--refined-dir", type=Path, default=Path("data/pdbbind/refined-set"), help="Path to refined set")
    parser.add_argument("--casf-table", type=Path, default=Path("data/casf2016_table.csv"), help="Path to CASF-2016 table")
    parser.add_argument("--casf-dir", type=Path, default=Path("data/CASF-2016/coreset"), help="Path to CASF-2016 coreset")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Directory to save predictions and metrics")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for GBR")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits_df = pd.read_csv(args.splits)
    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]
    test_df = splits_df[splits_df["split"] == "test"]

    casf_df = pd.read_csv(args.casf_table, header=None, names=["pdb_id", "year", "pK"])

    logger.info(f"Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, casf2016={len(casf_df)}")

    # Featurize train
    train_ids, X_train, y_train = featurize_dataset(train_df, args.refined_dir, args.casf_dir, desc="train")

    logger.info(f"Training GradientBoostingRegressor on {len(X_train)} samples with seed {args.seed}...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    eval_sets = [
        ("val", val_df),
        ("test", test_df),
        ("casf2016", casf_df),
    ]

    summary_rows = []
    all_preds = []

    for split_name, df in eval_sets:
        ids, X_eval, y_eval = featurize_dataset(df, args.refined_dir, args.casf_dir, desc=split_name)
        preds = model.predict(X_eval)
        metrics = compute_metrics(y_eval, preds)

        logger.info(
            f"ligand_only [{split_name}]: n={metrics['n']}, Pearson R={metrics['pearson_r']:.4f}, "
            f"Spearman rho={metrics['spearman_rho']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, SD={metrics['sd']:.4f}"
        )

        row = {"model": "ligand_only", "split": split_name, **metrics}
        summary_rows.append(row)

        pred_df = pd.DataFrame({
            "pdb_id": ids,
            "split": split_name,
            "y_true": y_eval,
            "y_pred": preds,
        })
        pred_df.to_csv(args.out_dir / f"preds_ligand_only_{split_name}.csv", index=False)
        all_preds.append(pred_df)

    pd.concat(all_preds, ignore_index=True).to_csv(args.out_dir / "preds_ligand_only.csv", index=False)
    update_baselines_csv(args.out_dir / "baselines.csv", summary_rows)


if __name__ == "__main__":
    main()
