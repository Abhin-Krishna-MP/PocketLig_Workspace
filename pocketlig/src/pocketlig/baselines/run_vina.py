"""Driver script for the AutoDock Vina score-only baseline.

Evaluates on val, test, and CASF-2016 core set, reporting Pearson R, Spearman rho,
RMSE, MAE, and SD via pocketlig.eval.affinity_metrics.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pocketlig.baselines.vina_score import prepare_receptor, vina_score_only
from pocketlig.data.pocket import load_ligand
from pocketlig.eval.affinity_metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vina_baseline")


def get_complex_paths(pdb_id: str, refined_dir: Path, casf_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    for base in [refined_dir / pdb_id, casf_dir / pdb_id]:
        rec = base / f"{pdb_id}_protein.pdb"
        sdf = base / f"{pdb_id}_ligand.sdf"
        mol2 = base / f"{pdb_id}_ligand.mol2"
        if rec.exists() and (sdf.exists() or mol2.exists()):
            return rec, sdf, mol2
    return None, None, None


def score_single_complex(item: tuple[str, str, float, Path, Path, Path, Path]) -> dict:
    pdb_id, split, pK, rec_pdb, sdf, mol2, cache_dir = item
    rec_pdbqt = cache_dir / f"{pdb_id}.pdbqt"

    # Step 1: Receptor preparation (cached)
    try:
        if not rec_pdbqt.exists() or rec_pdbqt.stat().st_size == 0:
            prepare_receptor(rec_pdb, rec_pdbqt)
    except Exception as e:
        return {"pdb_id": pdb_id, "split": split, "pK": pK, "vina_score": np.nan, "error": f"receptor prep: {e}"}

    # Step 2: Ligand load and score
    try:
        mol = load_ligand(sdf, mol2)
        score = vina_score_only(rec_pdbqt, mol)
        return {"pdb_id": pdb_id, "split": split, "pK": pK, "vina_score": score, "error": None}
    except Exception as e:
        return {"pdb_id": pdb_id, "split": split, "pK": pK, "vina_score": np.nan, "error": f"scoring: {e}"}


def update_baselines_csv(baselines_csv: Path, new_rows: list[dict]) -> None:
    cols = ["model", "split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "sd"]
    if baselines_csv.exists():
        existing_df = pd.read_csv(baselines_csv)
        new_splits = {(r["model"], r["split"]) for r in new_rows}
        keep_mask = ~existing_df.apply(lambda r: (r["model"], r["split"]) in new_splits, axis=1)
        combined = pd.concat([existing_df[keep_mask], pd.DataFrame(new_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(new_rows)
    ordered_cols = [c for c in cols if c in combined.columns] + [c for c in combined.columns if c not in cols]
    combined[ordered_cols].to_csv(baselines_csv, index=False)
    logger.info(f"Updated summary table at {baselines_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AutoDock Vina score-only baseline.")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"), help="Path to splits.csv")
    parser.add_argument("--refined-dir", type=Path, default=Path("data/pdbbind/refined-set"), help="Path to refined set")
    parser.add_argument("--casf-table", type=Path, default=Path("data/casf2016_table.csv"), help="Path to CASF-2016 table")
    parser.add_argument("--casf-dir", type=Path, default=Path("data/CASF-2016/coreset"), help="Path to CASF-2016 coreset")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/pdbqt_cache"), help="Directory for prepared receptor PDBQTs")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Directory to save predictions and metrics")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Parallel worker count")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    splits_df = pd.read_csv(args.splits)
    val_df = splits_df[splits_df["split"] == "val"].copy()
    test_df = splits_df[splits_df["split"] == "test"].copy()

    casf_df = pd.read_csv(args.casf_table, header=None, names=["pdb_id", "year", "pK"]).copy()
    casf_df["split"] = "casf2016"

    eval_sets = [
        ("val", val_df),
        ("test", test_df),
        ("casf2016", casf_df),
    ]

    all_tasks = []
    for split_name, df in eval_sets:
        for _, row in df.iterrows():
            pid = row["pdb_id"]
            pK = float(row["pK"])
            rec_pdb, sdf, mol2 = get_complex_paths(pid, args.refined_dir, args.casf_dir)
            all_tasks.append((pid, split_name, pK, rec_pdb, sdf, mol2, args.cache_dir))

    logger.info(f"Scoring {len(all_tasks)} complexes across val, test, and CASF-2016 using {args.workers} workers...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(score_single_complex, all_tasks):
            results.append(res)
            if len(results) % 50 == 0 or len(results) == len(all_tasks):
                logger.info(f"Progress: {len(results)}/{len(all_tasks)} complexes processed")

    res_df = pd.DataFrame(results)
    res_df["pred_pK"] = -res_df["vina_score"]

    summary_rows = []
    for split_name, _ in eval_sets:
        sub = res_df[res_df["split"] == split_name]
        valid = sub.dropna(subset=["vina_score"])
        failed = sub[sub["vina_score"].isna()]
        if len(failed) > 0:
            logger.warning(f"vina_score [{split_name}]: {len(failed)} failed out of {len(sub)}")
            for _, f_row in failed.head(3).iterrows():
                logger.warning(f"  sample failure: {f_row['pdb_id']} - {f_row['error']}")

        y_true = valid["pK"].values
        y_pred = valid["pred_pK"].values  # -vina_score for CASF scoring power parity

        metrics = compute_metrics(y_true, y_pred)
        logger.info(
            f"vina_score [{split_name}]: n={metrics['n']}/{len(sub)}, "
            f"Pearson R={metrics['pearson_r']:.4f}, Spearman rho={metrics['spearman_rho']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, SD={metrics['sd']:.4f}"
        )

        row = {"model": "vina_score", "split": split_name, **metrics}
        summary_rows.append(row)

        sub.to_csv(args.out_dir / f"preds_vina_{split_name}.csv", index=False)

    res_df.to_csv(args.out_dir / "preds_vina.csv", index=False)
    update_baselines_csv(args.out_dir / "baselines.csv", summary_rows)


if __name__ == "__main__":
    main()
