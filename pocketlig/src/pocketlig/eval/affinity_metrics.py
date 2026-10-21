"""Affinity metrics used for every model, identically (CASF-2016 'scoring power' style)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Pearson R, Spearman rho, RMSE, MAE and the CASF-style regression SD.

    SD follows the CASF scoring-power definition (verify in the CASF-2016 paper): fit y_true = a + b*y_pred
    by least squares and report sqrt(sum((y_true - a - b*y_pred)^2) / (N - 2)).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length")
    n = len(y)
    if n < 3:
        raise ValueError("need at least 3 samples")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    if np.std(p) == 0 or np.std(y) == 0:
        pearson = spearman = float("nan")
        sd = float("nan")
    else:
        pearson = float(stats.pearsonr(y, p)[0])
        spearman = float(stats.spearmanr(y, p)[0])
        b, a = np.polyfit(p, y, 1)
        sd = float(np.sqrt(np.sum((y - (a + b * p)) ** 2) / (n - 2)))
    return {"n": n, "pearson_r": pearson, "spearman_rho": spearman, "rmse": rmse, "mae": mae, "sd": sd}
