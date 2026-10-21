"""Ligand-only baseline: RDKit descriptors -> gradient boosting. If your GNN does not beat this,
it is not learning protein-ligand interactions."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingRegressor


def featurize(mol: Chem.Mol) -> np.ndarray:
    vals = []
    for _, fn in Descriptors.descList:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(0.0)
    arr = np.nan_to_num(np.asarray(vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    f32_max = np.finfo(np.float32).max * 0.99
    return np.clip(arr, -f32_max, f32_max).astype(np.float32)


def fit_predict(train_mols, y_train, test_mols, seed: int = 0) -> np.ndarray:
    X = np.stack([featurize(m) for m in train_mols])
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=seed)
    model.fit(X, np.asarray(y_train))
    return model.predict(np.stack([featurize(m) for m in test_mols]))
