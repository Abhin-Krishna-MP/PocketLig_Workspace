"""Synthetic complexes for TESTS ONLY. Never used to produce reported results."""
from __future__ import annotations

import numpy as np

from .data.graph import AtomTable, NODE_DIM, build_graph, to_pyg


def make_synthetic_data(n: int, seed: int = 0, n_prot: int = 40, n_lig: int = 8):
    """Label = (# protein-ligand atom pairs within 4 A)/4, a learnable geometric signal."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        prot = rng.normal(scale=4.0, size=(n_prot, 3)).astype(np.float32)
        lig = (rng.normal(scale=1.5, size=(n_lig, 3)) + rng.normal(scale=2.0, size=(1, 3))).astype(np.float32)
        pos = np.concatenate([prot, lig])
        x = np.zeros((n_prot + n_lig, NODE_DIM), dtype=np.float32)
        x[np.arange(len(x)), rng.integers(0, 4, size=len(x))] = 1.0
        x[n_prot:, 10] = 1.0
        is_lig = np.array([False] * n_prot + [True] * n_lig)
        bonds = [(n_prot + j, n_prot + j + 1) for j in range(n_lig - 1)]
        g = build_graph(AtomTable(x, pos, is_lig, bonds))
        d = np.linalg.norm(prot[:, None] - lig[None], axis=-1)
        y = float((d < 4.0).sum()) / 4.0
        out.append(to_pyg(g, y=y, pdb_id=f"syn{i:03d}"))
    return out
