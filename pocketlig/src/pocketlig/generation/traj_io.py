"""Trajectory file helpers: atomic .npz writes, frame selection, multi-model PDB, frontend JSON, final-mol bonding."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def select_frames(num_steps: int, n_frames: int = 50) -> np.ndarray:
    """Evenly spaced frame indices that always include the first and last step."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    n = min(n_frames, num_steps)
    return np.unique(np.linspace(0, num_steps - 1, n).round().astype(int))


def atomic_save_npz(path: str | Path, **arrays) -> None:
    """Write to a temporary name then rename, so a timeout never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def npz_is_valid(path: str | Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            return "positions" in z.files and "atomic_numbers" in z.files
    except Exception:
        return False


def write_multimodel_pdb(positions: np.ndarray, atomic_numbers: np.ndarray, path: str | Path) -> None:
    """Atoms only (no CONECT): intermediate frames have no real bonds. positions [F,N,3], atomic_numbers [F,N]."""
    lines = []
    for f in range(positions.shape[0]):
        lines.append(f"MODEL     {f + 1:4d}")
        for i, (xyz, z) in enumerate(zip(positions[f], atomic_numbers[f]), start=1):
            sym = SYMBOL.get(int(z), "C")
            lines.append(f"HETATM{i:5d} {sym:<4s} LIG L   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {sym:>2s}")
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def npz_to_frontend_json(npz_path: str | Path, out_path: str | Path) -> dict:
    """Convert a saved trajectory to the compact JSON the web viewer loads."""
    with np.load(npz_path, allow_pickle=False) as z:
        pos = z["positions"].astype(float)
        zs = z["atomic_numbers"]
        steps = z["step_indices"].tolist() if "step_indices" in z.files else list(range(pos.shape[0]))
        final_ok = bool(z["final_ok"]) if "final_ok" in z.files else False
    doc = {
        "elements": [SYMBOL.get(int(a), "C") for a in zs[-1]],
        "frame_elements": [[SYMBOL.get(int(a), "C") for a in f] for f in zs],
        "steps": steps,
        "final_bonded": final_ok,
        "frames": np.round(pos, 3).tolist(),
    }
    Path(out_path).write_text(json.dumps(doc, separators=(",", ":")))
    return {"frames": pos.shape[0], "atoms": pos.shape[1]}


def mol_from_final_frame(positions: np.ndarray, atomic_numbers: np.ndarray):
    """Infer bonds for the final frame with RDKit. Returns (mol or None, error or None)."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    rw = Chem.RWMol()
    for z in atomic_numbers:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(atomic_numbers))
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, tuple(float(v) for v in p))
    rw.AddConformer(conf)
    mol = rw.GetMol()
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        return mol, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
