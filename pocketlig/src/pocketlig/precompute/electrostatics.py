"""PDB2PQR + APBS wrappers and an OpenDX reader. The solver steps are NOT run in the build sandbox:
verify CLI flags against your installed pdb2pqr/apbs (`pdb2pqr --help`). State force field, pH and ionic
strength in the manifest."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def run_pdb2pqr(pdb: str | Path, pqr: str | Path, apbs_in: str | Path, ff: str = "AMBER", ph: float = 7.0, cwd: str | Path | None = None) -> dict:
    exe = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    if exe is None:
        raise FileNotFoundError("pdb2pqr not found (pip install pdb2pqr)")
    subprocess.run([exe, f"--ff={ff}", "--with-ph", str(ph), "--apbs-input", str(apbs_in), str(pdb), str(pqr)],
                   check=True, timeout=1800, cwd=str(cwd) if cwd else None)
    return {"tool": "pdb2pqr", "force_field": ff, "ph": ph}


def run_apbs(apbs_in: str | Path, workdir: str | Path) -> None:
    exe = shutil.which("apbs")
    if exe is None:
        raise FileNotFoundError("apbs not found (conda install -c conda-forge apbs)")
    subprocess.run([exe, str(apbs_in)], check=True, cwd=str(workdir), timeout=3600)


def read_dx(path: str | Path):
    """Read an OpenDX scalar grid. Returns (origin[3], delta[3], data[nx,ny,nz]) (assumes axis-aligned grid)."""
    origin, deltas, counts, vals = None, [], None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("object 1"):
                counts = [int(t) for t in line.split()[-3:]]
            elif line.startswith("origin"):
                origin = np.array([float(t) for t in line.split()[1:4]])
            elif line.startswith("delta"):
                deltas.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("object") or line.startswith("attribute") or line.startswith("component"):
                continue
            else:
                vals.extend(float(t) for t in line.split())
    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError("not a valid OpenDX grid")
    n = int(np.prod(counts))
    data = np.asarray(vals[:n], dtype=float).reshape(counts)  # x slowest, z fastest (OpenDX order)
    delta = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])
    return origin, delta, data


def sample_dx(path: str | Path, points: np.ndarray) -> np.ndarray:
    """Trilinear-interpolate the potential (kT/e) at Cartesian points [N,3]; NaN outside the grid."""
    origin, delta, data = read_dx(path)
    idx = ((np.asarray(points) - origin) / delta).T
    out = map_coordinates(data, idx, order=1, mode="constant", cval=np.nan)
    return out
