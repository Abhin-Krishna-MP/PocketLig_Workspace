"""Per-molecule evaluation of generated ligands. Every tool failure is recorded as null + an error string.

Always shows an INDEPENDENT score (Vina/GNINA) next to your own GNN pK; PoseBusters/Vina/GNINA are NOT
exercised in the build sandbox and must be verified on your machine.
Usage: python -m pocketlig.eval.run_eval --sdf_dir artifacts/<t>/trajectories --pocket artifacts/<t>/pocket.pdb \
         --out artifacts/<t>/metrics.json [--receptor_pdbqt r.pdbqt] [--gnn runs/gnn_s0/best.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import QED, rdFingerprintGenerator


def _sa_scorer():
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # type: ignore

        return sascorer.calculateScore
    except Exception:
        return None


def basic_metrics(mol: Chem.Mol) -> dict:
    out = {"valid": True, "qed": None, "sa": None, "n_heavy": mol.GetNumHeavyAtoms(), "errors": {}}
    try:
        out["qed"] = float(QED.qed(mol))
    except Exception as e:  # noqa: BLE001
        out["errors"]["qed"] = str(e)
    sa = _sa_scorer()
    if sa is None:
        out["errors"]["sa"] = "RDKit Contrib SA_Score not found"
    else:
        try:
            out["sa"] = float(sa(mol))
        except Exception as e:  # noqa: BLE001
            out["errors"]["sa"] = str(e)
    return out


def diversity(mols: list[Chem.Mol]) -> float | None:
    """1 - mean pairwise Morgan-2 Tanimoto similarity."""
    if len(mols) < 2:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for m in mols:
        try:
            Chem.FastFindRings(m)
            fps.append(gen.GetFingerprint(m))
        except Exception:
            pass
    if len(fps) < 2:
        return None
    sims = [s for i in range(1, len(fps)) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
    return float(1.0 - np.mean(sims)) if sims else None


def gnina_scores(receptor_pdb: str, sdf: str, ref_sdf_or_pdb: str | None = None) -> dict:
    exe = shutil.which("gnina")
    if exe is None:
        raise FileNotFoundError("gnina not on PATH")
    cmd = [exe, "-r", receptor_pdb, "-l", sdf, "--score_only"]
    cmd += ["--autobox_ligand", ref_sdf_or_pdb or sdf]
    txt = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True).stdout
    get = lambda key: (lambda m: float(m.group(1)) if m else None)(re.search(rf"{key}:\s*(-?[0-9.]+)", txt))  # noqa: E731
    return {"gnina_affinity": get("Affinity"), "gnina_cnn_score": get("CNNscore"), "gnina_cnn_affinity": get("CNNaffinity")}


def posebusters_checks(mol: Chem.Mol, pocket_pdb: str) -> dict:
    from posebusters import PoseBusters

    buster = PoseBusters(config="dock")
    df = buster.bust([mol], None, pocket_pdb)
    row = df.iloc[0].to_dict()
    row = {k: bool(v) for k, v in row.items()}
    return {"posebusters": row, "posebusters_all_pass": all(row.values())}


def gnn_pk(model_ckpt: str, pocket_pdb: str, mol: Chem.Mol, hidden: int = 128, layers: int = 4) -> float:
    import torch
    from torch_geometric.data import Batch

    from ..data.graph import atom_table, build_graph, to_pyg
    from ..data.pocket import load_structure
    from ..models.gnn import AffinityGNN

    st = load_structure(pocket_pdb)
    residues = list(st[0].get_residues())
    data = to_pyg(build_graph(atom_table(residues, mol)))
    model = AffinityGNN(hidden=hidden, layers=layers, dropout=0.0)
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return float(model(Batch.from_data_list([data])).item())


def evaluate_folder(sdf_dir: str, pocket_pdb: str, receptor_pdbqt: str | None = None, gnn_ckpt: str | None = None) -> dict:
    sdfs = sorted(Path(sdf_dir).rglob("*.sdf"))
    rows, mols = [], []
    for p in sdfs:
        row = {"file": str(p), "valid": False, "errors": {}}
        mol = next(iter(Chem.SDMolSupplier(str(p), removeHs=False)), None)
        if mol is None:
            mol = next(iter(Chem.SDMolSupplier(str(p), removeHs=False, sanitize=False)), None)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                except Exception:
                    pass
        if mol is None:
            row["errors"]["load"] = "RDKit could not parse/sanitize"
            rows.append(row)
            continue
        row.update(basic_metrics(mol))
        try:
            row["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
        except Exception:
            row["smiles"] = ""
        try:
            mols.append(Chem.RemoveHs(mol, sanitize=False))
        except Exception:
            mols.append(mol)
        for name, fn in {
            "vina_score_only": lambda: {"vina_score_only": __import__("pocketlig.baselines.vina_score", fromlist=["x"]).vina_score_only(receptor_pdbqt, mol)} if receptor_pdbqt else (_ for _ in ()).throw(RuntimeError("no receptor_pdbqt given")),
            "gnina": lambda: gnina_scores(pocket_pdb, str(p)),
            "posebusters": lambda: posebusters_checks(mol, pocket_pdb),
            "gnn_pk": lambda: {"gnn_pk": gnn_pk(gnn_ckpt, pocket_pdb, mol)} if gnn_ckpt else (_ for _ in ()).throw(RuntimeError("no gnn checkpoint given")),
        }.items():
            try:
                row.update(fn())
            except Exception as e:  # noqa: BLE001  (recorded, never dropped)
                row.setdefault(name, None)
                row["errors"][name] = f"{type(e).__name__}: {e}"
        rows.append(row)
    smiles = [r["smiles"] for r in rows if r.get("smiles")]
    return {
        "n_files": len(sdfs), "n_valid": len(mols),
        "uniqueness": (len(set(smiles)) / len(smiles)) if smiles else None,
        "diversity": diversity(mols), "molecules": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf_dir", required=True)
    ap.add_argument("--pocket", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--receptor_pdbqt")
    ap.add_argument("--gnn")
    a = ap.parse_args()
    res = evaluate_folder(a.sdf_dir, a.pocket, a.receptor_pdbqt, a.gnn)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"files={res['n_files']} valid={res['n_valid']} diversity={res['diversity']}")


if __name__ == "__main__":
    main()
