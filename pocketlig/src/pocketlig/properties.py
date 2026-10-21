"""Real molecular properties and distance-based contact detection for generated ligands (RDKit + Biopython)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.spatial import cKDTree

HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS"}
BACKBONE = {"N", "CA", "C", "O"}


def _load_sdf(sdf_path: str | Path, removeHs: bool = False) -> Chem.Mol | None:
    p = str(sdf_path)
    mol = next(iter(Chem.SDMolSupplier(p, removeHs=removeHs)), None)
    if mol is None:
        mol = next(iter(Chem.SDMolSupplier(p, removeHs=False, sanitize=False)), None)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            except Exception:
                pass
    return mol


def molecule_properties(sdf_path: str | Path) -> dict:
    mol = _load_sdf(sdf_path, removeHs=False)
    if mol is None:
        return {
            "molecular_weight": 0.0, "logp": 0.0, "hbd": 0, "hba": 0,
            "rotatable_bonds": 0, "tpsa": 0.0, "lipinski_violations": []
        }
    try:
        m = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        m = mol
    try:
        mw = float(Descriptors.MolWt(m))
    except Exception:
        mw = float(sum(a.GetMass() for a in m.GetAtoms()))
    try:
        logp = float(Crippen.MolLogP(m))
    except Exception:
        logp = 0.0
    try:
        hbd = int(Lipinski.NumHDonors(m))
        hba = int(Lipinski.NumHAcceptors(m))
    except Exception:
        hbd, hba = 0, 0
    try:
        rot = int(rdMolDescriptors.CalcNumRotatableBonds(m))
    except Exception:
        rot = 0
    try:
        tpsa = float(rdMolDescriptors.CalcTPSA(m))
    except Exception:
        tpsa = 0.0
    violations = [name for name, bad in (("MW > 500", mw > 500), ("LogP > 5", logp > 5),
                                         ("HBD > 5", hbd > 5), ("HBA > 10", hba > 10)) if bad]
    return {"molecular_weight": mw, "logp": logp, "hbd": hbd, "hba": hba,
            "rotatable_bonds": rot, "tpsa": tpsa,
            "lipinski_violations": violations}


def detect_contacts(pocket_pdb: str | Path, sdf_path: str | Path, hbond_cutoff: float = 3.5,
                    hydrophobic_cutoff: float = 4.0, limit: int = 12) -> list[dict]:
    """DISTANCE-ONLY contact detection (no hydrogens, no angles): H-bond candidates (ligand N/O to protein N/O
    within cutoff) and hydrophobic contacts (ligand carbons with no heteroatom neighbours to side-chain carbons
    of hydrophobic residues). These are candidates, not proven interactions."""
    from Bio.PDB import PDBParser, is_aa

    mol = _load_sdf(sdf_path, removeHs=True)
    if mol is None:
        mol = _load_sdf(sdf_path, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return []
    conf = mol.GetConformer()
    lig = [(a.GetIdx(), a.GetSymbol(), np.array(conf.GetAtomPosition(a.GetIdx())),
            all(n.GetSymbol() in ("C",) for n in a.GetNeighbors())) for a in mol.GetAtoms()]
    st = PDBParser(QUIET=True).get_structure("p", str(pocket_pdb))
    patoms = [(r.get_resname(), r.get_id()[1], a.get_id(), (a.element or "").upper(), a.coord)
              for r in st[0].get_residues() if is_aa(r, standard=True) for a in r.get_atoms()]
    if not patoms:
        return []
    tree = cKDTree(np.array([p[4] for p in patoms]))
    best: dict[tuple, dict] = {}
    for idx, sym, xyz, apolar in lig:
        cutoff = max(hbond_cutoff, hydrophobic_cutoff)
        for j in tree.query_ball_point(xyz, r=cutoff):
            resn, resi, an, el, pxyz = patoms[j]
            d = float(np.linalg.norm(xyz - pxyz))
            kind = None
            if sym in ("N", "O") and el in ("N", "O") and d <= hbond_cutoff:
                kind = "Hydrogen bond"
            elif sym == "C" and apolar and el == "C" and resn in HYDROPHOBIC_RES and an not in BACKBONE and d <= hydrophobic_cutoff:
                kind = "Hydrophobic"
            if kind:
                key = (kind, idx, resn, resi)
                if key not in best or d < best[key]["distance"]:
                    best[key] = {"type": kind, "ligand_atom": f"{sym}{idx + 1}", "residue": f"{resn}{resi}",
                                 "protein_atom": an, "distance": round(d, 2)}
    out = sorted(best.values(), key=lambda r: (r["type"] != "Hydrogen bond", r["distance"]))
    return out[:limit]
