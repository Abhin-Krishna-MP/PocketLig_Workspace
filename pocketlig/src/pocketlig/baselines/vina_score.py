"""AutoDock Vina score-only baseline. NOT executed in the build sandbox: verify on your machine.

Receptor PDBQT preparation uses Meeko's CLI (`mk_prepare_receptor.py`); check its flags for your Meeko
version (`mk_prepare_receptor.py --help`) before running.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem


from Bio.PDB import PDBIO, PDBParser, Select, is_aa


def _find_mk_prepare_receptor() -> str:
    exe = shutil.which("mk_prepare_receptor.py")
    if exe:
        return exe
    venv_exe = Path(sys.executable).parent / "mk_prepare_receptor.py"
    if venv_exe.exists():
        return str(venv_exe)
    return "mk_prepare_receptor.py"


class _KeepStandardAAHeavyAtoms(Select):
    def accept_residue(self, residue):
        return is_aa(residue, standard=True)

    def accept_atom(self, atom):
        return (atom.element or "").upper() not in ("H", "D") and atom.get_altloc() in (" ", "A")


def prepare_receptor(pdb_path: str | Path, out_pdbqt: str | Path) -> None:
    mk_exe = _find_mk_prepare_receptor()
    out_base = Path(out_pdbqt).with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        clean_pdb = Path(td) / "clean.pdb"
        st = PDBParser(QUIET=True).get_structure("prot", str(pdb_path))
        io = PDBIO()
        io.set_structure(st[0])
        io.save(str(clean_pdb), select=_KeepStandardAAHeavyAtoms())
        subprocess.run([mk_exe, "-i", str(clean_pdb), "-o", str(out_base),
                        "-p", "-x"], check=True, timeout=300, capture_output=True)


def _strip_query_flags(mol: Chem.Mol) -> Chem.Mol:
    """Rebuild an RDKit Mol without query atoms/bonds (e.g. from SDF with bond type 4)."""
    rw = Chem.RWMol()
    for a in mol.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsAromatic(a.GetIsAromatic())
        rw.AddAtom(na)
    for b in mol.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
    m = rw.GetMol()
    m.AddConformer(Chem.Conformer(mol.GetConformer()))
    Chem.SanitizeMol(m)
    return m


def ligand_to_pdbqt(mol: Chem.Mol) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    if mol.HasQuery():
        mol = _strip_query_flags(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko failed: {err}")
    return pdbqt


def vina_score_only(receptor_pdbqt: str | Path, mol: Chem.Mol, box_padding: float = 8.0) -> float:
    """Vina score of the ligand exactly as posed (no minimization). Returns kcal/mol (lower = better)."""
    from vina import Vina

    xyz = np.array(mol.GetConformer().GetPositions())
    center = xyz.mean(axis=0)
    size = (xyz.max(axis=0) - xyz.min(axis=0)) + 2 * box_padding
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_string(ligand_to_pdbqt(mol))
    v.compute_vina_maps(center=center.tolist(), box_size=size.tolist())
    return float(v.score()[0])
