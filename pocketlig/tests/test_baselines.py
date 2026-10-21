import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen

from pocketlig.baselines.ligand_only import fit_predict

SMILES = ["CCO", "CCCC", "c1ccccc1", "CC(=O)O", "CCN", "CCCCCC", "c1ccccc1O", "CCCCO", "CC(C)C", "c1ccncc1",
          "CCCCCCCC", "CCOC", "CC(=O)N", "c1ccc(C)cc1", "CCCCN", "OCCO", "CCCCCCO", "c1ccc(Cl)cc1", "CCCl", "CCBr"]


def test_ligand_only_learns_simple_signal():
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    y = np.array([Crippen.MolLogP(m) for m in mols])
    pred = fit_predict(mols[:15], y[:15], mols[15:], seed=0)
    assert pred.shape == (5,) and np.corrcoef(pred, y[15:])[0, 1] > 0.5
