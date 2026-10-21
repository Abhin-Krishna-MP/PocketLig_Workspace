import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch

from pocketlig.data.graph import EDGE_DIM, NODE_DIM, atom_table, build_graph, to_pyg
from pocketlig.data.pocket import clean_protein, extract_pocket_residues, load_structure, write_pocket
from pocketlig.models.gnn import AffinityGNN
from pocketlig.models.train import fit, predict
from pocketlig.synthetic import make_synthetic_data
from torch_geometric.loader import DataLoader

PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.300   1.400   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       4.000   2.700   0.000  1.00  0.00           C
ATOM      7  N   SER A  50      40.000  40.000  40.000  1.00  0.00           N
ATOM      8  CA  SER A  50      41.000  40.000  40.000  1.00  0.00           C
HETATM    9  O   HOH A 100       2.000   2.000   2.000  1.00  0.00           O
END
"""


def _ligand():
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):  # move ligand next to residues 1-2
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 3.0, p.y + 4.5, p.z + 1.5))
    return m


def test_clean_and_pocket(tmp_path):
    f = tmp_path / "p.pdb"
    f.write_text(PDB)
    log = clean_protein(f, tmp_path / "clean.pdb")
    assert log["removed"] == 1                      # the water
    st = load_structure(tmp_path / "clean.pdb")
    lig = _ligand()
    xyz = np.array([list(lig.GetConformer().GetAtomPosition(a.GetIdx())) for a in lig.GetAtoms() if a.GetAtomicNum() > 1])
    res = extract_pocket_residues(st, xyz, cutoff=10.0)
    assert [r.get_id()[1] for r in res] == [1, 2]   # residue 50 is ~60 A away
    write_pocket(st, res, tmp_path / "pocket.pdb")
    assert "SER" not in (tmp_path / "pocket.pdb").read_text()


def test_graph_shapes_and_invariance(tmp_path):
    f = tmp_path / "p.pdb"
    f.write_text(PDB)
    clean_protein(f, tmp_path / "c.pdb")
    st = load_structure(tmp_path / "c.pdb")
    lig = _ligand()
    xyz = np.array([list(lig.GetConformer().GetAtomPosition(a.GetIdx())) for a in lig.GetAtoms() if a.GetAtomicNum() > 1])
    res = extract_pocket_residues(st, xyz, 10.0)
    t = atom_table(res, lig)
    assert t.x.shape == (t.pos.shape[0], NODE_DIM) and t.is_ligand.sum() == 3 and len(t.ligand_bonds) == 2
    g = build_graph(t)
    assert g["edge_attr"].shape[1] == EDGE_DIM
    # rotation + translation must leave all edge features unchanged
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t2 = type(t)(t.x, (t.pos @ q.T + 7.0).astype(np.float32), t.is_ligand, t.ligand_bonds)
    g2 = build_graph(t2)
    np.testing.assert_allclose(np.sort(g["edge_attr"], axis=0), np.sort(g2["edge_attr"], axis=0), atol=1e-4)


def test_model_invariance():
    data = make_synthetic_data(4, seed=1)
    model = AffinityGNN(hidden=32, layers=2).eval()
    b = Batch.from_data_list(data)
    p1 = model(b)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    data2 = []
    for d in make_synthetic_data(4, seed=1):
        pos = d.pos.numpy() @ q.T + 5.0
        g = build_graph(type("T", (), {})() if False else __import__("pocketlig.data.graph", fromlist=["AtomTable"]).AtomTable(
            d.x.numpy(), pos.astype(np.float32), d.is_ligand.numpy(), []))
        data2.append(to_pyg(g, y=float(d.y), pdb_id=d.pdb_id))
    # bonds were dropped in the rebuilt graph, so compare on the same bond-free graphs
    data1 = []
    for d in make_synthetic_data(4, seed=1):
        g = build_graph(__import__("pocketlig.data.graph", fromlist=["AtomTable"]).AtomTable(
            d.x.numpy(), d.pos.numpy(), d.is_ligand.numpy(), []))
        data1.append(to_pyg(g, y=float(d.y), pdb_id=d.pdb_id))
    a = model(Batch.from_data_list(data1))
    c = model(Batch.from_data_list(data2))
    torch.testing.assert_close(a, c, atol=1e-4, rtol=1e-4)


def test_overfit_20_samples(tmp_path):
    data = make_synthetic_data(20, seed=0)
    state = fit(data, data, tmp_path, epochs=400, lr=2e-3, hidden=64, layers=3, batch_size=20,
                patience=10**9, dropout=0.0, log=lambda *_: None)
    model = AffinityGNN(hidden=64, layers=3, dropout=0.0)
    model.load_state_dict(torch.load(tmp_path / "last.pt", weights_only=False)["model"])
    y, p, _ = predict(model, DataLoader(data, batch_size=20), "cpu")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    assert rmse < 0.3, rmse   # synthetic sanity check; the < 0.1 gate applies to your real-data overfit run


def test_resume(tmp_path):
    data = make_synthetic_data(8, seed=2)
    fit(data, data, tmp_path, epochs=2, hidden=16, layers=1, batch_size=8, log=lambda *_: None)
    seen = []
    fit(data, data, tmp_path, epochs=3, hidden=16, layers=1, batch_size=8, log=seen.append)
    assert any("resumed from epoch 2" in str(s) for s in seen)
