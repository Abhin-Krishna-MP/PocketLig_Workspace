import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem

from pocketlig.api.main import create_app
from pocketlig.eval.run_eval import evaluate_folder
from pocketlig.generation.traj_io import (atomic_save_npz, mol_from_final_frame, npz_is_valid, npz_to_frontend_json,
                                          select_frames, write_multimodel_pdb)
from pocketlig.precompute.electrostatics import read_dx, sample_dx
from pocketlig.precompute.validate_artifacts import validate_target


def test_select_frames():
    idx = select_frames(1000, 50)
    assert idx[0] == 0 and idx[-1] == 999 and len(idx) == 50
    assert list(select_frames(3, 50)) == [0, 1, 2]


def _ethanol():
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    pos = np.array([list(m.GetConformer().GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    return m, pos, z


def test_traj_roundtrip_and_final_bonding(tmp_path):
    m, pos, z = _ethanol()
    rng = np.random.default_rng(0)
    frames = np.stack([pos + rng.normal(scale=s, size=pos.shape) for s in (3.0, 1.0, 0.0)]).astype(np.float32)
    zs = np.stack([z, z, z])
    p = tmp_path / "mol_0" / "frames.npz"
    atomic_save_npz(p, positions=frames, atomic_numbers=zs, step_indices=np.array([0, 500, 999]), final_ok=np.array(True))
    assert npz_is_valid(p) and not list(p.parent.glob("*.tmp"))
    info = npz_to_frontend_json(p, p.with_name("frames.json"))
    assert info == {"frames": 3, "atoms": 9}
    write_multimodel_pdb(frames, zs, tmp_path / "t.pdb")
    assert (tmp_path / "t.pdb").read_text().count("MODEL") == 3
    mol, err = mol_from_final_frame(frames[-1], zs[-1])
    assert err is None and Chem.MolToSmiles(Chem.RemoveHs(mol)) == "CCO"
    bad, err2 = mol_from_final_frame(frames[0] * 5, zs[0])     # exploded geometry: must NOT pretend to be a molecule
    assert bad is None and err2


def test_dx_interpolation(tmp_path):
    nx = ny = nz = 4
    lines = [f"object 1 class gridpositions counts {nx} {ny} {nz}", "origin 0 0 0", "delta 1 0 0", "delta 0 1 0",
             "delta 0 0 1", f"object 3 class array type double rank 0 items {nx*ny*nz} data follows"]
    vals = [x + 10 * y + 100 * z for x in range(nx) for y in range(ny) for z in range(nz)]  # x slowest, z fastest
    lines += [" ".join(str(float(v)) for v in vals[i:i + 3]) for i in range(0, len(vals), 3)]
    f = tmp_path / "g.dx"
    f.write_text("\n".join(lines) + "\n")
    origin, delta, data = read_dx(f)
    assert data.shape == (4, 4, 4) and data[1, 2, 3] == 1 + 20 + 300
    got = sample_dx(f, np.array([[1.5, 2.0, 0.5], [9.0, 9.0, 9.0]]))
    assert got[0] == pytest.approx(1.5 + 20 + 50) and np.isnan(got[1])


def test_eval_records_failures_not_drops(tmp_path):
    m, _, _ = _ethanol()
    d = tmp_path / "mol_0"
    d.mkdir()
    w = Chem.SDWriter(str(d / "final.sdf"))
    w.write(m)
    w.close()
    (tmp_path / "mol_1").mkdir()
    (tmp_path / "mol_1" / "final.sdf").write_text("garbage")
    res = evaluate_folder(str(tmp_path), "none.pdb")
    assert res["n_files"] == 2 and res["n_valid"] == 1
    bad = [r for r in res["molecules"] if not r["valid"]][0]
    assert "load" in bad["errors"]
    good = [r for r in res["molecules"] if r["valid"]][0]
    assert good["qed"] is not None and "vina_score_only" in good["errors"] and "gnina" in good["errors"]


def _fixture(root):
    t = root / "tgt1"
    (t / "trajectories" / "mol_0").mkdir(parents=True)
    (t / "protein.pdb").write_text("END\n")
    (t / "pocket.pdb").write_text("END\n")
    (t / "surface_potential.dx").write_text("x")
    (t / "metrics.json").write_text(json.dumps({"n_valid": 1, "molecules": [
        {"file": "a", "gnn_pk": 6.5}, {"file": "b", "gnn_pk": None}, {"file": "c", "gnn_pk": 7.5}]}))
    (t / "manifest.json").write_text(json.dumps({"target_id": "tgt1", "date": "2026-10-18", "versions": {},
                                                 "generator": {"checkpoint_sha256": "ab"}, "electrostatics": {}}))
    (t / "trajectories" / "mol_0" / "frames.json").write_text(json.dumps({"frames": [[[0, 0, 0]]], "steps": [0]}))
    (t / "secret.txt").write_text("nope")
    (root / "outside.txt").write_text("outside")
    return t


def test_api_and_validation(tmp_path):
    arts = tmp_path / "artifacts"
    t = _fixture(arts)
    c = TestClient(create_app(arts))
    assert c.get("/targets").json()[0] == {"id": "tgt1", "source": None, "date": "2026-10-18", "n_molecules": 1, "has_potential": True}
    mols = c.get("/targets/tgt1/molecules").json()["molecules"]
    assert [m["gnn_pk"] for m in mols] == [7.5, 6.5, None]
    assert c.get("/targets/tgt1/trajectory/mol_0").json()["steps"] == [0]
    assert c.get("/targets/tgt1/files/protein.pdb").status_code == 200
    # attacks
    assert c.get("/targets/tgt1/files/secret.txt").status_code == 404
    assert c.get("/targets/..%2F/files/protein.pdb").status_code == 404
    assert c.get("/targets/%2e%2e/files/protein.pdb").status_code == 404
    assert c.get("/targets/tgt1/trajectory/..%2F..%2Fsecret").status_code == 404
    assert c.get("/targets/nope").status_code == 404
    assert validate_target(t) == []
    (t / "surface_potential.dx").unlink()
    assert "missing surface_potential.dx" in validate_target(t)
