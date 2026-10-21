import numpy as np
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem

from pocketlig.api.main import create_app
from pocketlig.properties import detect_contacts, molecule_properties


def _write_ethanol(path, shift=(0.0, 0.0, 0.0)):
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=2)
    c = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = c.GetAtomPosition(i)
        c.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    w = Chem.SDWriter(str(path)); w.write(m); w.close()
    heavy = [(a.GetSymbol(), list(c.GetAtomPosition(a.GetIdx()))) for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    return heavy


def test_properties_ethanol(tmp_path):
    _write_ethanol(tmp_path / "l.sdf")
    p = molecule_properties(tmp_path / "l.sdf")
    assert abs(p["molecular_weight"] - 46.07) < 0.05 and p["hbd"] == 1 and p["hba"] == 1 and p["lipinski_violations"] == []
    assert abs(p["tpsa"] - 20.23) < 0.1


def test_contacts_detects_hbond_and_ignores_far(tmp_path):
    heavy = _write_ethanol(tmp_path / "l.sdf")
    ox = [xyz for sym, xyz in heavy if sym == "O"][0]
    near = (ox[0] + 2.8, ox[1], ox[2])
    far = (ox[0] + 30, ox[1], ox[2])
    lines = [f"ATOM      1  OD1 ASP A  10    {near[0]:8.3f}{near[1]:8.3f}{near[2]:8.3f}  1.00  0.00           O",
             f"ATOM      2  OD2 ASP A  11    {far[0]:8.3f}{far[1]:8.3f}{far[2]:8.3f}  1.00  0.00           O", "END"]
    (tmp_path / "pocket.pdb").write_text("\n".join(lines) + "\n")
    res = detect_contacts(tmp_path / "pocket.pdb", tmp_path / "l.sdf")
    assert [(r["type"], r["residue"]) for r in res] == [("Hydrogen bond", "ASP10")]
    assert abs(res[0]["distance"] - 2.8) < 0.01


def test_new_endpoints(tmp_path):
    t = tmp_path / "art" / "t1"
    (t / "trajectories" / "mol_0").mkdir(parents=True)
    _write_ethanol(t / "trajectories" / "mol_0" / "final.sdf")
    (t / "pocket.pdb").write_text("ATOM      1  N   ALA A   1     100.000 100.000 100.000  1.00  0.00           N\nEND\n")
    c = TestClient(create_app(tmp_path / "art"))
    assert c.get("/targets/t1/molecule/mol_0/properties").json()["hbd"] == 1
    assert c.get("/targets/t1/molecule/mol_0/contacts").json()["contacts"] == []
    assert c.get("/targets/t1/pocket").json()["residues"] == 1
    assert c.get("/targets/t1/molecule/mol_9/properties").status_code == 404
