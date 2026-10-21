import math

import numpy as np
import pandas as pd
import pytest

from pocketlig.data.parse_index import parse_affinity_string, parse_index, parse_index_line
from pocketlig.data import split as sp
from pocketlig.eval.affinity_metrics import compute_metrics

INDEX = """# header comment
# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
3zzf  2.20  2012  0.40  Ki=400mM      // 3zzf.pdf (NAG)
1abc  1.50  2005  9.00  Kd=1nM        // 1abc.pdf (LIG)
2xyz  NMR   2019  6.30  IC50~0.5uM    // 2xyz.pdf (LIG)
bad line here
4aaa  2.0   2020  8.00  Ki<10nM       // 4aaa.pdf (LIG)
5mis  2.0   2020  5.00  Kd=1nM        // 5mis.pdf (LIG)
"""


def test_affinity_units():
    assert parse_affinity_string("Kd=1nM")["pK"] == pytest.approx(9.0)
    assert parse_affinity_string("Ki=400mM")["pK"] == pytest.approx(-math.log10(0.4))
    assert parse_affinity_string("IC50=2.5uM")["pK"] == pytest.approx(5.60206, abs=1e-4)
    with pytest.raises(ValueError):
        parse_affinity_string("Kd=1xx")


def test_parse_index(tmp_path):
    p = tmp_path / "INDEX"
    p.write_text(INDEX)
    df, failures = parse_index(p)
    assert len(df) == 5 and len(failures) == 1 and "bad line" in failures[0][1]
    row = df.set_index("pdb_id")
    assert math.isnan(row.loc["2xyz", "resolution"])
    assert "inexact_operator" in row.loc["2xyz", "flag"]
    assert "inexact_operator" in row.loc["4aaa", "flag"]
    assert "label_mismatch" in row.loc["5mis", "flag"]   # file says 5.00 but Kd=1nM is 9.00
    assert row.loc["1abc", "flag"] == ""


def test_comment_returns_none():
    assert parse_index_line("# comment") is None and parse_index_line("   ") is None


def _toy():
    df = pd.DataFrame({"pdb_id": ["a1", "a2", "b1", "c1", "d1", "e1"],
                       "year": [2010, 2012, 2013, 2018, 2019, 2020]})
    clusters = {"a1": frozenset({"A"}), "a2": frozenset({"A"}), "b1": frozenset({"B"}),
                "c1": frozenset({"C"}), "d1": frozenset({"A"}), "e1": frozenset({"E"})}  # d1 (test) leaks into A
    return df, clusters


def test_temporal_split_and_leak_detected_then_fixed():
    df, clusters = _toy()
    s = sp.temporal_split(df)
    assert list(s) == ["train", "train", "train", "val", "test", "test"]
    rep = sp.leakage_report(s, df["pdb_id"], clusters)
    assert rep["test_complexes_sharing_cluster_with_train"] == 1 and not rep["clean"]
    fixed = sp.enforce_cluster_separation(s, df["pdb_id"], clusters)
    assert fixed[0].startswith("dropped") and fixed[1].startswith("dropped")  # a1, a2 share cluster A with test d1
    rep2 = sp.leakage_report(fixed, df["pdb_id"], clusters)
    assert rep2["clean"] and rep2["test_complexes_sharing_cluster_with_train"] == 0


def test_benchmark_overlap():
    df, clusters = _toy()
    s = sp.temporal_split(df)
    s2 = sp.remove_benchmark_overlap(s, df["pdb_id"], clusters, {"b1"})
    assert s2[2] == "dropped:benchmark_overlap" and s2[0] == "train"


def test_read_cluster_tsv(tmp_path):
    t = tmp_path / "c.tsv"
    t.write_text("1abc_A\t1abc_A\n1abc_A\t2xyz_B\n3zzf_A\t3zzf_A\n")
    c = sp.read_cluster_tsv(t)
    assert c["1abc"] == c["2xyz"] == frozenset({"1abc_A"})


def test_metrics_known():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0 and m["pearson_r"] == pytest.approx(1.0) and m["sd"] == pytest.approx(0, abs=1e-9)
    m2 = compute_metrics(y, y + 1.0)
    assert m2["rmse"] == pytest.approx(1.0) and m2["pearson_r"] == pytest.approx(1.0)
    assert m2["sd"] == pytest.approx(0, abs=1e-9)  # constant offset removed by regression


def test_make_splits_end_to_end_clean():
    from pocketlig.data.make_splits import make_splits

    rows = []
    for i, (pid, yr) in enumerate([("a1", 2010), ("a2", 2012), ("b1", 2013), ("c1", 2018), ("d1", 2019), ("e1", 2020), ("f1", 2015)]):
        rows.append({"pdb_id": pid, "year": yr, "pK": 5.0 + i, "is_exact": True, "flag": ""})
    rows.append({"pdb_id": "x1", "year": 2011, "pK": 6.0, "is_exact": False, "flag": "inexact_operator"})
    table = pd.DataFrame(rows)
    clusters = {"a1": frozenset({"A"}), "a2": frozenset({"A"}), "b1": frozenset({"B"}), "c1": frozenset({"C"}),
                "d1": frozenset({"A"}), "e1": frozenset({"E"}), "f1": frozenset({"F"}), "x1": frozenset({"X"})}
    out, report = make_splits(table, clusters, benchmark_ids={"f1"})
    d = dict(zip(out["pdb_id"], out["split"]))
    assert "x1" not in d                                   # inexact label removed
    assert d["a1"].startswith("dropped") and d["a2"].startswith("dropped")   # share cluster A with test d1
    assert d["f1"] == "dropped:benchmark_overlap" and d["b1"] == "train"
    assert report["clean"] is True
