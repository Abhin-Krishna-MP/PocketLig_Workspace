"""Validate artifacts/<target>/ folders. Exit code 1 if anything required is missing or inconsistent."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ["protein.pdb", "pocket.pdb", "metrics.json", "manifest.json"]
MANIFEST_KEYS = ["target_id", "date", "versions", "generator", "electrostatics"]


def validate_target(d: Path) -> list[str]:
    problems = [f"missing {f}" for f in REQUIRED if not (d / f).exists()]
    if not (d / "surface_potential.dx").exists():
        problems.append("missing surface_potential.dx")
    if (d / "manifest.json").exists():
        m = json.loads((d / "manifest.json").read_text())
        problems += [f"manifest missing key {k}" for k in MANIFEST_KEYS if k not in m]
        if "generator" in m and not m["generator"].get("checkpoint_sha256"):
            problems.append("manifest.generator.checkpoint_sha256 empty")
    mols = sorted((d / "trajectories").glob("mol_*"))
    if not mols:
        problems.append("no trajectories/mol_* folders")
    for mdir in mols:
        fj = mdir / "frames.json"
        if not fj.exists():
            problems.append(f"{mdir.name}: missing frames.json")
            continue
        doc = json.loads(fj.read_text())
        if len(doc["frames"]) != len(doc["steps"]):
            problems.append(f"{mdir.name}: frames/steps length mismatch")
    return problems


def main(root: str = "artifacts") -> int:
    bad = 0
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        problems = validate_target(d)
        print(f"{d.name}: {'OK' if not problems else 'FAIL'}")
        for p in problems:
            print("   -", p)
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "artifacts"))
