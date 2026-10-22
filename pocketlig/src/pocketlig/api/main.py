"""API over artifacts/ and asynchronous GPU diffusion generation."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MOL_RE = re.compile(r"^mol_\d{1,6}$")
PUBLIC_FILES = {"protein.pdb", "pocket.pdb", "surface_potential.dx"}

POCKETLIG_ROOT = Path("/home/abhin-krishna-m-p/pocketlig").resolve()
TARGETDIFF_ROOT = Path("/home/abhin-krishna-m-p/pocketlig_workspace/targetdiff").resolve()
POCKETFORGE_PY = Path("/home/abhin-krishna-m-p/anaconda3/envs/pocketforge/bin/python")


class TargetSummary(BaseModel):
    id: str
    source: str | None = None
    date: str | None = None
    n_molecules: int
    has_potential: bool


class GenerateRequest(BaseModel):
    num_molecules: int = 1
    diffusion_steps: int = 100
    temperature: float = 1.0
    seed: int | None = None
    device: str = "cuda:0"


class JobStatus(BaseModel):
    id: str
    target_id: str
    status: str  # pending, running, completed, failed
    progress: int
    current_step: int | None = None
    total_steps: int | None = None
    molecules_generated: list[str] = []
    error: str | None = None
    created_at: str


JOBS: dict[str, dict[str, Any]] = {}


def create_app(artifacts_dir: str | Path, origins: list[str] | None = None) -> FastAPI:
    root = Path(artifacts_dir).resolve()
    app = FastAPI(title="pocketlig artifacts API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:8008", "http://127.0.0.1:8008",
            "*"
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def target_dir(tid: str) -> Path:
        known = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
        if not ID_RE.match(tid) or tid not in known:
            raise HTTPException(404, "unknown target")
        return root / tid

    def read_json(path: Path):
        if not path.exists():
            raise HTTPException(404, f"{path.name} not found")
        return json.loads(path.read_text())

    @app.get("/targets", response_model=list[TargetSummary])
    def list_targets():
        out = []
        for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
            out.append(TargetSummary(id=d.name, source=m.get("source"), date=m.get("date"),
                                     n_molecules=len(list((d / "trajectories").glob("mol_*"))),
                                     has_potential=(d / "surface_potential.dx").exists()))
        return out

    @app.get("/targets/{tid}")
    def manifest(tid: str):
        return read_json(target_dir(tid) / "manifest.json")

    @app.get("/targets/{tid}/molecules")
    def molecules(tid: str, sort_by: str = "gnn_pk", descending: bool = True):
        p = target_dir(tid) / "metrics.json"
        if not p.exists():
            return {"summary": {}, "molecules": []}
        res = read_json(p)
        rows = res.get("molecules", [])
        key = lambda r: (r.get(sort_by) is None, -(r.get(sort_by) or 0) if descending else (r.get(sort_by) or 0))  # noqa: E731
        return {"summary": {k: v for k, v in res.items() if k != "molecules"}, "molecules": sorted(rows, key=key)}

    @app.get("/targets/{tid}/trajectory/{mol}")
    def trajectory(tid: str, mol: str):
        if not MOL_RE.match(mol):
            raise HTTPException(404, "bad molecule id")
        return read_json(target_dir(tid) / "trajectories" / mol / "frames.json")

    @app.get("/targets/{tid}/molecule/{mol}/final.sdf")
    def final_sdf(tid: str, mol: str):
        if not MOL_RE.match(mol):
            raise HTTPException(404, "bad molecule id")
        p = target_dir(tid) / "trajectories" / mol / "final.sdf"
        if not p.exists():
            raise HTTPException(404, "no sanitized final molecule for this sample")
        return FileResponse(p, media_type="chemical/x-mdl-sdfile")

    def _sdf(tid: str, mol: str) -> Path:
        if not MOL_RE.match(mol):
            raise HTTPException(404, "bad molecule id")
        p = target_dir(tid) / "trajectories" / mol / "final.sdf"
        if not p.exists():
            raise HTTPException(404, "no sanitized final molecule for this sample")
        return p

    @app.get("/targets/{tid}/molecule/{mol}/properties")
    def properties(tid: str, mol: str):
        from ..properties import molecule_properties
        return molecule_properties(_sdf(tid, mol))

    @app.get("/targets/{tid}/molecule/{mol}/contacts")
    def contacts(tid: str, mol: str):
        from ..properties import detect_contacts
        return {"method": "distance-only candidates (no hydrogens or angles)",
                "contacts": detect_contacts(target_dir(tid) / "pocket.pdb", _sdf(tid, mol))}

    @app.get("/targets/{tid}/pocket")
    def pocket_info(tid: str):
        import numpy as np
        d = target_dir(tid)
        xyz, res = [], set()
        for line in (d / "pocket.pdb").read_text().splitlines():
            if line.startswith("ATOM"):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                res.add((line[21], line[22:27]))
        m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        prep = m.get("steps", {}).get("prepare", {})
        return {"residues": len(res), "atoms": len(xyz), "centroid": np.mean(xyz, axis=0).round(1).tolist() if xyz else None,
                "reference_ligand": prep.get("reference_ligand"), "cutoff_A": prep.get("pocket_cutoff_A")}

    @app.get("/targets/{tid}/files/{name}")
    def files(tid: str, name: str):
        if name not in PUBLIC_FILES:
            raise HTTPException(404, "file not available")
        p = target_dir(tid) / name
        if not p.exists():
            raise HTTPException(404, "file not found")
        return FileResponse(p)

    # ─── REAL BACKGROUND DIFFUSION GENERATION ───
    def _run_diffusion_worker(job_id: str, tid: str, req: GenerateRequest):
        job = JOBS[job_id]
        tdir = target_dir(tid)
        traj_dir = tdir / "trajectories"
        pocket_pdb = tdir / "pocket.pdb"
        traj_dir.mkdir(parents=True, exist_ok=True)

        existing_indices = []
        for p in traj_dir.glob("mol_*"):
            m = re.match(r"^mol_(\d+)$", p.name)
            if m:
                existing_indices.append(int(m.group(1)))
        start_idx = max(existing_indices) + 1 if existing_indices else 0

        job["status"] = "running"
        job["progress"] = 2
        job["current_step"] = 0
        job["total_steps"] = req.diffusion_steps

        cmd = [
            str(POCKETFORGE_PY),
            str(POCKETLIG_ROOT / "scripts" / "targetdiff_sample_pocket.py"),
            str(TARGETDIFF_ROOT / "configs" / "sampling.yml"),
            "--pdb_path", str(pocket_pdb),
            "--out_dir", str(traj_dir),
            "--num_samples", str(req.num_molecules),
            "--chunk", str(min(10, req.num_molecules)),
            "--num_steps", str(req.diffusion_steps),
            "--start_idx", str(start_idx),
            "--device", req.device,
        ]
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])

        env = {
            **os.environ,
            "PYTHONPATH": f"{POCKETLIG_ROOT}/src:{TARGETDIFF_ROOT}:.",
        }

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(TARGETDIFF_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            # Read stderr for tqdm progress
            step_re = re.compile(r"sampling:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)")
            simple_re = re.compile(r"(\d+)/(\d+)")

            while True:
                line = process.stderr.readline() if process.stderr else ""
                if not line and process.poll() is not None:
                    break
                if line:
                    m = step_re.search(line)
                    if m:
                        pct = int(m.group(1))
                        cur = int(m.group(2))
                        tot = int(m.group(3))
                        job["current_step"] = cur
                        job["total_steps"] = tot
                        job["progress"] = min(80, max(2, int((pct / 100) * 80)))
                    else:
                        m2 = simple_re.search(line)
                        if m2:
                            cur = int(m2.group(1))
                            tot = int(m2.group(2))
                            job["current_step"] = cur
                            job["total_steps"] = tot
                            job["progress"] = min(80, max(2, int((cur / tot) * 80)))

            rc = process.wait()
            if rc != 0:
                err_text = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"Diffusion sampling exited with code {rc}: {err_text[:300]}")

            # Export trajectories to frames.json and ensure final.sdf
            job["progress"] = 85
            from ..generation.traj_io import npz_to_frontend_json, mol_from_final_frame
            from rdkit import Chem
            generated_mols = []
            for k in range(start_idx, start_idx + req.num_molecules):
                mol_k = f"mol_{k}"
                mol_path = traj_dir / mol_k
                npz = mol_path / "frames.npz"
                if npz.exists():
                    npz_to_frontend_json(npz, mol_path / "frames.json")
                    sdf_path = mol_path / "final.sdf"
                    if not sdf_path.exists():
                        try:
                            with np.load(npz, allow_pickle=False) as z:
                                pos = z["positions"]
                                zs = z["atomic_numbers"]
                                mol, err = mol_from_final_frame(pos[-1], zs[-1])
                                if mol is None:
                                    from rdkit.Chem import rdDetermineBonds
                                    rw = Chem.RWMol()
                                    for an in zs[-1]:
                                        rw.AddAtom(Chem.Atom(int(an)))
                                    conf = Chem.Conformer(len(zs[-1]))
                                    for ai, ap in enumerate(pos[-1]):
                                        conf.SetAtomPosition(ai, tuple(float(v) for v in ap))
                                    rw.AddConformer(conf)
                                    try:
                                        rdDetermineBonds.DetermineConnectivity(rw)
                                        for atom in rw.GetAtoms():
                                            an = atom.GetAtomicNum()
                                            ev = atom.GetTotalValence()
                                            if an == 7 and ev == 4:
                                                atom.SetFormalCharge(1)
                                            elif an == 8 and ev == 3:
                                                atom.SetFormalCharge(1)
                                            elif an == 16 and ev == 3:
                                                atom.SetFormalCharge(1)
                                        try:
                                            Chem.SanitizeMol(rw)
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                    mol = rw.GetMol()
                                if mol is not None:
                                    w = Chem.SDWriter(str(sdf_path))
                                    w.write(mol)
                                    w.close()
                        except Exception:
                            pass
                    generated_mols.append(mol_k)

            # Re-evaluate multi-objective scores (GNN affinity, PoseBusters, QED, SA)
            job["progress"] = 92
            from ..eval.run_eval import evaluate_folder
            pdbqt_path = POCKETLIG_ROOT / "data" / "pdbqt_cache" / f"{tid}.pdbqt"
            gnn_path = POCKETLIG_ROOT / "runs" / "gnn_s0" / "best.pt"
            res = evaluate_folder(
                str(traj_dir),
                str(pocket_pdb),
                str(pdbqt_path) if pdbqt_path.exists() else None,
                str(gnn_path) if gnn_path.exists() else None,
            )
            (tdir / "metrics.json").write_text(json.dumps(res, indent=2))

            manifest_path = tdir / "manifest.json"
            if manifest_path.exists():
                try:
                    man = json.loads(manifest_path.read_text())
                    man["date"] = datetime.now().strftime("%Y-%m-%d")
                    manifest_path.write_text(json.dumps(man, indent=2))
                except Exception:
                    pass

            job["molecules_generated"] = generated_mols
            job["progress"] = 100
            job["status"] = "completed"

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)

    @app.post("/targets/{tid}/generate")
    def trigger_generation(tid: str, req: GenerateRequest, background_tasks: BackgroundTasks):
        tdir = target_dir(tid)
        pocket_pdb = tdir / "pocket.pdb"
        if not pocket_pdb.exists():
            raise HTTPException(400, "Target has no pocket.pdb")

        job_id = str(uuid.uuid4())[:8]
        JOBS[job_id] = {
            "id": job_id,
            "target_id": tid,
            "status": "pending",
            "progress": 0,
            "current_step": 0,
            "total_steps": req.diffusion_steps,
            "molecules_generated": [],
            "error": None,
            "created_at": datetime.now().isoformat(),
        }

        # Run worker in background thread
        thread = threading.Thread(target=_run_diffusion_worker, args=(job_id, tid, req), daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "started", "target_id": tid}

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found")
        return JOBS[job_id]

    @app.get("/jobs", response_model=list[JobStatus])
    def list_jobs():
        return list(JOBS.values())

    return app


app = create_app(os.environ.get("POCKETLIG_ARTIFACTS", "artifacts"))
