"""Resumable TargetDiff sampling for ONE pocket, saving ~50 trajectory frames per molecule.

NOT run in the build sandbox (needs a GPU, the TargetDiff checkpoint and its dependencies).
Run it FROM THE ROOT OF A TargetDiff CLONE (github.com/guanjq/targetdiff, MIT license), with this repo's
`src` on PYTHONPATH:

  cd targetdiff
  PYTHONPATH=/path/to/pocketlig/src:. python /path/to/pocketlig/scripts/targetdiff_sample_pocket.py \
      configs/sampling.yml --pdb_path pocket.pdb --out_dir /path/to/pocketlig/artifacts/<target>/trajectories \
      --num_samples 100 --chunk 10 --n_frames 50 --device cuda:0

It follows TargetDiff's own scripts/sample_for_pocket.py (same model loading, featurizers, sampler,
reconstruction) and only adds: chunking, per-molecule skip-if-exists, atomic writes, frame subsampling.
Determinism: a chunk is seeded with seed + chunk_start, so re-running a partially finished chunk regenerates
identical molecules (same chunk size and same environment required).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem  # noqa: E402
from torch_geometric.transforms import Compose  # noqa: E402

import utils.misc as misc  # noqa: E402  (TargetDiff)
import utils.transforms as trans  # noqa: E402
from models.molopt_score_model import ScorePosNet3D  # noqa: E402
from scripts.sample_diffusion import sample_diffusion_ligand  # noqa: E402
from scripts.sample_for_pocket import pdb_to_pocket_data  # noqa: E402
from utils import reconstruct  # noqa: E402

from pocketlig.generation.traj_io import atomic_save_npz, npz_is_valid, select_frames  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pdb_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    config = misc.load_config(args.config)
    seed = args.seed if args.seed is not None else config.sample.seed
    num_steps = args.num_steps if args.num_steps is not None else int(config.sample.num_steps)
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt["config"].data.transform.ligand_atom_mode)
    model = ScorePosNet3D(ckpt["config"].model, protein_atom_feature_dim=protein_featurizer.feature_dim,
                          ligand_atom_feature_dim=ligand_featurizer.feature_dim).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False if "train_config" in config.model else True)
    data = Compose([protein_featurizer])(pdb_to_pocket_data(args.pdb_path))

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"checkpoint": config.model.checkpoint, "checkpoint_sha256": sha256(config.model.checkpoint),
            "num_steps": int(num_steps), "seed": seed, "chunk": args.chunk,
            "torch": torch.__version__, "pdb_path": args.pdb_path, "chunks": []}
    start_offset = args.start_idx
    for chunk_start in range(0, args.num_samples, args.chunk):
        start = start_offset + chunk_start
        n = min(args.chunk, args.num_samples - chunk_start)
        paths = [os.path.join(args.out_dir, f"mol_{k}", "frames.npz") for k in range(start, start + n)]
        if all(npz_is_valid(p) for p in paths):
            print(f"chunk {start}: all done, skipping")
            continue
        misc.seed_all(seed + start)
        t0 = time.time()
        pos_final, v_final, pos_traj, v_traj, *_ = sample_diffusion_ligand(
            model, data, n, batch_size=n, device=args.device, num_steps=num_steps,
            pos_only=config.sample.pos_only, center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms)
        for j in range(n):
            k, path = start + j, paths[j]
            if npz_is_valid(path):
                continue
            mol_dir = os.path.dirname(path)
            os.makedirs(mol_dir, exist_ok=True)
            idx = select_frames(pos_traj[j].shape[0], args.n_frames)
            z_frames = np.stack([np.array(trans.get_atomic_number_from_index(torch.tensor(v_traj[j][i]), mode="add_aromatic"))
                                 for i in idx])
            aromatic = trans.is_aromatic_from_index(torch.tensor(v_final[j]), mode="add_aromatic")
            smiles, ok = "", False
            mol = None
            try:
                mol = reconstruct.reconstruct_from_generated(pos_final[j], z_frames[-1].tolist(), aromatic)
                smiles = Chem.MolToSmiles(mol)
                ok = "." not in smiles
            except Exception:
                pass

            if mol is None:
                try:
                    from rdkit.Chem import rdDetermineBonds
                    rw = Chem.RWMol()
                    for z in z_frames[-1].tolist():
                        rw.AddAtom(Chem.Atom(int(z)))
                    conf = Chem.Conformer(len(z_frames[-1]))
                    for ai, ap in enumerate(pos_final[j]):
                        conf.SetAtomPosition(ai, tuple(float(v) for v in ap))
                    rw.AddConformer(conf)
                    rdDetermineBonds.DetermineConnectivity(rw)
                    mol = rw.GetMol()
                    try:
                        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                    except Exception:
                        pass
                    smiles = Chem.MolToSmiles(mol)
                    ok = bool(smiles)
                except Exception:
                    pass

            if mol is not None:
                try:
                    w = Chem.SDWriter(os.path.join(mol_dir, "final.sdf"))
                    w.write(mol)
                    w.close()
                except Exception:
                    pass
            atomic_save_npz(path, positions=pos_traj[j][idx].astype(np.float32), atomic_numbers=z_frames,
                            step_indices=idx, final_ok=np.array(ok), final_smiles=np.array(smiles),
                            seed=np.array(seed + start), num_steps=np.array(num_steps))
        meta["chunks"].append({"start": start, "n": n, "seconds": round(time.time() - t0, 1)})
        print(f"chunk {start}: {n} molecules in {time.time() - t0:.0f}s")
    with open(os.path.join(args.out_dir, "sampling_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
