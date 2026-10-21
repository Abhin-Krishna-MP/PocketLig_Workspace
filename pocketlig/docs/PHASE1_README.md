# Phase 1 Documentation: Data Ingestion, Benchmark IDs & TargetDiff Integration

This document provides a complete record of all setup steps, data preparation, benchmark curation, model checkpoint acquisitions, structural verifications, and workspace reorganizations completed during **Phase 1**.

---

## 1. Directory & Workspace Structure

Both `pocketlig` and `targetdiff` have been organized into a unified root workspace folder:

```
/home/abhin-krishna-m-p/pocketlig_workspace/
├── pocketlig/                            # Main project repository
│   ├── data/
│   │   ├── pdbbind/                      # PDBbind v2020 refined set & plain text index
│   │   │   ├── index/                    # INDEX_refined_data.2020, INDEX_general_PL_data.2020, etc.
│   │   │   ├── refined-set/              # 5,316 refined complexes (<id>/<id>_protein.pdb, etc.)
│   │   │   ├── PDBbind_v2020_plain_text_index.tar.gz
│   │   │   └── PDBbind_v2020_refined.tar.gz
│   │   ├── casf2016_ids.txt              # 285 CASF-2016 Core Set PDB IDs (1 per line)
│   │   ├── posebusters_ids.txt           # 428 PoseBusters benchmark PDB IDs (1 per line)
│   │   ├── posebusters_428_ids.txt       # Copy of full 428-set IDs
│   │   ├── posebusters_308_ids.txt       # Filtered 308-set IDs (no crystal artifacts)
│   │   └── index_table.csv               # Parsed affinity table (5,316 rows)
│   ├── docs/                             # Project documentation & status
│   ├── scripts/                          # Precomputation & sampling scripts
│   ├── src/pocketlig/                    # Core pipeline package
│   │   ├── data/                         # parse_index, make_splits, build_dataset, pocket, graph
│   │   ├── eval/                         # Affinity & PoseBusters evaluation metrics
│   │   ├── generation/                   # Trajectory IO & frame selection
│   │   └── models/                       # GNN affinity model architectures & training
│   ├── tests/                            # Unit & integration tests (19 test cases)
│   └── .venv/                            # Python 3.11 virtual environment
└── targetdiff/                           # Cloned TargetDiff generator repository
    ├── configs/                          # Model & sampling configs (e.g. sampling.yml)
    ├── models/                           # 3D equivariant score networks
    ├── pretrained_models/                # TargetDiff pretrained weights
    │   ├── pretrained_diffusion.pt       # Checkpoint (iteration 176k, 384 parameter tensors)
    │   ├── egnn_pdbbind_v2016.pt         # Supervised EGNN affinity model
    │   └── pk_reg_para.pkl               # Regression parameters
    └── scripts/                          # Sampling & evaluation scripts
```

> **Symlink Compatibility:**  
> For backward compatibility with existing terminal paths and external scripts, symlinks are preserved in `~`:
> - `~/pocketlig` -> `/home/abhin-krishna-m-p/pocketlig_workspace/pocketlig`
> - `~/targetdiff` -> `/home/abhin-krishna-m-p/pocketlig_workspace/targetdiff`

---

## 2. Environment Configuration

- **Python Version**: `3.11.17`
- **Virtual Environment**: `.venv`
- **Key Libraries**:
  - `torch` & `torch_geometric` (PyG)
  - `rdkit` (2025/2026 release)
  - `biopython`
  - `pandas`, `numpy`, `scipy`
  - `fastapi`, `httpx`
  - `pytest`
- **Added Dependencies**:
  - `gdown`: Installed for programmatic downloading of shared artifacts.
  - `easydict`: Installed to deserialize TargetDiff checkpoint configurations.
  - `pocketlig.pth`: Configured in `.venv/lib/python3.11/site-packages/` pointing to `src/`, enabling clean module imports (`import pocketlig`) without requiring manual `PYTHONPATH` exports.

---

## 3. PDBbind Dataset Setup (`data/pdbbind/`)

### Data Ingestion
The raw archives provided in `~/Downloads/` were copied to `data/pdbbind/`:
1. `PDBbind_v2020_refined.tar.gz` (659 MB)
2. `PDBbind_v2020_plain_text_index.tar.gz` (3.1 MB)

### Archive Extraction
- Extracted `PDBbind_v2020_plain_text_index.tar.gz` to `data/pdbbind/index/`:
  - Contains `INDEX_refined_data.2020`, `INDEX_general_PL_data.2020`, `INDEX_structure.2020`, and documentation.
- Extracted `PDBbind_v2020_refined.tar.gz` to `data/pdbbind/refined-set/`:
  - Contains **5,316** refined protein-ligand complexes.

### Compliance Rule
- Verified that `data/` is strictly ignored in `.gitignore` (line 1: `data/`). Raw PDBbind files, indices, and processed tensors are prevented from being tracked or committed to version control.

### Index Parsing Validation
Executed the index parser:
```bash
python -m pocketlig.data.parse_index data/pdbbind/index/INDEX_refined_data.2020 --out data/index_table.csv
```
- **Result**:
  - Total complexes parsed: **5,316**
  - Failures: **0**
  - Affinity breakdown: `Kd`: 2,783, `Ki`: 2,533
  - Flagged for inexact operators (`<`, `>`): 2
  - Output written to: `data/index_table.csv`

---

## 4. Benchmark IDs Gathering

### CASF-2016 Core Set (`data/casf2016_ids.txt`)
- Source: `/home/abhin-krishna-m-p/Downloads/CASF-2016.tar.gz` (`CASF-2016/power_screening/CoreSet.dat`).
- Extracted **285 unique PDB IDs** (4-character lowercase strings, sorted, 1 ID per line).
- Saved to: `data/casf2016_ids.txt` (285 lines).

### PoseBusters Benchmark (`data/posebusters_ids.txt`)
- Source: Official publication repository & Zenodo record (`maabuu/posebusters` / Zenodo ID: `8278563`).
- Extracted **428 unique PDB IDs** from the primary `posebusters_benchmark_set_ids.txt`.
- Saved to:
  - `data/posebusters_ids.txt` (428 lines, standard benchmark set).
  - `data/posebusters_428_ids.txt` (full 428-complex set).
  - `data/posebusters_308_ids.txt` (filtered 308-complex subset without crystallographic artifacts, from `oxpig/RLDiff`).

### Leakage Analysis
- Cross-referenced all PoseBusters benchmark IDs against PDBbind v2020 refined set:
  - **Direct PDB ID Overlap: 0** (expected, as PoseBusters complexes were deposited in 2021+, post-dating PDBbind v2020).
  - Downstream MMseqs2 sequence clustering will further exclude any refined-set targets with >30% sequence homology to these benchmark targets.

---

## 5. TargetDiff Integration & Checkpoint Setup

### Repository Clone
Cloned from `https://github.com/guanjq/targetdiff` into:
```bash
/home/abhin-krishna-m-p/pocketlig_workspace/targetdiff
```

### Pretrained Weights
Downloaded the official model weights mirrored from the author's archive on Zenodo (`targetdiff_pretrained_models.tar.gz`) into `targetdiff/pretrained_models/`:
- `pretrained_diffusion.pt` (33 MB)
  - Iteration: **176,000**
  - Parameter tensors: **384** layers
- `egnn_pdbbind_v2016.pt` (31 MB)
  - Supervised baseline affinity model
- `pk_reg_para.pkl`
  - Linear regression calibration parameters

### Checkpoint Validation
Verified with PyTorch that the checkpoint deserializes completely:
```python
import torch, easydict
ckpt = torch.load('targetdiff/pretrained_models/pretrained_diffusion.pt', map_location='cpu')
# Confirmed keys: ['config', 'model', 'optimizer', 'scheduler', 'iteration']
```

---

## 6. Pass Condition: PDBbind Folder Layout & Data Builder

### Folder Layout Verification
Inspected extracted complexes under `data/pdbbind/refined-set/`:
```
data/pdbbind/refined-set/<pdb_id>/
├── <pdb_id>_protein.pdb      # Cleaned receptor
├── <pdb_id>_ligand.sdf       # Co-crystallized ligand (SDF)
├── <pdb_id>_ligand.mol2      # Co-crystallized ligand (MOL2)
└── <pdb_id>_pocket.pdb       # 10 Å pocket slice
```
The naming convention matches the required layout `<root>/<id>/<id>_protein.pdb` and `<id>_ligand.sdf` exactly.

### Code Verification in `src/pocketlig/data/build_dataset.py`
Examined `build_one()`:
```python
def build_one(root: Path, pdb_id: str, y: float, cutoff_pocket: float, cutoff_edge: float):
    d = root / pdb_id
    lig = load_ligand(d / f"{pdb_id}_ligand.sdf", d / f"{pdb_id}_ligand.mol2")
    st = load_structure(d / f"{pdb_id}_protein.pdb")
    residues = extract_pocket_residues(st, ligand_heavy_coords(lig), cutoff_pocket)
    graph = build_graph(atom_table(residues, lig), cutoff_edge)
    return to_pyg(graph, y=y, pdb_id=pdb_id)
```
- No changes to filename patterns were needed since the downloaded files already conform 100%.
- Executed `build_one()` on real refined complex `1a30`:
  - Successfully produced PyG graph: `Data(x=[548, 40], edge_index=[2, 11490], edge_attr=[11490, 12], pos=[548, 3], is_ligand=[548], y=[1], pdb_id='1a30')`.

---

## 7. Test Suite Status

Executed `pytest -q` across all modules:
```
19 passed, 3 warnings in 32.27s
```
Covered tests:
- `test_affinity_units`
- `test_parse_index`
- `test_comment_returns_none`
- `test_temporal_split_and_leak_detected_then_fixed`
- `test_benchmark_overlap`
- `test_read_cluster_tsv`
- `test_metrics_known`
- `test_make_splits_end_to_end_clean`
- `test_clean_and_pocket`
- `test_graph_shapes_and_invariance`
- `test_model_invariance`
- `test_overfit_20_samples`
- `test_resume`
- `test_select_frames`
- `test_traj_roundtrip_and_final_bonding`
- `test_dx_interpolation`
- `test_eval_records_failures_not_drops`
- `test_api_and_validation`
- `test_ligand_only_learns_simple_signal`

---

---

## 8. Data Splits & Leakage Report (Gate 1 Passed)

Executed leakage-controlled split generation using MMseqs2:
```bash
python -m pocketlig.data.make_splits --index-table data/index_table.csv \
    --root data/pdbbind/refined-set \
    --benchmark-ids data/casf2016_ids.txt data/posebusters_ids.txt \
    --out data/splits.csv
```

### Leakage Report Results (`data/leakage_report.json`):
- `test_complexes_sharing_cluster_with_train`: **0**
- `test_complexes_sharing_cluster_with_val`: **0**
- `val_complexes_sharing_cluster_with_train`: **0**
- `pdb_ids_in_both_train_and_test`: **0**
- `complexes_without_cluster_info`: **0**
- `clean`: **True**

### Final Split Distribution:
- **Train**: 1,789 complexes (<= 2017, no overlap with val, test, or benchmarks)
- **Val**: 162 complexes (2018, no overlap with test or benchmarks)
- **Test**: 425 complexes (>= 2019, including held-out benchmark targets)
- **Dropped (test cluster overlap)**: 2,202 complexes
- **Dropped (benchmark cluster overlap)**: 390 complexes
- **Dropped (val cluster overlap)**: 346 complexes

---

## 9. Next Steps (Dataset Graph Construction & Baselines)

1. **Build Processed Graph Dataset**:
   ```bash
   python -m pocketlig.data.build_dataset --root data/pdbbind/refined-set \
       --splits data/splits.csv --out data/processed
   ```
2. **Model Training & Baselines (Phase 2 & 3)**:
   - Run overfit check (20 samples) to verify optimization.
   - Train affinity GNN with 5 seeds (`runs/gnn_s0` through `s4`).

