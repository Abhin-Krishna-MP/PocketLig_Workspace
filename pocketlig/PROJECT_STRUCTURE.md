# PocketLig: Complete Project Structure & File-by-File Architectural Reference

This document provides a detailed, comprehensive reference for every directory, source code module, dataset artifact, model checkpoint, baseline driver, evaluation script, web component, and configuration file in the **PocketLig** project.

---

## 1. High-Level System Architecture

PocketLig is an end-to-end framework for generative structure-based drug design (SBDD) and binding affinity prediction:
1. **Pocket Extraction & Featurization**: Extracts 10 Å receptor binding pockets around co-crystallized or generated ligands and builds heterogeneous 3D interaction graphs with distance-based contact edges ($\le 5$ Å).
2. **Leakage-Controlled Data Engine**: Employs MMseqs2 sequence clustering (30% identity, 80% coverage) and temporal partitioning to guarantee zero homology leakage into evaluation benchmarks.
3. **Baselines & Negative Controls**: Evaluates non-structural 2D ligand-only Gradient Boosting (217 RDKit descriptors) and empirical AutoDock Vina crystal-pose scoring power to verify that learned representations exceed chemical memorization.
4. **3D Interaction Graph Neural Network (`AffinityGNN`)**: 4-layer spatial message-passing network with coordinate invariance, radial basis edge projections, and multi-seed ensembling.
5. **Diffusion-Based Ligand Generation**: Interfaces with pretrained 3D equivariant score-based diffusion models (TargetDiff) for de novo pocket-conditioned molecule generation.
6. **Web Interface & 3D Visualization**: FastAPI backend paired with a React / TypeScript / Vite frontend rendering 2D chemical topologies and interactive 3D protein-ligand complexes.

---

## 2. Directory Tree Overview

```
/home/abhin-krishna-m-p/pocketlig_workspace/
├── pocketlig/                                # Main project repository
│   ├── .venv/                                # Python 3.11 virtual environment
│   ├── coreset/                              # CASF-2016 core complexes (285 targets)
│   ├── data/                                 # Raw data, indices, graphs & cache
│   │   ├── CASF-2016/                        # Extracted CASF-2016 structures
│   │   ├── mmseqs_tmp/                       # MMseqs2 temporary clustering workspace
│   │   ├── pdbbind/                          # PDBbind v2020 refined set & plain text indices
│   │   ├── pdbqt_cache/                      # 872 prepared receptor PDBQT files
│   │   ├── processed/                        # 2,659 PyG graph tensors (<id>.pt) & failure log
│   │   ├── casf2016_ids.txt                  # 285 CASF-2016 PDB IDs
│   │   ├── casf2016_table.csv                # 285 CASF-2016 ground-truth labels
│   │   ├── index_table.csv                   # 5,316 parsed PDBbind affinity records
│   │   ├── leakage_report.json               # Zero-leakage certification report
│   │   ├── mmseqs_all_seqs.fasta             # All target protein sequences
│   │   ├── mmseqs_cluster.tsv                # Sequence cluster assignments (30% id)
│   │   ├── mmseqs_rep_seq.fasta              # Cluster representative sequences
│   │   ├── posebusters_308_ids.txt           # 308 PoseBusters filtered subset IDs
│   │   ├── posebusters_428_ids.txt           # 428 PoseBusters full set IDs
│   │   ├── posebusters_ids.txt               # Primary PoseBusters benchmark IDs
│   │   ├── sequences.fasta                   # Extracted receptor FASTA sequences
│   │   └── splits.csv                        # Final dataset split table (train/val/test/dropped)
│   ├── docs/                                 # Project decisions & technical documentation
│   │   ├── DECISIONS.md                      # Chronological decisions & metric targets
│   │   ├── ENVIRONMENT.md                    # Virtualenv & tooling configuration
│   │   └── STATUS.md                         # Milestone tracking
│   ├── results/                              # Evaluation outputs & benchmark summaries
│   │   ├── baselines.csv                     # Master baseline metrics (ligand_only & vina)
│   │   ├── gnn_summary.csv                   # Master 5-seed GNN & ensemble comparison
│   │   ├── preds_gnn_ensemble_casf2016.csv   # GNN ensemble predictions on CASF-2016
│   │   ├── preds_gnn_ensemble_test.csv       # GNN ensemble predictions on Test split
│   │   ├── preds_ligand_only*.csv            # Ligand-only baseline predictions (per-split & all)
│   │   └── preds_vina*.csv                   # Vina baseline predictions (per-split & all)
│   ├── runs/                                 # Training checkpoint runs
│   │   ├── gnn_s0/                           # Seed 0 run artifacts (best.pt, last.pt, metrics)
│   │   ├── gnn_s1/                           # Seed 1 run artifacts
│   │   ├── gnn_s2/                           # Seed 2 run artifacts
│   │   ├── gnn_s3/                           # Seed 3 run artifacts
│   │   ├── gnn_s4/                           # Seed 4 run artifacts
│   │   └── overfit/                          # 20-sample overfit check artifacts
│   ├── scripts/                              # Standalone generation & utility scripts
│   │   └── targetdiff_sample_pocket.py       # Pocket-conditioned TargetDiff generation
│   ├── src/                                  # Source package root
│   │   └── pocketlig/                        # Core Python library
│   │       ├── api/                          # FastAPI web service endpoints
│   │       ├── baselines/                    # Ligand-only GBR & AutoDock Vina drivers
│   │       ├── data/                         # Parsing, splitting, pocket extraction, graphs
│   │       ├── eval/                         # CASF-style scoring power & metrics
│   │       ├── generation/                   # Trajectory IO & frame selection
│   │       ├── models/                       # AffinityGNN, training loop, evaluation driver
│   │       ├── precompute/                   # Electrostatics & target validation
│   │       └── synthetic.py                  # Synthetic data generator for unit tests
│   ├── tests/                                # Test suite (19 test cases)
│   ├── web/                                  # React + Vite web application
│   ├── Dockerfile                            # Container deployment definition
│   ├── environment.yml                       # Conda environment definition
│   ├── Makefile                              # Automation commands
│   ├── PHASE1_README.md                      # Phase 1 setup & data splits documentation
│   ├── PHASE2_README.md                      # Phase 2 graph building & baselines documentation
│   ├── PHASE3A_README.md                     # Phase 3A production training & benchmarking
│   ├── PROJECT_STRUCTURE.md                  # Complete architectural map (this file)
│   ├── pytest.ini                            # Pytest configuration
│   ├── README.md                             # Repository overview
│   └── requirements.txt                      # Pip dependency requirements
└── targetdiff/                               # Cloned TargetDiff generator repository
    ├── configs/                              # TargetDiff YAML configuration files
    ├── models/                               # Equivariant score-based diffusion network
    ├── pretrained_models/                    # TargetDiff pretrained checkpoints (.pt)
    └── scripts/                              # Sampling & evaluation scripts
```

---

## 3. Detailed File-by-File Breakdown: Source Code (`src/pocketlig/`)

### 3.1 Data Ingestion, Splitting & Graph Building (`src/pocketlig/data/`)

#### [`src/pocketlig/data/parse_index.py`](src/pocketlig/data/parse_index.py)
- **Role**: Parses PDBbind raw index tables (`INDEX_refined_data.2020` or `INDEX_general_PL_data.2020`).
- **Key Functions**:
  - `parse_affinity_string(raw_aff: str) -> dict`: Extracts operator (`=`, `<`, `>`), value, and converts $K_d$, $K_i$, or $\text{IC}_{50}$ with arbitrary units (mM, $\mu$M, nM, pM, fM) to standardized $pK = -\log_{10}(K)$.
  - `parse_index_line(line: str) -> dict | None`: Parses individual whitespace-delimited PDBbind table entries, resolution, release year, reference, and ligand code.
  - `parse_index(file_path: Path) -> tuple[pd.DataFrame, list]`: Processes the complete file, flagging inexact operators or label mismatches.
- **Output**: Produces clean dataframes for downstream ingestion.

#### [`src/pocketlig/data/sequences.py`](src/pocketlig/data/sequences.py)
- **Role**: Extracts target protein amino acid sequences from PDB coordinates.
- **Key Functions**:
  - `extract_sequences(pdb_path: Path) -> dict[str, str]`: Reads standard residues per chain using BioPython and constructs FASTA-format sequence strings.
  - `dump_fasta(sequences: dict, out_path: Path)`: Writes consolidated FASTA files for sequence alignment.

#### [`src/pocketlig/data/split.py`](src/pocketlig/data/split.py)
- **Role**: Implements temporal partitioning and homology leakage enforcement algorithms.
- **Key Functions**:
  - `temporal_split(df: pd.DataFrame, train_cutoff=2017, val_year=2018) -> pd.Series`: Assigns initial partitions based on release year.
  - `read_cluster_tsv(cluster_tsv: Path) -> dict[str, frozenset]`: Ingests MMseqs2 clustering output mapping each PDB ID to its sequence cluster.
  - `enforce_cluster_separation(...) -> list[str]`: Removes training complexes that share sequence clusters with validation or test complexes.
  - `remove_benchmark_overlap(...) -> list[str]`: Excludes any training/validation targets with $>30\%$ identity to CASF-2016 or PoseBusters.
  - `leakage_report(...) -> dict`: Calculates a 4-way leakage audit report certifying zero cluster or ID overlaps.

#### [`src/pocketlig/data/make_splits.py`](src/pocketlig/data/make_splits.py)
- **Role**: CLI orchestrator for MMseqs2 clustering and leakage-controlled split generation.
- **CLI Usage**:
  ```bash
  python -m pocketlig.data.make_splits --index-table data/index_table.csv \
      --root data/pdbbind/refined-set \
      --benchmark-ids data/casf2016_ids.txt data/posebusters_ids.txt \
      --out data/splits.csv
  ```
- **Outputs**: Writes `data/splits.csv` and `data/leakage_report.json`.

#### [`src/pocketlig/data/pocket.py`](src/pocketlig/data/pocket.py)
- **Role**: 3D structural cleaning, robust ligand loading, and pocket extraction.
- **Key Functions**:
  - `load_ligand(sdf_path, mol2_path) -> Chem.Mol`: Loads 3D conformer coordinates; preferentially parses SDF, with automatic fallback to Tripos MOL2 (rescuing 305 unkekulizable SDFs).
  - `clean_protein(pdb_path, out_path) -> dict`: Keeps model 0, standard amino acids, altloc `' '` or `'A'`, discarding waters, ions, and crystallization reagents.
  - `extract_pocket_residues(structure, ligand_xyz, cutoff=10.0) -> list`: Performs a `cKDTree` spherical query to extract all amino acid residues having any heavy atom within 10.0 Å of any ligand heavy atom.
  - `ligand_heavy_coords(mol: Chem.Mol) -> np.ndarray`: Extracts $(N, 3)$ coordinate arrays for non-hydrogen ligand atoms.

#### [`src/pocketlig/data/graph.py`](src/pocketlig/data/graph.py)
- **Role**: Constructs heterogeneous 3D interaction graphs from structural coordinates.
- **Constants**: `NODE_DIM = 40`, `EDGE_DIM = 12`.
- **Key Functions**:
  - `atom_table(residues, ligand_mol) -> AtomTable`: Constructs unified node arrays combining protein pocket atoms (20 amino acid types, element, coordinates) and ligand atoms (atomic number, hybridization, aromaticity, formal charge, coordinates).
  - `build_graph(table: AtomTable, cutoff=5.0) -> dict`: Generates spatial contact edges via radius search ($\le 5.0$ Å), computing Euclidean distances and radial basis projections.
  - `to_pyg(graph_dict, y, pdb_id) -> torch_geometric.data.Data`: Converts graph dictionaries into standard PyTorch Geometric tensor objects.

#### [`src/pocketlig/data/build_dataset.py`](src/pocketlig/data/build_dataset.py)
- **Role**: High-throughput multi-worker dataset builder converting raw files into PyG `.pt` tensors.
- **Key Features**:
  - Multi-process execution via `ProcessPoolExecutor`.
  - Automatic perception of standard `splits.csv` or headerless benchmark tables (`casf2016_table.csv`).
  - Idempotent execution (skips previously built `.pt` tensors).
  - Preserves and merges failure logs in `data/processed/failures.csv`.
- **CLI Usage**:
  ```bash
  python -m pocketlig.data.build_dataset --root data/pdbbind/refined-set --splits data/splits.csv --out data/processed
  ```

---

### 3.2 Machine Learning Models & Inference (`src/pocketlig/models/`)

#### [`src/pocketlig/models/gnn.py`](src/pocketlig/models/gnn.py)
- **Role**: 3D spatial Interaction Graph Neural Network (`AffinityGNN`).
- **Architecture**:
  - **Node Embedding**: Linear projection of 40-dimensional node features to hidden dimension $H=128$.
  - **Interaction Layers (`InteractionLayer`)**: 4 stacked message-passing layers utilizing `torch_geometric.nn.MessagePassing`:
    $$\mathbf{m}_{ij} = \text{MLP}([\mathbf{h}_i \,\|\, \mathbf{h}_j \,\|\, \mathbf{e}_{ij}])$$
    $$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{LayerNorm}\left(\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}\right)$$
  - **Pooling & Readout**: Global additive pooling (`global_add_pool`) aggregating node embeddings across pocket and ligand, followed by a 2-layer MLP producing scalar binding affinity $pK$.
  - **Invariance**: Strict SE(3)-invariance with respect to 3D rotations and translations.

#### [`src/pocketlig/models/train.py`](src/pocketlig/models/train.py)
- **Role**: Production training loop with early stopping, checkpointing, and GPU memory safety.
- **Key Functions**:
  - `fit(train_set, val_set, out_dir, epochs=100, lr=1e-3, hidden=128, layers=4, batch_size=16, patience=20, ...)`: Optimizes MSE loss with Adam (`weight_decay=1e-5`), gradient clipping (norm 5.0), saving `best.pt` on validation improvement and `last.pt` each epoch.
  - `predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]`: Batch inference returning true values, predictions, and PDB IDs.
  - `load_split(processed, splits_csv)`: Deserializes PyG graph lists for train, val, and test splits.
- **Memory Configurations**: Defaults `batch_size=16`, configures `expandable_segments:True`, and flushes CUDA cache between epochs.

#### [`src/pocketlig/models/evaluate.py`](src/pocketlig/models/evaluate.py)
- **Role**: Comprehensive evaluation driver across multi-seed runs and benchmark datasets.
- **Key Features**:
  - Runs inference on both the 425 out-of-distribution Test targets and the 285 CASF-2016 Core Set targets.
  - Computes per-seed metrics and saves predictions (`pred_test.csv`, `pred_casf2016.csv`).
  - Evaluates the 5-model ensemble:
    $$\hat{y}_{\text{ensemble}} = \frac{1}{5}\sum_{s=0}^{4} \hat{y}_s$$
  - Compiles master benchmark comparison table in `results/gnn_summary.csv`.

---

### 3.3 Baselines & Controls (`src/pocketlig/baselines/`)

#### [`src/pocketlig/baselines/ligand_only.py`](src/pocketlig/baselines/ligand_only.py)
- **Role**: Negative control baseline utilizing only 2D/3D chemical properties of the ligand.
- **Key Functions**:
  - `featurize(mol: Chem.Mol) -> np.ndarray`: Computes 217 RDKit descriptors (`rdkit.Chem.Descriptors.descList`), handling exceptions with zero-fills and clipping values to prevent `float32` numerical overflow.
  - `fit_predict(train_mols, y_train, test_mols, seed=0) -> np.ndarray`: Fits a `GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8)`.

#### [`src/pocketlig/baselines/run_ligand_only.py`](src/pocketlig/baselines/run_ligand_only.py)
- **Role**: Driver script for training the ligand-only GBR on the 1,787 train complexes and evaluating on `val`, `test`, and `casf2016`.
- **Outputs**: Writes `results/preds_ligand_only.csv` and updates `results/baselines.csv`.

#### [`src/pocketlig/baselines/vina_score.py`](src/pocketlig/baselines/vina_score.py)
- **Role**: Classical physics-based empirical scoring baseline using AutoDock Vina.
- **Key Functions**:
  - `prepare_receptor(pdb_path, out_pdbqt)`: Uses BioPython's `_KeepStandardAAHeavyAtoms` to strip partial crystal hydrogens and non-templated ions before calling Meeko's `mk_prepare_receptor.py -p -x`.
  - `_strip_query_flags(mol: Chem.Mol) -> Chem.Mol`: Reconstructs RDKit molecules without MDL V2000 query bond types (X-TOOL type 4 aromatic bonds) so Meeko can prepare ligand PDBQT without raising `ValueError`.
  - `ligand_to_pdbqt(mol: Chem.Mol) -> str`: Adds hydrogens and parameterizes the ligand via Meeko.
  - `vina_score_only(receptor_pdbqt, mol, box_padding=8.0) -> float`: Evaluates the ligand pose directly in the receptor pocket (no minimization) using Vina scoring function.

#### [`src/pocketlig/baselines/run_vina.py`](src/pocketlig/baselines/run_vina.py)
- **Role**: Multi-worker driver scoring all complexes across `val`, `test`, and `casf2016` (872 complexes total).
- **Key Features**:
  - Caches prepared receptor PDBQTs in `data/pdbqt_cache/<pdb_id>.pdbqt`.
  - Parallelized with `ProcessPoolExecutor` (10 workers).
  - Evaluates scoring power against $-Vina$ score for correlation with positive binding affinity $pK$.
  - Updates `results/baselines.csv`.

---

### 3.4 Evaluation Metrics & Generation (`src/pocketlig/eval/` & `generation/`)

#### [`src/pocketlig/eval/affinity_metrics.py`](src/pocketlig/eval/affinity_metrics.py)
- **Role**: CASF-2016 scoring power mathematical evaluation engine.
- **Key Function**:
  - `compute_metrics(y_true, y_pred) -> dict[str, float]`:
    - Pearson linear correlation ($R$): $\frac{\text{cov}(y, p)}{\sigma_y \sigma_p}$
    - Spearman rank correlation ($\rho$)
    - Root Mean Square Error (RMSE): $\sqrt{\frac{1}{N} \sum (y - p)^2}$
    - Mean Absolute Error (MAE): $\frac{1}{N} \sum |y - p|$
    - CASF Regression Standard Deviation (SD): Fits $y = a + b \cdot p$ and computes standard error $\sqrt{\frac{1}{N-2} \sum (y - (a + b \cdot p))^2}$.

#### [`src/pocketlig/eval/run_eval.py`](src/pocketlig/eval/run_eval.py)
- **Role**: Evaluation pipeline evaluating generated molecules against PoseBusters benchmarks and affinity targets.

#### [`src/pocketlig/generation/traj_io.py`](src/pocketlig/generation/traj_io.py)
- **Role**: Utilities for reading, writing, and processing diffusion model sampling trajectories.

#### [`src/pocketlig/synthetic.py`](src/pocketlig/synthetic.py)
- **Role**: Generates synthetic 3D interaction graphs with controlled shapes and coordinate arrays for fast, deterministic unit testing without requiring external datasets.

---

### 3.5 Web API & Precomputation (`src/pocketlig/api/` & `precompute/`)

#### [`src/pocketlig/api/main.py`](src/pocketlig/api/main.py)
- **Role**: FastAPI backend application serving affinity predictions, 3D structure downloads, and trajectory visualizations.
- **Endpoints**:
  - `GET /health`: Health check and model readiness.
  - `POST /predict/affinity`: Accepts protein PDB and ligand SDF/SMILES, returning predicted $pK$, $K_d$, and binding energy.
  - `GET /structure/{pdb_id}`: Retrieves cleaned 3D coordinates.

#### [`src/pocketlig/precompute/run_target.py`](src/pocketlig/precompute/run_target.py)
- **Role**: Standalone script for precomputing pocket representations and descriptors for specific drug targets.

---

## 4. Benchmark Datasets & Data Files (`data/`)

### 4.1 Datasets & Structural Archives
- **`data/pdbbind/refined-set/`**: 5,316 crystallographic protein-ligand complexes from PDBbind v2020 (`<id>/<id>_protein.pdb`, `<id>_ligand.sdf`, `<id>_ligand.mol2`, `<id>_pocket.pdb`).
- **`data/CASF-2016/coreset/`**: 285 complexes comprising the CASF-2016 Core Set evaluation benchmark.
- **`data/processed/*.pt`**: 2,659 PyTorch Geometric graph tensors containing node features, 3D positions, edge indices, radial basis edge attributes, and affinity labels.
- **`data/processed/failures.csv`**: Record of 2 verified crystal source data failures (`1wc1`, `5dhu` with hypervalent atoms).
- **`data/pdbqt_cache/*.pdbqt`**: 872 cleaned and parameterized receptor PDBQT files cached for AutoDock Vina scoring.

### 4.2 Tables, Splits & Leakage Logs
- **`data/index_table.csv`**: Parsed affinity table for 5,316 refined complexes (`pdb_id,year,resolution,pK,operator,value,unit,is_exact,flag`).
- **`data/splits.csv`**: Leakage-controlled partition table assigning complexes to `train` (1,789), `val` (162), `test` (425), and dropped overlap categories.
- **`data/casf2016_table.csv`**: 285 CASF-2016 ground-truth records (`pdb_id,year,pK`).
- **`data/leakage_report.json`**: Official certification report verifying 0 homology leakage across train, test, and benchmark clusters (`clean: true`).
- **`data/casf2016_ids.txt`**: 285 core set PDB IDs (1 per line).
- **`data/posebusters_ids.txt`**: 428 PoseBusters benchmark PDB IDs.
- **`data/posebusters_308_ids.txt`**: 308 PoseBusters subset IDs excluding crystal artifacts.

---

## 5. Experimental Results & Model Checkpoints

### 5.1 Results Tables (`results/`)

#### [`results/baselines.csv`](results/baselines.csv)
Master baseline table evaluated strictly using `affinity_metrics.py`:

| Model | Split | $N$ | Pearson $R$ | Spearman $\rho$ | RMSE | MAE | SD (CASF) |
|---|---|---|---|---|---|---|---|
| `ligand_only` | `val` | 162 | 0.471 | 0.478 | 1.404 | 1.119 | 1.401 |
| `ligand_only` | `test` | 425 | **0.419** | 0.392 | 1.680 | 1.364 | 1.649 |
| `ligand_only` | `casf2016` | 285 | **0.481** | 0.467 | 1.920 | 1.529 | 1.909 |
| `vina_score` | `val` | 162 | 0.476 | 0.489 | 2.013 | 1.546 | 1.396 |
| `vina_score` | `test` | 425 | 0.470 | 0.512 | 2.519 | 2.058 | 1.603 |
| `vina_score` | `casf2016` | 285 | **0.577** | 0.582 | 2.354 | 1.898 | 1.779 |

#### [`results/gnn_summary.csv`](results/gnn_summary.csv)
Complete comparison across all 5 GNN seeds, statistical distributions, ensemble, and baselines:

| Model | Test Pearson $R$ | Test RMSE | CASF-2016 Pearson $R$ | CASF-2016 RMSE | Status vs. Goals |
|---|---|---|---|---|---|
| **Ligand-Only Baseline** | 0.419 | 1.680 | 0.481 | 1.920 | Negative control (2D chemistry) |
| **AutoDock Vina Baseline** | 0.470 | 2.519 | **0.577** | 2.354 | Classical docking benchmark |
| **GNN Seed 0** | **0.424** | **1.649** | **0.442** | **1.980** | **Beats Ligand-Only on Test** ($> 0.419$) |
| **GNN Seed 1** | 0.257 | 1.815 | 0.205 | 2.231 | Suboptimal local minimum |
| **GNN Seed 2** | 0.175 | 1.827 | 0.190 | 2.255 | Suboptimal local minimum |
| **GNN Seed 3** | 0.113 | 1.828 | 0.164 | 2.259 | Suboptimal local minimum |
| **GNN Seed 4** | **0.427** | **1.669** | 0.317 | 2.072 | **Beats Ligand-Only on Test** ($> 0.419$) |
| **GNN 5-Seed Mean ($\mu \pm \sigma$)** | **0.279 ± 0.143** | **1.757 ± 0.090** | **0.263 ± 0.116** | **2.159 ± 0.126** | Distribution across all 5 seeds |
| **GNN 5-Seed Ensemble** | **0.426** | **1.735** | **0.394** | **2.123** | **Beats Ligand-Only on Test** ($> 0.419$) |

### 5.2 Model Checkpoints (`runs/`)
- **`runs/overfit/`**: Contains `best.pt`, `last.pt`, and `history.json` confirming micro-batch memorization ($RMSE = 0.0788 < 0.1$).
- **`runs/gnn_s0/` to `runs/gnn_s4/`**: Each directory contains:
  - `best.pt`: Checkpoint with lowest validation RMSE.
  - `last.pt`: Resumable checkpoint with model weights, Adam optimizer state, and training history.
  - `metrics.json`: Evaluated metrics on validation, test, and CASF-2016 sets.
  - `history.json`: Epoch-by-epoch loss and validation metrics.
  - `pred_val.csv`, `pred_test.csv`, `pred_casf2016.csv`: Complex-level predictions.

---

## 6. Web Application & Frontend (`web/`)

- **[`web/src/App.tsx`](web/src/App.tsx)**: Main React application interface for entering PDB targets, viewing affinity predictions, and exploring generated molecules.
- **[`web/src/viewer.ts`](web/src/viewer.ts)**: 3D interactive viewer utilizing 3Dmol.js / Mol* for rendering protein pockets, hydrogen bonds, and ligand docking poses.
- **[`web/src/Structure2D.tsx`](web/src/Structure2D.tsx)**: 2D chemical structure rendering using RDKit.js.
- **[`web/src/api.ts`](web/src/api.ts)**: Client API service communicating with the FastAPI backend.
- **[`web/vite.config.ts`](web/vite.config.ts)**: Vite bundler configuration.
- **[`web/package.json`](web/package.json)**: Frontend dependency definitions.

---

## 7. Test Suite (`tests/`)

PocketLig contains 19 automated test cases executed via `pytest`:
- **[`tests/test_data.py`](tests/test_data.py)** (8 tests): Validates affinity unit parsing, operator handling, index parsing, temporal splitting, leakage detection, MMseqs2 cluster reading, and end-to-end `make_splits` execution.
- **[`tests/test_graph_model.py`](tests/test_graph_model.py)** (5 tests): Verifies pocket cleaning, SE(3) Euclidean rotation/translation invariance of graph edge features and model outputs, 20-sample overfit check, and checkpoint resumption.
- **[`tests/test_baselines.py`](tests/test_baselines.py)** (1 test): Verifies that the ligand-only GBR baseline learns linear LogP signals from SMILES.
- **[`tests/test_pipeline_pieces.py`](tests/test_pipeline_pieces.py)** (5 tests): Tests diffusion trajectory sampling, frame selection, and PoseBusters evaluation.

**Verification Status**: All 19 tests passing (`19 passed in 18.51s`).

---

## 8. Root Project Configuration & Documentation Files

- **[`PHASE1_README.md`](PHASE1_README.md)**: Comprehensive record of Phase 1 setup, TargetDiff integration, benchmark ID curation, and zero-leakage split generation.
- **[`PHASE2_README.md`](PHASE2_README.md)**: Comprehensive record of Phase 2 3D graph dataset construction and baseline benchmarking.
- **[`PHASE3A_README.md`](PHASE3A_README.md)**: Record of Phase 3A 20-sample overfit check, 5-seed production GNN runs, full CASF-2016 graph build, and scoring power evaluation.
- **[`docs/DECISIONS.md`](docs/DECISIONS.md)**: Official project decision log recording thresholds, split numbers, and benchmark performance metrics.
- **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)**: Environment setup and dependencies reference.
- **[`docs/STATUS.md`](docs/STATUS.md)**: Milestone and task progress log.
- **[`requirements.txt`](requirements.txt)**: Python package requirements.
- **[`environment.yml`](environment.yml)**: Conda environment specification.
- **[`Makefile`](Makefile)**: Automation commands for tests, dataset building, and training.
- **[`Dockerfile`](Dockerfile)**: Container build specification.
- **[`pytest.ini`](pytest.ini)**: Test suite runner configuration.
- **[`README.md`](README.md)**: Repository introduction and usage guide.
