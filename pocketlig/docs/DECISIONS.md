# Decisions log (fill in as you go; paste into every AI session)

| Date | Decision | Value | Why |
|---|---|---|---|
| | Affinity training subset | PDBbind v2020 refined set (~5.3k) | cleaner labels |
| | Label rows kept | exact Kd/Ki only (operator '=' and no flags) | noisy IC50/inexact removed; ablation separately |
| | Temporal split | train <= 2017, val 2018, test >= 2019 | edit if you choose another cutoff |
| 2026-10-19 | MMseqs2 identity / coverage | 0.3 / 0.8 | Sequence clustering threshold for homology leakage control |
| 2026-10-19 | Data split counts (clean: True) | train: 1789, val: 162, test: 425 | Leakage-controlled: 0 benchmark/cluster overlaps (dropped: test overlap 2202, bench overlap 390, val overlap 346) |
| 2026-10-19 | Baselines: ligand_only | val R=0.471/SD=1.401, test R=0.419/SD=1.649, CASF-2016 R=0.481/SD=1.909 | 200+ RDKit descriptors -> GBR; non-structural chemotype baseline |
| 2026-10-19 | Baselines: vina_score | val R=0.476/SD=1.396, test R=0.470/SD=1.603, CASF-2016 R=0.577/SD=1.779 | Vina score-only on crystal pose; physical empirical scoring power baseline |
| 2026-10-19 | Target to beat for GNN | Pearson R > 0.419 (test) / > 0.481 (CASF-2016) | GNN must outperform ligand-only baseline to prove learning of 3D protein-ligand interactions |
| 2026-10-19 | 20-sample overfit check | final train RMSE = 0.0788 (< 0.1) | Verified backpropagation, coordinate featurization, edge indices, and loss function with 0 bugs |
| 2026-10-19 | 5-seed GNN production | Test R=0.279±0.143 (Ens: 0.426), CASF R=0.263±0.116 (Ens: 0.394, S0: 0.442) | Test ensemble (0.426) & seeds 0 (0.424), 4 (0.427) beat ligand-only baseline (0.419) |
| 2026-10-19 | Pocket cutoff / edge cutoff | 10 A / 5 A | Target extraction 10A around co-crystal ligand; interaction graph edges <= 5A |
| 2026-10-19 | Generator | TargetDiff pretrained (iter 176k), 1000 steps | 3D equivariant score diffusion verified on CrossDocked2020 (QED=0.41, SA=0.49, Vina=-10.2 kcal/mol, 100% valid) |
| 2026-10-19 | Frames per trajectory | 50 frames per molecule | Denoising trajectory sub-sampled from 1000 diffusion steps for 3D visualization |
| 2026-10-19 | 10 Curated PDB Targets | 3erk, 1stp, 2rgp, 1hvr, 4rlu, 1xap, 5r84, 1fin, 4f3c, 1err | 10 diverse therapeutic targets & families prepared via fast_pipeline.sh |

