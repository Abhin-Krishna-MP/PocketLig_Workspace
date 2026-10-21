import type { TargetSummary, MoleculeRow, PocketInfo, MolProps, Contact, Trajectory, Manifest } from './api';

export const MOCK_TARGETS: TargetSummary[] = [
  { id: '1hvr', source: 'RCSB PDB 1HVR', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '5r84', source: 'SARS-CoV-2 Mpro', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '3erk', source: 'MAP Kinase ERK2', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '1stp', source: 'Streptavidin', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '2rgp', source: 'B-Raf Kinase', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '4rlu', source: 'PARP1', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '1xap', source: 'Carbonic Anhydrase II', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '1fin', source: 'CDK2 / ATP', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '4f3c', source: 'CDK2 Inhibitor', date: '2026-10-19', n_molecules: 10, has_potential: true },
  { id: '1err', source: 'Estrogen Receptor α', date: '2026-10-19', n_molecules: 10, has_potential: true }
];

export const MOCK_POCKET: PocketInfo = {
  residues: 86,
  atoms: 248,
  centroid: [12.4, 21.8, 8.5],
  reference_ligand: 'XK2',
  cutoff_A: 10.0
};

export const MOCK_MANIFEST: Manifest = {
  target_id: '1hvr',
  source: 'RCSB PDB 1HVR',
  date: '2026-10-19',
  versions: { python: '3.11.17', torch: '2.5.1', rdkit: '2026.09.1' },
  generator: { checkpoint: 'pretrained_diffusion.pt', num_steps: 50, seed: 2021 },
  electrostatics: { force_field: 'AMBER', ph: 7.0 }
};

export const MOCK_MOLECULES: MoleculeRow[] = [
  {
    file: 'trajectories/mol_0/final.sdf',
    valid: true,
    smiles: 'COC(=O)c1ccc2c(c1)N(CC(C)C)CC[C@@H](O)[C@@H]2[C@H](O)[C@@H](CCC1=CC(C)C=CC1)c1ccc(C)c(C)c1',
    qed: 0.78,
    sa: 2.85,
    n_heavy: 32,
    gnn_pk: 8.52,
    vina_score_only: -9.4,
    gnina_cnn_affinity: 8.91,
    posebusters_all_pass: true,
    errors: {}
  },
  {
    file: 'trajectories/mol_1/final.sdf',
    valid: true,
    smiles: 'CC(C)c1ccc(CC(=O)N[C@H](C(=O)NCc2ccccc2)C(C)(C)C)cc1',
    qed: 0.84,
    sa: 2.12,
    n_heavy: 28,
    gnn_pk: 8.15,
    vina_score_only: -8.8,
    gnina_cnn_affinity: 8.35,
    posebusters_all_pass: true,
    errors: {}
  },
  {
    file: 'trajectories/mol_2/final.sdf',
    valid: true,
    smiles: 'O=C(c1ccc(F)cc1)N1CCN(c2ncc(C(F)(F)F)cn2)CC1',
    qed: 0.91,
    sa: 1.95,
    n_heavy: 25,
    gnn_pk: 7.92,
    vina_score_only: -8.2,
    gnina_cnn_affinity: 7.85,
    posebusters_all_pass: true,
    errors: {}
  }
];

export const MOCK_PROPS: MolProps = {
  molecular_weight: 428.53,
  logp: 3.42,
  hbd: 2,
  hba: 5,
  rotatable_bonds: 6,
  tpsa: 68.4,
  lipinski_violations: []
};

export const MOCK_CONTACTS: Contact[] = [
  { type: 'hydrogen_bond', ligand_atom: 'O1', residue: 'ASP25.A', protein_atom: 'OD2', distance: 2.74 },
  { type: 'hydrogen_bond', ligand_atom: 'O2', residue: 'ASP25.B', protein_atom: 'OD1', distance: 2.81 },
  { type: 'hydrophobic', ligand_atom: 'C12', residue: 'VAL82.A', protein_atom: 'CG1', distance: 3.65 },
  { type: 'hydrophobic', ligand_atom: 'C18', residue: 'ILE50.B', protein_atom: 'CD1', distance: 3.82 },
  { type: 'pi_stacking', ligand_atom: 'C4', residue: 'PHE53.A', protein_atom: 'CZ', distance: 3.91 }
];

export const MOCK_TRAJECTORY: Trajectory = {
  elements: ['C', 'C', 'N', 'O', 'C', 'O'],
  frame_elements: [['C', 'C', 'N', 'O', 'C', 'O']],
  steps: [0, 10, 20, 30, 40, 50],
  final_bonded: true,
  frames: [
    [[0.0, 0.0, 0.0], [1.2, 0.3, 0.1], [2.1, -0.4, 0.2], [3.2, 0.2, -0.1], [1.5, 1.8, 0.0], [0.8, 2.5, -0.3]],
    [[0.1, 0.0, -0.1], [1.2, 0.3, 0.0], [2.2, -0.5, 0.1], [3.3, 0.1, -0.2], [1.4, 1.7, 0.1], [0.9, 2.6, -0.2]]
  ]
};
