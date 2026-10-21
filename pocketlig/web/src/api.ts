export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8008';

export interface TargetSummary { id: string; source: string | null; date: string | null; n_molecules: number; has_potential: boolean }
export interface MoleculeRow {
  file: string; valid: boolean; smiles?: string; qed?: number | null; sa?: number | null; n_heavy?: number;
  gnn_pk?: number | null; vina_score_only?: number | null; gnina_cnn_affinity?: number | null;
  posebusters_all_pass?: boolean | null; errors?: Record<string, string>;
}
export interface MoleculesResponse { summary: Record<string, unknown>; molecules: MoleculeRow[] }
export interface Trajectory {
  elements: string[]; frame_elements: string[][]; steps: number[]; final_bonded: boolean; frames: number[][][];
}
export interface MolProps { molecular_weight: number; logp: number; hbd: number; hba: number; rotatable_bonds: number; tpsa: number; lipinski_violations: string[] }
export interface Contact { type: string; ligand_atom: string; residue: string; protein_atom: string; distance: number }
export interface PocketInfo { residues: number; atoms: number; centroid: number[] | null; reference_ligand: string | null; cutoff_A: number | null }
export interface Manifest {
  target_id: string; source?: string; date: string; versions: Record<string, string | null>;
  generator?: { checkpoint?: string; checkpoint_sha256?: string; num_steps?: number; seed?: number };
  electrostatics?: { force_field?: string; ph?: number };
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const errText = await r.text().catch(() => '');
    throw new Error(`${path}: ${r.status} ${errText || r.statusText}`);
  }
  return (await r.json()) as T;
}

export interface GenerateRequest {
  num_molecules: number;
  diffusion_steps: number;
  temperature?: number;
  seed?: number;
  device?: string;
}

export interface JobStatus {
  id: string;
  target_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  current_step?: number | null;
  total_steps?: number | null;
  molecules_generated?: string[];
  error?: string | null;
  created_at: string;
}

export const triggerGeneration = (t: string, req: GenerateRequest) =>
  postJson<{ job_id: string; status: string; target_id: string }>(`/targets/${t}/generate`, req);

export const getJobStatus = (jobId: string) =>
  getJson<JobStatus>(`/jobs/${jobId}`);
export const listTargets = () => getJson<TargetSummary[]>('/targets');
export const getManifest = (t: string) => getJson<Manifest>(`/targets/${t}`);
export const getMolecules = (t: string) => getJson<MoleculesResponse>(`/targets/${t}/molecules`);
export const getTrajectory = (t: string, mol: string) => getJson<Trajectory>(`/targets/${t}/trajectory/${mol}`);
export const fileUrl = (t: string, name: string) => `${API_BASE}/targets/${t}/files/${name}`;
export const finalSdfUrl = (t: string, mol: string) => `${API_BASE}/targets/${t}/molecule/${mol}/final.sdf`;

/** 'artifacts/t/trajectories/mol_12/final.sdf' -> 'mol_12' */
export function molIdFromFile(file: string): string | null {
  const m = file.match(/(mol_\d+)/);
  return m ? m[1] : null;
}

export const getProps = (t: string, mol: string) => getJson<MolProps>(`/targets/${t}/molecule/${mol}/properties`);
export const getContacts = (t: string, mol: string) => getJson<{ method: string; contacts: Contact[] }>(`/targets/${t}/molecule/${mol}/contacts`);
export const getPocket = (t: string) => getJson<PocketInfo>(`/targets/${t}/pocket`);
