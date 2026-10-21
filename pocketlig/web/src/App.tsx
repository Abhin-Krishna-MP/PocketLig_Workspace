import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  API_BASE, fileUrl, finalSdfUrl, getContacts, getJobStatus, getManifest, getMolecules, getPocket,
  getProps, getTrajectory, listTargets, molIdFromFile, triggerGeneration,
  type Contact, type Manifest, type MolProps, type MoleculeRow,
  type PocketInfo, type TargetSummary, type Trajectory,
} from './api';
import { NebulaViewer } from './nebula';
import { Structure2D } from './Structure2D';
import { Conformer } from './Conformer';
import {
  IcTarget, IcBox, IcLayers, IcPlay, IcPause, IcAtom, IcDownload, IcGear,
  IcSearch, IcChevronDown, IcChevronUp, IcChevronLeft, IcChevronRight, IcLink,
  IcCube, IcMolecule, IcEye, IcZoomIn, IcZoomOut, IcRefresh, IcMaximize,
  IcArrowRight, IcCheck, IcSun, IcUser, IcActivity, IcDatabase, IcBook,
  IcTrendDown, IcFilter, IcClose, IcUpload, IcEdit, IcInfo, IcZap, IcWave,
  IcBarChart, IcFlask,
  IcMoon, IcSliders, IcTable, IcBookOpen, IcMinimize,
  IcFloppyDisk, IcExternalLink, IcGenerate, IcCopy,
} from './Icons';
/* ═══════════════════════════════════════════════════════════
   GENERATE VIEW COMPONENT (REAL GPU DIFFUSION)
   ═══════════════════════════════════════════════════════════ */
interface GenCandidate { id: string; smiles: string; qed: number; pkd: number; }
function GenerateView({ targets, initialTarget, onInspect, onGenerated }: {
  targets: TargetSummary[]; initialTarget: string | null;
  onInspect: (targetId: string, molId?: string) => void;
  onGenerated?: (targetId: string) => void;
}) {
  const [genTarget, setGenTarget] = useState(initialTarget ?? targets[0]?.id ?? '');
  const [numMols,   setNumMols]   = useState(1);
  const [steps,     setSteps]     = useState(100);
  const [temp,      setTemp]      = useState(1.0);
  const [seed,      setSeed]      = useState(42);
  const [running,   setRunning]   = useState(false);
  const [progress,  setProgress]  = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [genCands,  setGenCands]  = useState<GenCandidate[]>([]);
  const [showAdv,   setShowAdv]   = useState(false);
  const [solver,    setSolver]    = useState<'euler'|'rk4'>('euler');
  const [guidance,  setGuidance]  = useState(1.0);
  const [ffIter,    setFfIter]    = useState(200);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  async function startGeneration() {
    setRunning(true);
    setProgress(2);
    setGenCands([]);
    setProgressLabel(`Submitting diffusion job for ${genTarget.toUpperCase()} to GPU…`);

    try {
      const res = await triggerGeneration(genTarget, {
        num_molecules: numMols,
        diffusion_steps: steps,
        temperature: temp,
        seed: seed,
        device: 'cuda:0',
      });

      const jobId = res.job_id;
      setProgressLabel(`Job #${jobId} launched on NVIDIA RTX 2050 — initializing SE(3) diffusion model…`);

      // Poll real-time status from backend
      timerRef.current = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress);

          if (status.status === 'running') {
            if (status.progress < 80) {
              const cur = status.current_step ?? 0;
              const tot = status.total_steps ?? steps;
              setProgressLabel(`Denoising step ${cur} / ${tot} on GPU (RTX 2050)…`);
            } else if (status.progress < 90) {
              setProgressLabel(`Reconstructing 3D bonds & writing SDF trajectories…`);
            } else {
              setProgressLabel(`Evaluating GNN affinity, QED, and PoseBusters…`);
            }
          } else if (status.status === 'completed') {
            if (timerRef.current) clearInterval(timerRef.current);
            setRunning(false);
            setProgress(100);
            setProgressLabel(`✓ Real diffusion complete! Generated ${status.molecules_generated?.length || numMols} molecule(s) on GPU.`);

            // Fetch newly evaluated molecules from backend
            const molsResp = await getMolecules(genTarget);
            const genSet = new Set(status.molecules_generated || []);
            const newRows = molsResp.molecules.filter(m => {
              const mid = molIdFromFile(m.file);
              return mid && (genSet.size === 0 || genSet.has(mid));
            });

            const displayRows = newRows.length ? newRows.slice(-numMols) : molsResp.molecules.slice(-numMols);
            setGenCands(displayRows.map((r, i) => ({
              id: molIdFromFile(r.file) || `mol_${i}`,
              smiles: r.smiles || '',
              qed: +(r.qed ?? 0).toFixed(2),
              pkd: +(r.gnn_pk ?? 0).toFixed(2),
            })));
            onGenerated?.(genTarget);
          } else if (status.status === 'failed') {
            if (timerRef.current) clearInterval(timerRef.current);
            setRunning(false);
            setProgressLabel(`Generation failed: ${status.error || 'Check server logs.'}`);
          }
        } catch {
          // keep polling
        }
      }, 750);
    } catch (err: any) {
      setRunning(false);
      setProgressLabel(`Failed to start: ${err.message}`);
    }
  }

  function stopGeneration() {
    if (timerRef.current) clearInterval(timerRef.current);
    setRunning(false);
    setProgressLabel('Stopped.');
  }

  return (
    <div className="fullview">
      <div className="fullview-hdr">
        <div>
          <div className="fullview-title">
            <span className="fullview-title-icon"><IcFlask size={18} /></span>
            Generate Ligands
          </div>
          <div className="fullview-sub">Run SE(3) diffusion sampling to generate new drug candidates</div>
        </div>
      </div>

      <div className="generate-view">
        {/* ─ Left: Config form ─ */}
        <div className="generate-form">
          <div className="form-group">
            <label className="form-label">Target Protein</label>
            <select className="form-select" value={genTarget} onChange={e => setGenTarget(e.target.value)}>
              {targets.map(t => {
                const m = getMeta(t.id, t.source);
                return <option key={t.id} value={t.id}>{m.name} ({t.id.toUpperCase()})</option>;
              })}
            </select>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Number of Ligands</label>
              <input className="form-input" type="number" min={1} max={20} value={numMols}
                onChange={e => setNumMols(Number(e.target.value))} />
              <span className="form-hint">1 – 20 molecules</span>
            </div>
            <div className="form-group">
              <label className="form-label">Diffusion Steps</label>
              <input className="form-input" type="number" min={50} max={1000} step={50} value={steps}
                onChange={e => setSteps(Number(e.target.value))} />
              <span className="form-hint">50 – 1000 steps</span>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Temperature / Noise Scale: {temp.toFixed(2)}</label>
              <input className="form-range" type="range" min={0.1} max={1.5} step={0.05} value={temp}
                onChange={e => setTemp(Number(e.target.value))} />
              <span className="form-hint">Lower = more conservative; Higher = more diverse</span>
            </div>
            <div className="form-group">
              <label className="form-label">Random Seed</label>
              <input className="form-input" type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} />
            </div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <button className="btn-secondary" style={{ fontSize: 12, padding: '7px 14px' }}
              onClick={() => setShowAdv(a => !a)}>
              <IcSliders size={12} style={{ marginRight: 5 }} />
              {showAdv ? 'Hide' : 'Show'} Advanced Settings
            </button>
          </div>

          {showAdv && (
            <div style={{ background: 'var(--panel2)', border: '1px solid var(--border)', borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div className="form-group">
                <label className="form-label">ODE Solver</label>
                <div className="adv-radio-group">
                  {(['euler', 'rk4'] as const).map(s => (
                    <label key={s} className={`adv-radio-opt${solver === s ? ' selected' : ''}`}>
                      <input type="radio" value={s} checked={solver === s} onChange={() => setSolver(s)} />
                      {s === 'euler' ? 'Euler-Maruyama (fast)' : 'Runge-Kutta 4th Order (accurate)'}
                    </label>
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Guidance Scale: {guidance.toFixed(2)}</label>
                <input className="form-range" type="range" min={0} max={5} step={0.1} value={guidance}
                  onChange={e => setGuidance(Number(e.target.value))} />
              </div>
              <div className="form-group">
                <label className="form-label">Force-Field Relaxation Iterations: {ffIter}</label>
                <input className="form-range" type="range" min={0} max={500} step={50} value={ffIter}
                  onChange={e => setFfIter(Number(e.target.value))} />
                <span className="form-hint">MMFF94 geometry optimization iterations (0 = disabled)</span>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn-primary" onClick={running ? stopGeneration : startGeneration}>
              {running ? <><span className="gen-spinner" /> Stop</> : <><IcGenerate size={13} /> Start Diffusion Generation</>}
            </button>
          </div>
        </div>

        {/* ─ Right: Live preview panel ─ */}
        <div className="generate-preview">
          <div className="gen-preview-title">Generation Progress</div>
          <div>
            <div className="progress-wrap">
              <div className="progress-bar" style={{ width: `${progress}%` }} />
            </div>
            <div className="progress-label">
              {progressLabel || 'Configure settings and start.'} {running && <strong>{progress}%</strong>}
            </div>
          </div>

          {genCands.length > 0 && (
            <>
              <div className="gen-preview-title" style={{ marginTop: 4 }}>Candidates ({genCands.length})</div>
              <div className="gen-cand-list">
                {genCands.map((c, i) => (
                  <div key={c.id} className="gen-cand-row">
                    <span className="cand-badge">#{i + 1}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 10, color: 'var(--mut)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {c.smiles.slice(0, 30)}…
                      </div>
                      <div style={{ fontSize: 11, marginTop: 2 }}>
                        pKd <strong style={{ color: 'var(--gold)' }}>{c.pkd}</strong>
                        &nbsp;· QED <strong style={{ color: 'var(--gn)' }}>{c.qed}</strong>
                      </div>
                    </div>
                    <button className="inspect-btn" style={{ fontSize: 10, padding: '3px 7px', background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 5, cursor: 'pointer', color: 'var(--ink2)' }}
                      onClick={() => onInspect(genTarget, c.id)}>
                      Inspect in 3D
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ANALYZE VIEW COMPONENT
   ═══════════════════════════════════════════════════════════ */
type SortKey = 'gnn_pk' | 'qed' | 'sa' | 'vina_score_only';
function AnalyzeView({ rows, target, onInspect }: {
  rows: MoleculeRow[]; target: string | null;
  onInspect: (idx: number) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>('gnn_pk');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const av = a[sortKey] ?? -Infinity;
      const bv = b[sortKey] ?? -Infinity;
      return sortDir === 'desc' ? (bv as number) - (av as number) : (av as number) - (bv as number);
    });
  }, [rows, sortKey, sortDir]);

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortKey(k); setSortDir('desc'); }
  }

  function exportCSV() {
    const header = 'Candidate,SMILES,pKd,QED,SA,Docking Score,Lipinski Violations';
    const body = sorted.map((r, i) => [
      i + 1, `"${r.smiles ?? ''}"`, r.gnn_pk ?? '', r.qed ?? '', r.sa ?? '',
      r.vina_score_only ?? '', r.errors ? '' : 0,
    ].join(',')).join('\n');
    const blob = new Blob([header + '\n' + body], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${target ?? 'target'}_candidates.csv`;
    a.click();
  }

  const SortArrow = ({ k }: { k: SortKey }) => sortKey === k
    ? <span className="sort-arrow">{sortDir === 'desc' ? '↓' : '↑'}</span>
    : <span className="sort-arrow" style={{ opacity: 0.2 }}>↕</span>;

  return (
    <div className="fullview">
      <div className="fullview-hdr">
        <div>
          <div className="fullview-title">
            <span className="fullview-title-icon"><IcTable size={18} /></span>
            Candidate Analysis
            {target && <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--mut)', marginLeft: 8 }}>— {target.toUpperCase()}</span>}
          </div>
          <div className="fullview-sub">{rows.length} valid candidates · Sortable comparison table</div>
        </div>
        <div className="fullview-actions">
          <button className="btn-secondary" style={{ fontSize: 12, padding: '7px 14px' }} onClick={exportCSV}>
            <IcDownload size={12} style={{ marginRight: 5 }} /> Export CSV
          </button>
        </div>
      </div>

      {rows.length === 0 ? (
        <p style={{ color: 'var(--mut)', fontSize: 13 }}>Select a target on the Explore tab first to load candidates.</p>
      ) : (
        <div className="analyze-table-wrap">
          <table className="analyze-table">
            <thead>
              <tr>
                <th>#</th>
                <th>2D Structure</th>
                <th className={sortKey === 'gnn_pk' ? 'sorted' : ''} onClick={() => toggleSort('gnn_pk')}>
                  Predicted pKd <SortArrow k="gnn_pk" />
                </th>
                <th className={sortKey === 'qed' ? 'sorted' : ''} onClick={() => toggleSort('qed')}>
                  QED <SortArrow k="qed" />
                </th>
                <th className={sortKey === 'sa' ? 'sorted' : ''} onClick={() => toggleSort('sa')}>
                  SA Score <SortArrow k="sa" />
                </th>
                <th className={sortKey === 'vina_score_only' ? 'sorted' : ''} onClick={() => toggleSort('vina_score_only')}>
                  Docking (Vina) <SortArrow k="vina_score_only" />
                </th>
                <th>Lipinski</th>
                <th>Heavy Atoms</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const molIdx = rows.indexOf(r);
                return (
                  <tr key={i}>
                    <td><span className="mol-num">#{i + 1}</span></td>
                    <td>
                      {r.smiles
                        ? <div style={{ width: 80, height: 52, background: 'var(--panel2)', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, color: 'var(--mut)', padding: 2 }}>
                            {r.smiles.slice(0, 20)}…
                          </div>
                        : <span style={{ color: 'var(--mut)', fontSize: 11 }}>–</span>
                      }
                    </td>
                    <td style={{ color: 'var(--gold)', fontWeight: 700 }}>{f(r.gnn_pk)}</td>
                    <td style={{ color: 'var(--gn)', fontWeight: 600 }}>{f(r.qed)}</td>
                    <td style={{ color: 'var(--or)' }}>{f(r.sa)}</td>
                    <td style={{ color: 'var(--bl)' }}>{r.vina_score_only != null ? `${f(r.vina_score_only, 1)} kcal/mol` : '–'}</td>
                    <td>
                      <span className={r.errors && Object.keys(r.errors).length === 0 ? 'val-pass' : 'val-fail'}>
                        {r.errors && Object.keys(r.errors).length === 0 ? '✓ Pass' : '✗ Fail'}
                      </span>
                    </td>
                    <td>{r.n_heavy ?? '–'}</td>
                    <td>
                      <button className="inspect-btn" onClick={() => onInspect(molIdx)}>
                        Inspect in 3D
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   DATASET VIEW COMPONENT
   ═══════════════════════════════════════════════════════════ */
const DATASET_META: Record<string, { organism: string; resolution: string; class_: string }> = {
  '1hvr': { organism: 'HIV-1', resolution: '2.00 Å', class_: 'Protease' },
  '5r84': { organism: 'SARS-CoV-2', resolution: '1.27 Å', class_: 'Cysteine Protease' },
  '3erk': { organism: 'H. sapiens', resolution: '2.40 Å', class_: 'Kinase' },
  '1fin': { organism: 'H. sapiens', resolution: '2.05 Å', class_: 'Cyclin-dependent Kinase' },
  '1err': { organism: 'H. sapiens', resolution: '3.10 Å', class_: 'Nuclear Receptor' },
  '2rgp': { organism: 'H. sapiens', resolution: '3.50 Å', class_: 'Serine/Threonine Kinase' },
  '4rlu': { organism: 'H. sapiens', resolution: '2.30 Å', class_: 'Transferase' },
  '1xap': { organism: 'H. sapiens', resolution: '1.50 Å', class_: 'Lyase' },
  '4f3c': { organism: 'H. sapiens', resolution: '1.95 Å', class_: 'Kinase' },
  '1stp': { organism: 'S. avidinii', resolution: '2.60 Å', class_: 'Binding Protein' },
};

function DatasetView({ targets, onLoad }: { targets: TargetSummary[]; onLoad: (id: string) => void }) {
  return (
    <div className="fullview">
      <div className="fullview-hdr">
        <div>
          <div className="fullview-title">
            <span className="fullview-title-icon"><IcDatabase size={18} /></span>
            Target Library
          </div>
          <div className="fullview-sub">{targets.length} curated benchmark targets from RCSB PDB</div>
        </div>
      </div>
      <div className="dataset-grid">
        {targets.map(t => {
          const m = getMeta(t.id, t.source);
          const dm = DATASET_META[t.id.toLowerCase()] ?? { organism: '–', resolution: '–', class_: '–' };
          return (
            <div key={t.id} className="dataset-card">
              <div className="dataset-card-pdb">{t.id.toUpperCase()}</div>
              <div className="dataset-card-name">{m.name}</div>
              <div className="dataset-card-sub">{dm.organism} · {dm.class_}</div>
              <div className="dataset-card-kv">
                <div className="dataset-kv-item">
                  <div className="dataset-kv-label">Resolution</div>
                  <div className="dataset-kv-val">{dm.resolution}</div>
                </div>
                <div className="dataset-kv-item">
                  <div className="dataset-kv-label">Candidates</div>
                  <div className="dataset-kv-val">{t.n_molecules}</div>
                </div>
              </div>
              <div className="dataset-card-actions">
                <button className="dataset-load-btn" onClick={() => onLoad(t.id)}>
                  <IcArrowRight size={11} /> Load into Studio
                </button>
                <a className="dataset-dl-btn" href={fileUrl(t.id, 'pocket.pdb')} download>
                  <IcDownload size={10} /> Pocket
                </a>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   DOCS VIEW COMPONENT
   ═══════════════════════════════════════════════════════════ */
function DocsView() {
  const [section, setSection] = useState('model');
  const sections = [
    { id: 'model', label: 'Model Architecture' },
    { id: 'electrostatics', label: 'Electrostatics (APBS)' },
    { id: 'scoring', label: 'Scoring Pipeline' },
    { id: 'api', label: 'REST API Reference' },
  ];
  return (
    <div className="fullview">
      <div className="fullview-hdr">
        <div>
          <div className="fullview-title">
            <span className="fullview-title-icon"><IcBookOpen size={18} /></span>
            Documentation
          </div>
          <div className="fullview-sub">Technical reference for the PocketLab β diffusion pipeline</div>
        </div>
      </div>
      <div className="docs-view">
        <div className="docs-sidebar">
          {sections.map(s => (
            <button key={s.id} className={`docs-nav-item${section === s.id ? ' active' : ''}`} onClick={() => setSection(s.id)}>
              {s.label}
            </button>
          ))}
        </div>
        <div className="docs-content">
          {section === 'model' && <>
            <h2><IcFlask size={16} /> Model Architecture</h2>
            <p>PocketLab β uses an <strong>SE(3)-equivariant continuous diffusion model</strong> on molecular graph point clouds. Atoms are represented as nodes with position <code>x ∈ ℝ³</code> and element type embeddings, with bonds as edges.</p>
            <h3>Forward Process</h3>
            <p>Gaussian noise is gradually added to atom coordinates over <em>T</em> steps: <code>q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, (1−ᾱₜ)I)</code>. Atom types are independently diffused over a categorical distribution.</p>
            <h3>Reverse Denoising</h3>
            <p>A graph neural network <code>ε_θ(xₜ, t, pocket)</code> predicts the noise residual conditioned on the binding pocket geometry. The network uses equivariant message-passing layers (EGNN) with radial basis embeddings and attention.</p>
            <h3>Pocket Conditioning</h3>
            <p>Pocket residue atoms are incorporated as context nodes. Cross-attention between ligand and pocket atoms enables the model to place atoms in chemically plausible binding poses.</p>
          </>}
          {section === 'electrostatics' && <>
            <h2><IcZap size={16} /> Electrostatics (APBS)</h2>
            <p>Electrostatic surface potentials are computed using the <strong>Adaptive Poisson-Boltzmann Solver (APBS)</strong> solving the nonlinear Poisson-Boltzmann equation:</p>
            <pre><code>∇·(ε(r)∇φ(r)) − κ²(r)sinh(φ(r)) = −ρ_f(r)/kT</code></pre>
            <p>Partial charges are assigned from the <strong>AMBER force field</strong> at physiological pH 7.4. The resulting volumetric potential grid (<code>.dx</code> format) is uploaded to the GPU as a 3D texture for real-time raymarching in the WebGL2 viewer.</p>
            <h3>Visualization</h3>
            <p>The electrostatic color map ranges from <strong style={{ color: '#ff5d4d' }}>−5.0 kT/e (negative, red)</strong> through neutral (white) to <strong style={{ color: '#5b86ff' }}>+5.0 kT/e (positive, blue)</strong>.</p>
          </>}
          {section === 'scoring' && <>
            <h2><IcBarChart size={16} /> Scoring Pipeline</h2>
            <h3>GNN Affinity Predictor (pKd)</h3>
            <p>A pretrained Graph Neural Network predicts binding affinity as <code>pKd = −log₁₀(Kd)</code>. Higher values indicate stronger binding (pKd ≥ 8 = nanomolar, high affinity).</p>
            <h3>Drug-Likeness (QED)</h3>
            <p>Quantitative Estimate of Drug-likeness using the Bickerton et al. scoring function. Values 0–1; higher is more drug-like. Computed via RDKit.</p>
            <h3>Synthesizability Score (SA)</h3>
            <p>Ertl-Schuffenhauer Synthetic Accessibility Score (1–10). Lower values indicate easier synthesis. Computed via RDKit.</p>
            <h3>AutoDock Vina / GNINA CNN</h3>
            <p>Cross-docking scores (kcal/mol) from AutoDock Vina and the GNINA CNN pose scoring function. More negative Vina scores = stronger predicted binding.</p>
            <h3>PoseBusters</h3>
            <p>Structural plausibility checks: bond lengths, angles, clashes, chirality. All checks must pass for a candidate to be flagged ✓.</p>
          </>}
          {section === 'api' && <>
            <h2><IcExternalLink size={16} /> REST API Reference</h2>
            <p>The FastAPI backend serves pre-computed artifacts. Base URL: <code>{API_BASE}</code></p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets</span></div>
            <p>Returns list of all computed target summaries.</p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets/{'{tid}'}/molecules</span></div>
            <p>Returns sorted list of all generated molecules with scores for a target.</p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets/{'{tid}'}/trajectory/{'{mol}'}</span></div>
            <p>Returns diffusion trajectory frames (coordinates per step) for a molecule.</p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets/{'{tid}'}/molecule/{'{mol}'}/properties</span></div>
            <p>Returns computed molecular properties: MW, LogP, HBD, HBA, TPSA, Lipinski violations.</p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets/{'{tid}'}/molecule/{'{mol}'}/contacts</span></div>
            <p>Returns detected binding contacts (hydrogen bonds, hydrophobic, π-stacking) within cutoff distances.</p>
            <div className="api-endpoint-row"><span className="api-method get">GET</span><span className="api-path">/targets/{'{tid}'}/molecule/{'{mol}'}/final.sdf</span></div>
            <p>Serves the final sanitized SDF file for download.</p>
          </>}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   UPLOAD MODAL
   ═══════════════════════════════════════════════════════════ */
function UploadModal({ onClose, onAdd }: { onClose: () => void; onAdd: (id: string, name: string) => void }) {
  const [tab, setTab] = useState<'pdb' | 'file'>('pdb');
  const [pdbCode, setPdbCode] = useState('');
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) setFileName(f.name);
  }

  function submit() {
    if (tab === 'pdb' && pdbCode.trim().length !== 4) { setErr('Please enter a valid 4-letter PDB code.'); return; }
    if (tab === 'file' && !fileName) { setErr('Please select a .pdb file.'); return; }
    setLoading(true); setErr('');
    setTimeout(() => {
      const id = tab === 'pdb' ? pdbCode.toLowerCase() : fileName.replace(/\.pdb$/i, '').toLowerCase().slice(0, 8);
      onAdd(id, tab === 'pdb' ? `RCSB PDB ${pdbCode.toUpperCase()}` : fileName);
      onClose();
    }, 1200);
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-card">
        <div className="modal-hdr">
          <div className="modal-title">
            <span className="modal-title-icon"><IcUpload size={14} /></span>
            Upload Target Protein
          </div>
          <button className="modal-close" onClick={onClose}><IcClose size={12} /></button>
        </div>

        <div className="modal-tabs">
          <button className={`modal-tab${tab === 'pdb' ? ' active' : ''}`} onClick={() => setTab('pdb')}>Enter PDB Code</button>
          <button className={`modal-tab${tab === 'file' ? ' active' : ''}`} onClick={() => setTab('file')}>Upload .PDB File</button>
        </div>

        {tab === 'pdb' ? (
          <div className="form-group">
            <label className="form-label">RCSB PDB ID (4 letters)</label>
            <input className="form-input" placeholder="e.g. 7BUY" maxLength={4}
              value={pdbCode} onChange={e => setPdbCode(e.target.value.toUpperCase())}
              style={{ textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 700, fontSize: 16 }} />
            <span className="form-hint">Structure will be fetched from rcsb.org and pocket auto-detected.</span>
          </div>
        ) : (
          <div className="form-group">
            <label className="form-label">PDB File</label>
            <div className="form-upload-area" onClick={() => fileRef.current?.click()}>
              <div className="upload-icon"><IcUpload size={24} /></div>
              <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--ink2)' }}>
                {fileName ? <span className="upload-file-name">{fileName}</span> : 'Drop your .pdb file here or click to browse'}
              </div>
              <div className="upload-btn-text">Supports standard PDB format (ATOM / HETATM records)</div>
              <input ref={fileRef} type="file" accept=".pdb" style={{ display: 'none' }} onChange={handleFile} />
            </div>
          </div>
        )}

        {err && <p style={{ color: 'var(--rd)', fontSize: 11, marginTop: 0, marginBottom: 10 }}>{err}</p>}

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="gen-spinner" /> Fetching…</> : <><IcArrowRight size={13} /> Load Target</>}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   EDIT POCKET MODAL
   ═══════════════════════════════════════════════════════════ */
function EditPocketModal({ pocket, onClose, onSave }: {
  pocket: PocketInfo | null;
  onClose: () => void;
  onSave: (p: { centroid: [number, number, number]; cutoff: number; refLigand: string }) => void;
}) {
  const [cx, setCx] = useState(pocket?.centroid?.[0] ?? 0);
  const [cy, setCy] = useState(pocket?.centroid?.[1] ?? 0);
  const [cz, setCz] = useState(pocket?.centroid?.[2] ?? 0);
  const [cutoff, setCutoff] = useState(pocket?.cutoff_A ?? 10);
  const [refLig, setRefLig] = useState(pocket?.reference_ligand ?? '');

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-card">
        <div className="modal-hdr">
          <div className="modal-title">
            <span className="modal-title-icon"><IcEdit size={14} /></span>
            Edit Binding Pocket
          </div>
          <button className="modal-close" onClick={onClose}><IcClose size={12} /></button>
        </div>

        <div className="form-group">
          <label className="form-label">Pocket Centroid (Å)</label>
          <div className="pocket-coords-grid">
            {([['X', cx, setCx], ['Y', cy, setCy], ['Z', cz, setCz]] as const).map(([axis, val, set]) => (
              <div key={axis} className="pocket-coord-input">
                <label>{axis}</label>
                <input type="number" step="0.1" value={(val as number).toFixed(1)}
                  onChange={e => (set as (v: number) => void)(Number(e.target.value))} />
              </div>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Pocket Cutoff Radius</label>
          <div className="pocket-cutoff-row">
            <input type="range" min={4} max={20} step={0.5} value={cutoff}
              onChange={e => setCutoff(Number(e.target.value))} />
            <span className="pocket-cutoff-val">{cutoff.toFixed(1)} Å</span>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Reference Ligand (3-letter code)</label>
          <input className="form-input" placeholder="e.g. XK2, ATP, DRG" maxLength={3}
            value={refLig} onChange={e => setRefLig(e.target.value.toUpperCase())}
            style={{ textTransform: 'uppercase', letterSpacing: '0.1em' }} />
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => { onSave({ centroid: [cx, cy, cz], cutoff, refLigand: refLig }); onClose(); }}>
            <IcFloppyDisk size={13} /> Apply & Recenter
          </button>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   VIEWPORT SETTINGS POPOVER
   ═══════════════════════════════════════════════════════════ */
function VpSettingsPopover({ onClose }: { onClose: () => void }) {
  const [cut, setCut] = useState(0.05);
  const [density, setDensity] = useState(0.5);
  const [autoSpin, setAutoSpin] = useState(true);
  const [spinSpeed, setSpinSpeed] = useState(0.3);
  return (
    <div className="popover vp-settings-popover">
      <div className="popover-title"><IcGear size={11} style={{ marginRight: 5 }} />3D Viewport Settings</div>
      <div className="popover-section">
        <div className="popover-section-label">Front Cut Plane Depth</div>
        <div className="popover-slider-row">
          <input type="range" min={0} max={0.3} step={0.01} value={cut} onChange={e => setCut(Number(e.target.value))} />
          <span>{cut.toFixed(2)}</span>
        </div>
      </div>
      <div className="popover-section">
        <div className="popover-section-label">Cavity Emission Density</div>
        <div className="popover-slider-row">
          <input type="range" min={0.1} max={1.0} step={0.05} value={density} onChange={e => setDensity(Number(e.target.value))} />
          <span>{density.toFixed(2)}</span>
        </div>
      </div>
      <div className="popover-divider" />
      <div className="popover-section">
        <label className="popover-opt" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={autoSpin} onChange={e => setAutoSpin(e.target.checked)} />
          Auto-Rotation
        </label>
        {autoSpin && (
          <div className="popover-slider-row" style={{ marginTop: 8 }}>
            <span style={{ minWidth: 40, color: 'var(--mut)', fontSize: 10 }}>Speed</span>
            <input type="range" min={0.05} max={1.0} step={0.05} value={spinSpeed} onChange={e => setSpinSpeed(Number(e.target.value))} />
            <span>{spinSpeed.toFixed(2)}</span>
          </div>
        )}
      </div>
      <div className="popover-divider" />
      <button className="popover-opt" style={{ color: 'var(--gold)', fontWeight: 600 }} onClick={onClose}>
        <IcCheck size={11} /> Apply & Close
      </button>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   FILTER POPOVER
   ═══════════════════════════════════════════════════════════ */
function FilterPopover({ sortKey, sortDir, lipinskiOnly, onSortChange, onLipinskiChange, onClose }: {
  sortKey: string; sortDir: string; lipinskiOnly: boolean;
  onSortChange: (k: string, d: string) => void;
  onLipinskiChange: (v: boolean) => void;
  onClose: () => void;
}) {
  const opts = [
    { key: 'gnn_pk',         label: 'Affinity (pKd)' },
    { key: 'qed',            label: 'Drug-likeness (QED)' },
    { key: 'sa',             label: 'Synthesizability (SA)' },
    { key: 'vina_score_only',label: 'Docking Score (Vina)' },
  ];
  return (
    <div className="popover filter-popover">
      <div className="popover-title"><IcFilter size={11} style={{ marginRight: 5 }} />Sort & Filter</div>
      <div className="popover-section">
        <div className="popover-section-label">Sort By</div>
        <div className="popover-opts">
          {opts.map(o => (
            <button key={o.key} className={`popover-opt${sortKey === o.key ? ' active' : ''}`}
              onClick={() => onSortChange(o.key, sortKey === o.key && sortDir === 'desc' ? 'asc' : 'desc')}>
              {o.label}
              {sortKey === o.key && <span style={{ marginLeft: 'auto', fontSize: 10 }}>{sortDir === 'desc' ? '↓' : '↑'}</span>}
            </button>
          ))}
        </div>
      </div>
      <div className="popover-divider" />
      <div className="popover-section">
        <div className="popover-section-label">Filter</div>
        <label className="popover-opt" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={lipinskiOnly} onChange={e => onLipinskiChange(e.target.checked)} />
          Lipinski Pass Only (0 violations)
        </label>
      </div>
      <div className="popover-divider" />
      <button className="popover-opt" style={{ color: 'var(--gold)', fontWeight: 600 }} onClick={onClose}>
        <IcCheck size={11} /> Done
      </button>
    </div>
  );
}
const f = (v: number | null | undefined, d = 2) => (v == null ? '–' : v.toFixed(d));

const EL_COLOR: Record<string, string> = {
  C: '#b9c1d0', N: '#5b86ff', O: '#ff5d4d', S: '#f5c542',
  F: '#7ee0a5', P: '#ff9f43', Cl: '#69d17f', Br: '#c0713a',
};

/* ─── Target metadata ─── */
type TargetMeta = { name: string; badgeLabel: string; badgeCls: string; thumbCls: string };

const TARGET_META: Record<string, TargetMeta> = {
  '5r84': { name: 'SARS-CoV-2 Mpro',      badgeLabel: 'Viral',      badgeCls: 'badge-viral',      thumbCls: 'tth-red' },
  '1hvr': { name: 'HIV-1 Protease',        badgeLabel: 'Viral',      badgeCls: 'badge-viral',      thumbCls: 'tth-red' },
  '1err': { name: 'Estrogen Receptor α',   badgeLabel: 'Oncology',   badgeCls: 'badge-oncology',   thumbCls: 'tth-green' },
  '1fin': { name: 'CDK2 / ATP',            badgeLabel: 'Kinase',     badgeCls: 'badge-kinase',     thumbCls: 'tth-blue' },
  '1stp': { name: 'Streptavidin',          badgeLabel: 'PDB',        badgeCls: 'badge-pdb',        thumbCls: 'tth-gold' },
  '1xap': { name: 'Carbonic Anhydrase II', badgeLabel: 'PDB',        badgeCls: 'badge-pdb',        thumbCls: 'tth-gold' },
  '2rgp': { name: 'B-Raf Kinase',          badgeLabel: 'Kinase',     badgeCls: 'badge-kinase',     thumbCls: 'tth-blue' },
  '3erk': { name: 'MAP Kinase ERK2',       badgeLabel: 'Oncology',   badgeCls: 'badge-oncology',   thumbCls: 'tth-green' },
  '4f3c': { name: 'CDK2 Inhibitor',        badgeLabel: 'Kinase',     badgeCls: 'badge-kinase',     thumbCls: 'tth-blue' },
  '6lu7': { name: 'SARS-CoV-2 Mpro',      badgeLabel: 'Viral',      badgeCls: 'badge-viral',      thumbCls: 'tth-red' },
  '1h1q': { name: 'CDK2 Kinase',           badgeLabel: 'Kinase',     badgeCls: 'badge-kinase',     thumbCls: 'tth-purple' },
  '4bjx': { name: 'BRD4 (Bromodomain)',    badgeLabel: 'Epigenetic', badgeCls: 'badge-epigenetic', thumbCls: 'tth-gold' },
  '1m17': { name: 'EGFR Kinase',           badgeLabel: 'Oncology',   badgeCls: 'badge-oncology',   thumbCls: 'tth-green' },
  '1fkn': { name: 'BACE1',                 badgeLabel: "Alzheimer's", badgeCls: 'badge-alz',       thumbCls: 'tth-blue' },
  '4rlu': { name: 'PARP1',                 badgeLabel: 'Oncology',   badgeCls: 'badge-oncology',   thumbCls: 'tth-green' },
};
function getMeta(id: string, source?: string | null): TargetMeta {
  return TARGET_META[id.toLowerCase()] ?? {
    name: source?.replace('RCSB PDB ', '') ?? id.toUpperCase(),
    badgeLabel: 'PDB', badgeCls: 'badge-pdb', thumbCls: 'tth-gold',
  };
}

/* ─── Thumb icon SVG per category ─── */
function TargetThumbIcon({ cls }: { cls: string }) {
  if (cls === 'tth-red')    return <IcAtom size={15} />;
  if (cls === 'tth-blue')   return <IcActivity size={15} />;
  if (cls === 'tth-purple') return <IcDatabase size={15} />;
  if (cls === 'tth-green')  return <IcBarChart size={15} />;
  if (cls === 'tth-orange') return <IcInfo size={15} />;
  return <IcMolecule size={15} />;
}

/* ─── Affinity quality ─── */
function affQual(pk: number | null | undefined) {
  if (pk == null) return { label: '', cls: '' };
  if (pk >= 8) return { label: 'High',   cls: 'aff-hi' };
  if (pk >= 6) return { label: 'Medium', cls: 'aff-med' };
  return { label: 'Low', cls: 'aff-lo' };
}

/* ─── Interaction dot class ─── */
function intDot(type: string) {
  const t = type.toLowerCase();
  if (t.includes('hydrogen')) return 'int-dot dot-hbond';
  if (t.includes('hydrophob')) return 'int-dot dot-hydro';
  if (t.includes('pi') || t.includes('stack')) return 'int-dot dot-pistack';
  return 'int-dot dot-hydro';
}

function intLabel(type: string) {
  const t = type.toLowerCase();
  if (t.includes('pi') || t.includes('stack')) return 'π-π Stacking';
  return type.replace(/_/g, ' ').replace(/\b\w/g, m => m.toUpperCase());
}

/* ─── Toggle component ─── */
function Toggle({ on, onChange, tone = '' }: { on: boolean; onChange: (v: boolean) => void; tone?: string }) {
  return (
    <span className="toggle">
      <input type="checkbox" checked={on} onChange={e => onChange(e.target.checked)} />
      <span className={`toggle-track${tone ? ` t-${tone}` : ''}`} />
      <span className="toggle-thumb" />
    </span>
  );
}

/* ─── Trajectory thumb ─── */
function Thumb({ t, i, scale }: { t: Trajectory; i: number; scale: number }) {
  const frame = t.frames[i];
  const last  = t.frames[t.frames.length - 1];
  const cx = last.reduce((s, p) => s + p[0], 0) / last.length;
  const cy = last.reduce((s, p) => s + p[1], 0) / last.length;
  return (
    <svg viewBox="-50 -38 100 76" aria-hidden="true">
      {frame.map((p, k) => (
        <circle key={k}
          cx={(p[0] - cx) * scale} cy={(p[1] - cy) * scale}
          r={2.5} fill={EL_COLOR[t.frame_elements[i]?.[k]] ?? '#c084fc'} opacity={0.88} />
      ))}
    </svg>
  );
}

/* ══════════════════════════════════════════════════════════
   MAIN APP
   ══════════════════════════════════════════════════════════ */
export function App() {
  /* ─── Core state ─── */
  const [targets,   setTargets]   = useState<TargetSummary[]>([]);
  const [searchQ,   setSearchQ]   = useState('');
  const [target,    setTarget]    = useState<string | null>(null);
  const [manifest,  setManifest]  = useState<Manifest | null>(null);
  const [pocket,    setPocket]    = useState<PocketInfo | null>(null);
  const [summary,   setSummary]   = useState<Record<string, any>>({});
  const [rows,      setRows]      = useState<MoleculeRow[]>([]);
  const [idx,       setIdx]       = useState(0);
  const [traj,      setTraj]      = useState<Trajectory | null>(null);
  const [frame,     setFrame]     = useState(0);
  const [playing,   setPlaying]   = useState(false);
  const [speed,     setSpeed]     = useState(1);
  const [props,     setProps]     = useState<MolProps | null>(null);
  const [contacts,  setContacts]  = useState<{ method: string; contacts: Contact[] } | null>(null);
  const [layers, setLayers] = useState({
    surface: true, potential: true, pocket: true, ligand: true,
    atoms: true, hbonds: true, trajectory: true,
  });
  const [layersOpen, setLayersOpen] = useState(true);
  const [pocketTab,  setPocketTab]  = useState<'auto' | 'custom'>('auto');
  const [error,  setError]  = useState<string | null>(null);
  const [busy,   setBusy]   = useState('Loading targets…');

  /* ─── New UI state ─── */
  const [activeTab,     setActiveTab]     = useState<'explore'|'generate'|'analyze'|'dataset'|'docs'>('explore');
  const [theme,         setTheme]         = useState<'dark'|'light'>('dark');
  const [isFullscreen,  setIsFullscreen]  = useState(false);
  const [showFilter,    setShowFilter]    = useState(false);
  const [showVpSettings,setShowVpSettings]= useState(false);
  const [sortKey,       setSortKey]       = useState('gnn_pk');
  const [sortDir,       setSortDir]       = useState<'desc'|'asc'>('desc');
  const [lipinskiOnly,  setLipinskiOnly]  = useState(false);
  const [measureMode,   setMeasureMode]   = useState(false);
  const [showUpload,    setShowUpload]    = useState(false);
  const [showEditPocket,setShowEditPocket]= useState(false);
  const [srcTab,        setSrcTab]        = useState<'examples'|'upload'|'pdbbind'>('examples');
  const [customCentroid,setCustomCentroid]= useState<[number,number,number] | null>(null);
  const [customCutoff,  setCustomCutoff]  = useState<number | null>(null);
  const [customRefLig,  setCustomRefLig]  = useState<string>('');

  const canvasRef    = useRef<HTMLDivElement>(null);
  const canvasWrapRef= useRef<HTMLDivElement>(null);
  const viewerRef    = useRef<NebulaViewer | null>(null);
  const bindingRef   = useRef<HTMLDivElement>(null);

  /* ─── Theme effect ─── */
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  /* ─── Fullscreen effect ─── */
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  function toggleFullscreen() {
    if (!canvasWrapRef.current) return;
    if (document.fullscreenElement) document.exitFullscreen();
    else canvasWrapRef.current.requestFullscreen();
  }

  /* ─── Load targets ─── */
  useEffect(() => {
    listTargets()
      .then(t => { setTargets(t); setTarget(t[0]?.id ?? null); setBusy(t.length ? '' : 'No precomputed targets found.'); })
      .catch((e: Error) => { setError(`Cannot reach API — ${e.message}. Run: make api`); setBusy(''); });
  }, []);

  /* ─── Init 3D viewer ─── */
  useEffect(() => {
    let dead = false;
    NebulaViewer.create(canvasRef.current!, { spin: true })
      .then(v => { if (dead) v.dispose(); else viewerRef.current = v; })
      .catch((e: Error) => setError(`3D viewer failed: ${e.message}`));
    return () => { dead = true; viewerRef.current?.dispose(); viewerRef.current = null; };
  }, []);

  /* ─── Load selected target ─── */
  useEffect(() => {
    if (!target) return;
    setTraj(null); setRows([]); setIdx(0); setProps(null); setContacts(null);
    setError(null); setBusy('Loading target…');
    Promise.all([getManifest(target), getMolecules(target), getPocket(target)])
      .then(([m, mols, pk]) => {
        setManifest(m); setPocket(pk); setSummary(mols.summary);
        setRows(mols.molecules.filter(r => r.valid && molIdFromFile(r.file)));
        setBusy('');
      })
      .catch((e: Error) => { setError(e.message); setBusy(''); });

    const hasPot = targets.find(t => t.id === target)?.has_potential ?? false;
    const wait = setInterval(async () => {
      const v = viewerRef.current; if (!v) return;
      clearInterval(wait);
      try {
        await v.clear();
        await v.loadPocket(fileUrl(target, 'pocket.pdb'), hasPot ? fileUrl(target, 'surface_potential.dx') : null);
        v.setLayers({});
      } catch (e) { setError(`Pocket load failed: ${(e as Error).message}`); }
    }, 100);
    return () => clearInterval(wait);
  }, [target, targets]);

  /* ─── Load molecule ─── */
  const row   = rows[idx];
  const molId = row ? molIdFromFile(row.file)! : null;

  useEffect(() => {
    if (!target || !molId) return;
    let dead = false;
    setPlaying(false); setProps(null); setContacts(null); setBusy('Loading molecule…');
    getTrajectory(target, molId).then(async t => {
      if (dead) return;
      const last = t.frames.length - 1;
      setTraj(t); setFrame(last);
      const wait = setInterval(async () => {
        if (!viewerRef.current) return;
        clearInterval(wait);
        await viewerRef.current.loadTrajectory(t, finalSdfUrl(target, molId), last);
        await viewerRef.current.setFrame(last);
        viewerRef.current.setLayers({});
        setBusy('');
      }, 80);
    }).catch((e: Error) => { setError(e.message); setBusy(''); });
    getProps(target, molId).then(p => !dead && setProps(p)).catch(() => {});
    getContacts(target, molId).then(c => !dead && setContacts(c)).catch(() => {});
    return () => { dead = true; };
  }, [target, molId]);

  /* ─── Playback ─── */
  useEffect(() => { viewerRef.current?.setFrame(frame); }, [frame]);
  useEffect(() => {
    if (!playing || !traj) return;
    const id = setInterval(() => setFrame(x => {
      if (x >= traj.frames.length - 1) { setPlaying(false); return x; }
      return x + 1;
    }), 160 / speed);
    return () => clearInterval(id);
  }, [playing, traj, speed]);

  /* ─── Layer toggle ─── */
  const setLayer = useCallback((k: keyof typeof layers) => (v: boolean) => {
    setLayers(l => ({ ...l, [k]: v }));
    viewerRef.current?.setLayers({ [k]: v });
    if (k === 'trajectory' && !v && traj) { setPlaying(false); setFrame(traj.frames.length - 1); }
  }, [traj]);

  /* ─── Thumb frames ─── */
  const thumbs = useMemo(() => {
    if (!traj) return null;
    const n    = traj.frames.length;
    const pick = [0, 0.1, 0.2, 0.4, 0.7, 1].map(p => Math.round(p * (n - 1)));
    const last = traj.frames[n - 1];
    const cx   = last.reduce((s, p) => s + p[0], 0) / last.length;
    const cy   = last.reduce((s, p) => s + p[1], 0) / last.length;
    let ext = 1;
    pick.forEach(i => traj.frames[i].forEach(p => {
      ext = Math.max(ext, Math.abs(p[0] - cx), Math.abs(p[1] - cy));
    }));
    return { pick, scale: 32 / ext };
  }, [traj]);

  /* ─── Filtered/sorted rows ─── */
  const filteredRows = useMemo(() => {
    let r = [...rows];
    if (lipinskiOnly) r = r.filter(m => !m.errors || Object.keys(m.errors).length === 0);
    r.sort((a, b) => {
      const ak = (a as any)[sortKey] ?? -Infinity;
      const bk = (b as any)[sortKey] ?? -Infinity;
      return sortDir === 'desc' ? bk - ak : ak - bk;
    });
    return r;
  }, [rows, sortKey, sortDir, lipinskiOnly]);

  /* ─── Derived ─── */
  const shown  = targets.filter(t => {
    const q = searchQ.trim().toLowerCase();
    if (srcTab === 'pdbbind') return (t.source ?? '').toLowerCase().includes('rcsb');
    return !q || t.id.includes(q) || (t.source ?? '').toLowerCase().includes(q);
  });
  const pk         = row?.gnn_pk;
  const pkPos      = pk == null ? null : Math.min(100, Math.max(0, ((pk - 4) / 8) * 100));
  const qual       = affQual(pk);
  const nGnn       = rows.filter(r => r.gnn_pk != null).length;
  const nVina      = rows.filter(r => r.vina_score_only != null).length;
  const nPass      = rows.filter(r => r.posebusters_all_pass).length;
  const nCands     = (summary.n_files as number) ?? rows.length;
  const scrubFill  = traj ? (frame / (traj.frames.length - 1)) * 100 : 0;
  const curMeta    = target ? getMeta(target, manifest?.source) : null;
  const activeThumb = thumbs
    ? thumbs.pick.reduce((best, i) => (Math.abs(i - frame) < Math.abs(best - frame) ? i : best), thumbs.pick[0])
    : -1;

  /* ─── Add custom target ─── */
  function addCustomTarget(id: string, source: string) {
    const newT: TargetSummary = { id, source, date: new Date().toISOString().slice(0, 10), n_molecules: 0, has_potential: false };
    setTargets(prev => [...prev, newT]);
    setTarget(id);
    setActiveTab('explore');
  }

  /* ─── Apply pocket edits ─── */
  function applyPocketEdit(p: { centroid: [number,number,number]; cutoff: number; refLigand: string }) {
    setCustomCentroid(p.centroid);
    setCustomCutoff(p.cutoff);
    setCustomRefLig(p.refLigand);
    setPocket(prev => prev ? { ...prev, centroid: p.centroid, cutoff_A: p.cutoff, reference_ligand: p.refLigand || prev.reference_ligand } : prev);
  }

  /* ─── Refresh targets and molecules ─── */
  const refreshTargetsAndMols = useCallback(async (targetId?: string) => {
    try {
      const updatedTargets = await listTargets();
      setTargets(updatedTargets);
      const tid = targetId ?? target;
      if (tid) {
        const mols = await getMolecules(tid);
        setRows(mols.molecules.filter(r => r.valid && molIdFromFile(r.file)));
        setSummary(mols.summary);
      }
    } catch {}
  }, [target]);

  /* ─── Layer config ─── */
  const LAYER_CFG = [
    { key: 'surface'    as const, label: 'Protein Surface',       icCls: 'lic-gold',   tone: 'gold',   Icon: IcMolecule },
    { key: 'potential'  as const, label: 'Electrostatic Potential',icCls: 'lic-blue',   tone: 'blue',   Icon: IcZap },
    { key: 'pocket'     as const, label: 'Pocket (Cavity)',        icCls: 'lic-blue',   tone: 'blue',   Icon: IcTarget },
    { key: 'ligand'     as const, label: 'Generated Ligand',       icCls: 'lic-or',     tone: '',       Icon: IcAtom },
    { key: 'atoms'      as const, label: 'Atom Interactions',      icCls: 'lic-purple', tone: 'purple', Icon: IcActivity },
    { key: 'hbonds'     as const, label: 'Hydrogen Bonds',         icCls: 'lic-cyan',   tone: 'cyan',   Icon: IcWave },
    { key: 'trajectory' as const, label: 'Denoising Trajectory',   icCls: 'lic-green',  tone: 'green',  Icon: IcPlay },
  ] as const;

  /* ─── Pipeline steps ─── */
  const pipeSteps = [
    { Icon: IcFlask,    icCls: 'step-ic-gold', name: 'Generate', desc: 'Create candidate ligands',    stat: `${nCands} candidates generated`,    current: true },
    { Icon: IcActivity, icCls: 'step-ic-gray', name: 'Score',    desc: 'Predict binding affinity',    stat: `${nGnn} affinities computed`,       current: false },
    { Icon: IcCheck,    icCls: 'step-ic-gray', name: 'Validate', desc: 'Docking & drug-likeness',     stat: `${nVina} docked, ${nPass} pass`,    current: false },
    { Icon: IcEye,      icCls: 'step-ic-gray', name: 'Explore',  desc: 'Visualize and analyze',       stat: 'Ready to explore',                  current: false },
  ];

  /* ═══════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════ */
  return (
    <div className="app">

      {/* ══ MODALS ══ */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onAdd={addCustomTarget}
        />
      )}
      {showEditPocket && (
        <EditPocketModal
          pocket={pocket}
          onClose={() => setShowEditPocket(false)}
          onSave={applyPocketEdit}
        />
      )}

      {/* ══ TOPBAR ══ */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon"><IcAtom size={17} /></div>
          <div className="brand-text">
            <b>PocketLab<span className="brand-beta">β</span></b>
            <small>AI-POWERED LIGAND GENERATION</small>
          </div>
        </div>

        <nav className="topnav" aria-label="Primary">
          {(['explore','generate','analyze','dataset','docs'] as const).map(tab => (
            <button
              key={tab}
              className={`navtab${activeTab === tab ? ' active' : ''}`}
              aria-current={activeTab === tab ? 'page' : undefined}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>

        <label className="topsearch">
          <IcSearch size={14} />
          <input
            placeholder="Search protein (e.g. 6LU7)"
            value={searchQ}
            onChange={e => setSearchQ(e.target.value)}
            aria-label="Search protein"
          />
        </label>

        <div className="topright">
          <button className="topicon" title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
            onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? <IcSun size={14} /> : <IcMoon size={14} />}
          </button>
          <button className="avatar-btn" aria-label="User profile"><IcUser size={14} /></button>
        </div>
      </header>

      {/* ══ NON-EXPLORE FULL VIEWS ══ */}
      {activeTab === 'generate' && (
        <GenerateView
          targets={targets}
          initialTarget={target}
          onGenerated={refreshTargetsAndMols}
          onInspect={async (id, molId) => {
            await refreshTargetsAndMols(id);
            setTarget(id);
            setActiveTab('explore');
            if (molId) {
              try {
                const resp = await getMolecules(id);
                const validRows = resp.molecules.filter(r => r.valid && molIdFromFile(r.file));
                const foundIdx = validRows.findIndex(m => molIdFromFile(m.file) === molId);
                if (foundIdx >= 0) {
                  setTimeout(() => setIdx(foundIdx), 150);
                }
              } catch {}
            }
          }}
        />
      )}
      {activeTab === 'analyze' && (
        <AnalyzeView
          rows={filteredRows}
          target={target}
          onInspect={i => { setIdx(i); setActiveTab('explore'); }}
        />
      )}
      {activeTab === 'dataset' && (
        <DatasetView
          targets={targets}
          onLoad={id => { setTarget(id); setActiveTab('explore'); }}
        />
      )}
      {activeTab === 'docs' && <DocsView />}

      {/* ══ EXPLORE VIEW (3-column studio) ══ */}
      {activeTab === 'explore' && (
        <div className="main-content">

          {/* ━━ LEFT ━━ */}
          <div className="col-left">

            {/* Target Protein */}
            <div className="panel target-panel panel-gold-corner">
              <div className="panel-hdr">
                <div className="panel-title">
                  <span className="panel-icon panel-icon-gold"><IcTarget size={12} /></span>
                  Target Protein
                </div>
                <button className="panel-close" aria-label="Close"><IcClose size={11} /></button>
              </div>
              <p className="panel-sub">Select from examples or upload your own</p>

              <label className="target-search">
                <IcSearch size={12} />
                <input
                  placeholder="Search PDB ID or name…"
                  value={searchQ}
                  onChange={e => setSearchQ(e.target.value)}
                  aria-label="Search targets"
                />
              </label>

              <div className="seg-group">
                <button className={`seg-btn${srcTab === 'examples' ? ' active' : ''}`} onClick={() => setSrcTab('examples')}>Examples</button>
                <button className={`seg-btn${srcTab === 'upload' ? ' active' : ''}`} onClick={() => { setSrcTab('upload'); setShowUpload(true); }}>Upload</button>
                <button className={`seg-btn${srcTab === 'pdbbind' ? ' active' : ''}`} onClick={() => setSrcTab('pdbbind')}>PDBbind</button>
              </div>

              <ul className="target-list" role="listbox" aria-label="Target list">
                {shown.map(t => {
                  const m = getMeta(t.id, t.source);
                  return (
                    <li key={t.id} role="option" aria-selected={t.id === target}>
                      <button
                        className={`target-item${t.id === target ? ' active' : ''}`}
                        onClick={() => setTarget(t.id)}
                      >
                        <div className={`target-thumb ${m.thumbCls}`}>
                          <TargetThumbIcon cls={m.thumbCls} />
                        </div>
                        <div className="target-info">
                          <div className="target-name">{m.name}</div>
                          <div className="target-pdb">{t.id.toUpperCase()} · <strong style={{ color: 'var(--gold)' }}>{t.n_molecules} ligands</strong></div>
                        </div>
                        <span className={`badge ${m.badgeCls}`}>{m.badgeLabel}</span>
                      </button>
                    </li>
                  );
                })}
                {!shown.length && (
                  <li><p className="muted" style={{ padding: '8px 10px', fontSize: 11 }}>No targets match.</p></li>
                )}
              </ul>
            </div>

            {/* Pocket Selection */}
            <div className="panel pocket-panel">
              <div className="panel-hdr">
                <div className="panel-title">
                  <span className="panel-icon panel-icon-or"><IcBox size={12} /></span>
                  Pocket Selection
                </div>
              </div>
              <p className="panel-sub">Define or choose binding pocket</p>

              <div className="pocket-segs">
                <button className={`pocket-seg${pocketTab === 'auto' ? ' active' : ''}`} onClick={() => setPocketTab('auto')}>Auto-detect</button>
                <button className={`pocket-seg${pocketTab === 'custom' ? ' active' : ''}`} onClick={() => setPocketTab('custom')}>Custom</button>
              </div>

              {pocketTab === 'custom' && (
                <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ fontSize: 10.5, color: 'var(--mut2)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.04em' }}>Custom Centroid (Å)</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                    {(['X','Y','Z'] as const).map((axis, ai) => (
                      <div key={axis} className="pocket-coord-input">
                        <label>{axis}</label>
                        <input
                          type="number" step="0.1"
                          value={(customCentroid?.[ai] ?? pocket?.centroid?.[ai] ?? 0).toFixed(1)}
                          onChange={e => {
                            const base = customCentroid ?? (pocket?.centroid as [number,number,number] ?? [0,0,0]);
                            const next: [number,number,number] = [...base] as [number,number,number];
                            next[ai] = Number(e.target.value);
                            setCustomCentroid(next);
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  <div className="pocket-cutoff-row">
                    <span style={{ fontSize: 10.5, color: 'var(--mut)', minWidth: 52 }}>Cutoff</span>
                    <input type="range" min={4} max={20} step={0.5}
                      value={customCutoff ?? pocket?.cutoff_A ?? 10}
                      onChange={e => setCustomCutoff(Number(e.target.value))} />
                    <span className="pocket-cutoff-val">{(customCutoff ?? pocket?.cutoff_A ?? 10).toFixed(1)} Å</span>
                  </div>
                </div>
              )}

              {pocket ? (
                <div className="pocket-preview">
                  <div className="pocket-thumb-card" />
                  <div>
                    <div className="pocket-detected">
                      <span className="pocket-dot" />
                      {pocketTab === 'auto' ? 'Detected Pocket' : 'Custom Pocket'}
                    </div>
                    <div className="pocket-kv">
                      <div>Center (Å)</div>
                      {pocket.centroid && (
                        <div>
                          x: <span>{(customCentroid?.[0] ?? pocket.centroid[0]).toFixed(1)}</span>{' '}
                          y: <span>{(customCentroid?.[1] ?? pocket.centroid[1]).toFixed(1)}</span>{' '}
                          z: <span>{(customCentroid?.[2] ?? pocket.centroid[2]).toFixed(1)}</span>
                        </div>
                      )}
                      <div style={{ marginTop: 4 }}>
                        Residues: <span>{pocket.residues}</span>
                      </div>
                      <div>Atoms: <span>{pocket.atoms}</span></div>
                      {(customCutoff ?? pocket.cutoff_A) && <div>Cutoff: <span>{(customCutoff ?? pocket.cutoff_A)} Å</span></div>}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="muted" style={{ padding: '8px 11px', fontSize: 11 }}>
                  {busy || 'Select a target to view pocket info.'}
                </p>
              )}

              <button className="edit-pocket-btn" onClick={() => setShowEditPocket(true)}>
                <IcEdit size={12} style={{ marginRight: 5, verticalAlign: 'middle' }} />
                Edit Pocket
              </button>
            </div>
          </div>

          {/* ━━ MIDDLE ━━ */}
          <div className="col-mid">

            {/* Viewer */}
            <div className="panel viewer-panel panel-gold-corner">
              <div className="viewer-hdr">
                <div className="viewer-title">
                  <span className="viewer-title-icon"><IcAtom size={13} /></span>
                  {curMeta ? curMeta.name : 'No target selected'}
                </div>
                <div className="viewer-badges">
                  {curMeta && (
                    <span className={`badge ${curMeta.badgeCls}`}>{curMeta.badgeLabel}</span>
                  )}
                  <span className="badge badge-pdb">PDB</span>
                </div>
                {busy && (
                  <div className="busy-row">
                    <span className="spinner" />
                    {busy}
                  </div>
                )}
              </div>

              {error && <div className="alert-bar" role="alert"><IcInfo size={12} style={{ marginRight: 5 }} />{error}</div>}

              <div className="viewer-canvas-wrap" ref={canvasWrapRef}>
                {/* 3D canvas */}
                <div className="canvas" ref={canvasRef} />

                {/* Measurement overlay */}
                {measureMode && contacts && contacts.contacts.length > 0 && (
                  <div className="measure-overlay">
                    {contacts.contacts.slice(0, 5).map((c, i) => (
                      <div key={i} className="measure-chip" style={{
                        left: `${20 + i * 14}%`, top: `${30 + i * 10}%`,
                      }}>
                        {c.ligand_atom}→{c.residue}: {c.distance.toFixed(1)} Å
                      </div>
                    ))}
                  </div>
                )}

                {/* Left toolbar */}
                <div className="viewer-toolbar">
                  <button className="vt-btn" title="Reset camera" onClick={() => viewerRef.current?.resetCamera()}>
                    <IcRefresh size={13} />
                  </button>
                  <button className="vt-btn" title="Toggle Visualization Layers"
                    onClick={() => setLayersOpen(o => !o)}>
                    <IcEye size={13} />
                  </button>
                  <div style={{ position: 'relative' }}>
                    <button className={`vt-btn${showFilter ? ' active' : ''}`} title="Sort & Filter candidates"
                      onClick={() => { setShowFilter(o => !o); setShowVpSettings(false); }}>
                      <IcFilter size={13} />
                    </button>
                    {showFilter && (
                      <FilterPopover
                        sortKey={sortKey} sortDir={sortDir} lipinskiOnly={lipinskiOnly}
                        onSortChange={(k, d) => { setSortKey(k); setSortDir(d as 'desc'|'asc'); }}
                        onLipinskiChange={setLipinskiOnly}
                        onClose={() => setShowFilter(false)}
                      />
                    )}
                  </div>
                  <button className={`vt-btn${measureMode ? ' active' : ''}`} title="Distance Measurement Mode"
                    onClick={() => setMeasureMode(m => !m)}>
                    <IcBarChart size={13} />
                  </button>
                  <div style={{ position: 'relative' }}>
                    <button className={`vt-btn${showVpSettings ? ' active' : ''}`} title="3D Viewport Settings"
                      onClick={() => { setShowVpSettings(o => !o); setShowFilter(false); }}>
                      <IcGear size={13} />
                    </button>
                    {showVpSettings && (
                      <VpSettingsPopover onClose={() => setShowVpSettings(false)} />
                    )}
                  </div>
                </div>

                {/* Right toolbar */}
                <div className="viewer-toolbar-right">
                  <button className="vt-btn" title="Reset" onClick={() => viewerRef.current?.resetCamera()}><IcRefresh size={13} /></button>
                  <button className="vt-btn" title="Zoom in"  onClick={() => viewerRef.current?.zoomIn()}><IcZoomIn size={13} /></button>
                  <button className="vt-btn" title="Zoom out" onClick={() => viewerRef.current?.zoomOut()}><IcZoomOut size={13} /></button>
                  <button className="vt-btn" title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                    onClick={toggleFullscreen}>
                    {isFullscreen ? <IcMinimize size={13} /> : <IcMaximize size={13} />}
                  </button>
                </div>

                {/* Visualization Layers overlay */}
                <div className="layers-panel">
                  <button
                    className="layers-hdr"
                    onClick={() => setLayersOpen(o => !o)}
                    aria-expanded={layersOpen}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                      <span className="layers-hdr-icon"><IcLayers size={11} /></span>
                      Visualization Layers
                    </div>
                    <span className={`layers-chevron${layersOpen ? ' open' : ''}`}>
                      {layersOpen ? <IcChevronUp size={12} /> : <IcChevronDown size={12} />}
                    </span>
                  </button>
                  {layersOpen && (
                    <div className="layers-body">
                      {LAYER_CFG.map(({ key, label, icCls, tone, Icon }) => (
                        <label key={key} className="layer-row">
                          <div className="layer-left">
                            <span className={`layer-ic ${icCls}`}><Icon size={10} /></span>
                            <span>{label}</span>
                          </div>
                          <Toggle on={layers[key]} onChange={setLayer(key)} tone={tone} />
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                {/* ES bar */}
                <div className="esbar">
                  <div className="esbar-label">Electrostatic Potential (kT/e)</div>
                  <div className="esbar-track">
                    <span className="esbar-val">−5.0</span>
                    <div className="esbar-grad" />
                    <span className="esbar-val">+5.0</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Trajectory */}
            <div className="panel traj-panel">
              <div className="traj-hdr">
                <div className="traj-hdr-left">
                  <div className="traj-title">
                    <span className="traj-icon"><IcActivity size={11} /></span>
                    Denoising Trajectory
                  </div>
                  <div className="traj-sub">Watch the ligand form inside the pocket</div>
                </div>
                <div className="traj-controls">
                  <button
                    className="play-btn"
                    disabled={!traj}
                    onClick={() => {
                      if (traj && frame >= traj.frames.length - 1) setFrame(0);
                      setPlaying(p => !p);
                    }}
                  >
                    {playing ? <IcPause size={12} /> : <IcPlay size={12} />}
                    {playing ? 'Pause' : 'Play'}
                  </button>
                  <select className="speed-sel" value={speed} onChange={e => setSpeed(Number(e.target.value))} aria-label="Speed">
                    {[0.5, 1, 2, 4].map(s => <option key={s} value={s}>{s}x</option>)}
                  </select>
                </div>
              </div>

              <div className="thumbs-row">
                {traj && thumbs ? thumbs.pick.map(i => (
                  <button
                    key={i}
                    className={`thumb-card${i === activeThumb ? ' active' : ''}`}
                    onClick={() => { setPlaying(false); setFrame(i); }}
                    aria-label={`Step ${traj.steps[i]}`}
                  >
                    <Thumb t={traj} i={i} scale={thumbs.scale} />
                    <span className="thumb-lbl">Step {traj.steps[i]}</span>
                  </button>
                )) : (
                  <p className="muted" style={{ gridColumn: '1/-1', fontSize: 11, padding: '6px 4px' }}>
                    Select a ligand candidate to view its denoising trajectory.
                  </p>
                )}
              </div>

              <div className="scrubber-wrap">
                <input
                  type="range"
                  className="scrubber"
                  min={0}
                  max={traj ? traj.frames.length - 1 : 0}
                  value={frame}
                  disabled={!traj}
                  onChange={e => { setPlaying(false); setFrame(Number(e.target.value)); }}
                  style={{ '--fill': `${scrubFill}%` } as React.CSSProperties}
                  aria-label="Denoising step"
                />
              </div>
            </div>
          </div>

          {/* ━━ RIGHT ━━ */}
          <div className="col-right">

            {/* Generated Ligand */}
            <div className="panel ligand-panel panel-gold-corner">
              <div className="ligand-hdr">
                <div>
                  <div className="ligand-title">
                    <span className="ligand-ic"><IcMolecule size={12} /></span>
                    Generated Ligand
                    {(lipinskiOnly || sortKey !== 'gnn_pk') && (
                      <span style={{ fontSize: 9, background: 'var(--gold-glow2)', color: 'var(--gold)', borderRadius: 4, padding: '1px 5px', marginLeft: 5 }}>
                        Filtered
                      </span>
                    )}
                  </div>
                  <div className="ligand-cand">
                    Candidate {filteredRows.length ? idx + 1 : 0} / {filteredRows.length}
                  </div>
                </div>
                <div className="pager">
                  <button className="pager-btn" onClick={() => setIdx(i => Math.max(0, i - 1))} disabled={idx <= 0} aria-label="Previous">
                    <IcChevronLeft size={12} />
                  </button>
                  <button className="pager-btn" onClick={() => setIdx(i => Math.min(filteredRows.length - 1, i + 1))} disabled={idx >= filteredRows.length - 1} aria-label="Next">
                    <IcChevronRight size={12} />
                  </button>
                </div>
              </div>

              <div className="lig-top-row">
                {/* 2D structure */}
                <div className="s2d-card">
                  {row
                    ? <Structure2D smiles={row.smiles} />
                    : <p className="muted" style={{ fontSize: 11, textAlign: 'center' }}>No molecule</p>
                  }
                </div>

                {/* Affinity */}
                <div className="aff-card">
                  <div className="aff-label">Predicted Affinity (pKd)</div>
                  <div className="aff-row">
                    <span className="aff-val">{f(pk)}</span>
                    {pk != null && <span className={`aff-badge ${qual.cls}`}>{qual.label}</span>}
                  </div>
                  <div className="aff-grad-track">
                    {pkPos != null && <div className="aff-marker" style={{ left: `${pkPos}%` }} />}
                  </div>
                  <div className="aff-ticks">
                    <span>4</span><span>6</span><span>8</span><span>10</span><span>12</span>
                  </div>
                </div>
              </div>

              {/* Metrics */}
              {row && (
                <div className="metrics-tbl">
                  <span className="mlabel">QED</span>
                  <div className="mval">
                    <span>{f(row.qed)}</span>
                    <div className="mbar">
                      <div className="mbar-fill" style={{ width: `${(row.qed ?? 0) * 100}%` }} />
                    </div>
                  </div>

                  <span className="mlabel">SA Score</span>
                  <div className="mval">
                    <span>{f(row.sa)}</span>
                    <div className="mbar">
                      <div className="mbar-fill or" style={{ width: `${Math.min(100, ((row.sa ?? 0) / 10) * 100)}%` }} />
                    </div>
                  </div>

                  <span className="mlabel">Lipinski</span>
                  <div className="mval">
                    {props
                      ? props.lipinski_violations.length === 0
                        ? <span className="ok"><IcCheck size={11} /> Passes (0 violations)</span>
                        : <span className="warn">{props.lipinski_violations.length} violation(s)</span>
                      : <span className="muted">–</span>
                    }
                  </div>

                  <span className="mlabel">Docking Score</span>
                  <div className="mval">
                    {row.vina_score_only != null
                      ? <span className="dock-val">
                          <span className="dock-arrow"><IcTrendDown size={11} /></span>
                          {f(row.vina_score_only, 1)} kcal/mol
                        </span>
                      : <span className="muted">–</span>
                    }
                  </div>

                  {row.gnina_cnn_affinity != null && <>
                    <span className="mlabel">GNINA CNN</span>
                    <div className="mval">
                      <span>{f(row.gnina_cnn_affinity)}</span>
                      <div className="mbar">
                        <div className="mbar-fill gold" style={{ width: `${Math.min(100, (row.gnina_cnn_affinity / 12) * 100)}%` }} />
                      </div>
                    </div>
                  </>}

                  {row.posebusters_all_pass != null && <>
                    <span className="mlabel">PoseBusters</span>
                    <div className="mval">
                      {row.posebusters_all_pass
                        ? <span className="ok"><IcCheck size={11} /> All checks pass</span>
                        : <span className="warn">Some checks fail</span>
                      }
                    </div>
                  </>}
                </div>
              )}

              <button
                className="view-int-btn"
                disabled={!row}
                onClick={() => bindingRef.current?.scrollIntoView({ behavior: 'smooth' })}
              >
                View Interactions <IcArrowRight size={13} />
              </button>
            </div>

            {/* 3D Conformation */}
            <div className="panel conf-panel panel-gold-corner">
              <div className="conf-hdr">
                <div className="conf-title">
                  <span className="conf-ic"><IcCube size={12} /></span>
                  3D Conformation
                </div>
                <span className="props-label">Properties</span>
              </div>

              <div className="conf-grid">
                <Conformer sdfUrl={target && molId ? finalSdfUrl(target, molId) : null} />

                <div className="conf-props-tbl">
                  {props ? (
                    <>
                      <span className="cp-label">Molecular Weight</span>
                      <span className="cp-val">{f(props.molecular_weight, 1)} g/mol</span>
                      <span className="cp-label">LogP</span>
                      <span className="cp-val">{f(props.logp, 1)}</span>
                      <span className="cp-label">H-Bond Donors</span>
                      <span className="cp-val">{props.hbd}</span>
                      <span className="cp-label">H-Bond Acceptors</span>
                      <span className="cp-val">{props.hba}</span>
                      <span className="cp-label">Rotatable Bonds</span>
                      <span className="cp-val">{props.rotatable_bonds}</span>
                      <span className="cp-label">TPSA</span>
                      <span className="cp-val">{f(props.tpsa, 1)} Å²</span>
                    </>
                  ) : (
                    <span className="muted" style={{ fontSize: 11, gridColumn: '1/-1' }}>
                      {row ? 'Loading…' : 'Select a molecule'}
                    </span>
                  )}
                  {target && molId && (
                    <a
                      className="dl-btn"
                      href={finalSdfUrl(target, molId)}
                      download={`${target}_${molId}.sdf`}
                    >
                      <IcDownload size={13} /> Download
                    </a>
                  )}
                </div>
              </div>
            </div>

            {/* Binding Interactions */}
            <div className="panel binding-panel panel-gold-corner" ref={bindingRef}>
              <div className="binding-hdr">
                <div className="binding-title">
                  <span className="binding-ic"><IcLink size={12} /></span>
                  Binding Interactions
                </div>
              </div>

              <div className="int-list">
                {contacts ? (
                  contacts.contacts.length ? contacts.contacts.map((c, i) => (
                    <div key={i} className="int-row">
                      <span className={intDot(c.type)}><span /></span>
                      <span className="int-type">{intLabel(c.type)}</span>
                      <span className="int-desc">
                        Ligand ({c.ligand_atom}) → {c.residue} ({c.protein_atom})
                      </span>
                      <span className="int-dist">{c.distance.toFixed(1)} Å</span>
                    </div>
                  )) : (
                    <p className="muted" style={{ fontSize: 11 }}>No contacts within cutoffs.</p>
                  )
                ) : (
                  <p className="muted" style={{ fontSize: 11 }}>
                    {row ? 'Computing interactions…' : 'Select a ligand candidate.'}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══ FOOTER — PIPELINE ══ */}
      <footer className="footer-pipeline">
        <div className="pipe-steps">
          {pipeSteps.map(({ Icon, icCls, name, desc, stat, current }) => (
            <div key={name} className="pipe-cell">
              <div className={`pipe-step${current ? ' current' : ''}`}>
                <span className="pipe-shape" aria-hidden="true" />
                <div className={`pipe-step-icon ${icCls}`}><Icon size={16} /></div>
                <div className="pipe-step-info">
                  <div className="pipe-step-name">{name}</div>
                  <div className="pipe-step-desc">{desc}</div>
                </div>
              </div>
              <div className="pipe-step-stat">
                <span className="pipe-check"><IcCheck size={9} /></span> {stat}
              </div>
            </div>
          ))}
        </div>

        <div className="pipe-right">
          <div className="gen-row">
            <button className="gen-btn" onClick={() => setActiveTab('generate')}>
              Generate New Set <IcArrowRight size={14} />
            </button>
            <button className="settings-btn" title="Advanced settings" aria-label="Settings"
              onClick={() => setActiveTab('generate')}>
              <IcGear size={14} />
            </button>
          </div>
          <div className="adv-link" style={{ cursor: 'pointer' }} onClick={() => setActiveTab('generate')}>
            <IcGear size={11} /> Advanced Settings
          </div>
        </div>
      </footer>
    </div>
  );
}

