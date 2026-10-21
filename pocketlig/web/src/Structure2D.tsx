import { useEffect, useState } from 'react';
import initRDKitModule from '@rdkit/rdkit';
import rdkitWasmUrl from '@rdkit/rdkit/RDKit_minimal.wasm?url';

let rdkitPromise: Promise<any> | null = null;
function getRDKit(): Promise<any> {
  if (!rdkitPromise) rdkitPromise = (initRDKitModule as any)({ locateFile: () => rdkitWasmUrl });
  return rdkitPromise as Promise<any>;
}

/** 2D depiction drawn by RDKit.js; black default strokes are recolored for the dark theme. */
export function Structure2D({ smiles }: { smiles?: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setSvg(null); setError(null);
    if (!smiles) return;
    getRDKit().then((rd: any) => {
      const mol = rd.get_mol(smiles);
      if (!mol || !mol.is_valid()) { if (alive) setError('RDKit could not read this SMILES'); if (mol) mol.delete(); return; }
      const out: string = mol.get_svg_with_highlights(JSON.stringify({ width: 300, height: 220, backgroundColour: [0, 0, 0, 0] }));
      mol.delete();
      if (alive) setSvg(out.replaceAll('#000000', '#e8ecf5'));
    }).catch((e: Error) => alive && setError(e.message));
    return () => { alive = false; };
  }, [smiles]);
  if (!smiles) return <p className="muted">No sanitized molecule for this sample.</p>;
  if (error) return <p className="muted">{error}</p>;
  return <div className="s2d" role="img" aria-label={`2D structure, SMILES ${smiles}`} dangerouslySetInnerHTML={{ __html: svg ?? '' }} />;
}
