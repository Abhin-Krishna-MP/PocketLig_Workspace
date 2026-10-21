import { useEffect, useRef } from 'react';
import { PocketViewer } from './viewer';

/** Small standalone Mol* view of the final ligand (the '3D Conformation' panel). */
export function Conformer({ sdfUrl }: { sdfUrl: string | null }) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<PocketViewer | null>(null);
  useEffect(() => {
    let dead = false;
    PocketViewer.create(host.current!).then((v) => { if (dead) v.dispose(); else viewer.current = v; });
    return () => { dead = true; viewer.current?.dispose(); viewer.current = null; };
  }, []);
  useEffect(() => {
    if (!sdfUrl) return;
    const t = setInterval(async () => {
      const v = viewer.current;
      if (!v) return;
      clearInterval(t);
      await v.clear();
      try { await v.loadSdf(sdfUrl); } catch { /* sample has no sanitized final molecule */ }
    }, 80);
    return () => clearInterval(t);
  }, [sdfUrl]);
  return <div className="conformer" ref={host} aria-label="3D conformation of the generated ligand" />;
}
