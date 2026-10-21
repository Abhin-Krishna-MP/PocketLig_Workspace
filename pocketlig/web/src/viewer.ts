// Thin wrapper over Mol* (molstar 5.x). Patterns follow Mol*'s own code: the same ModelFromTrajectory
// update that its built-in "model index" animation uses, and the built-in 'external-volume' color theme.
import { Viewer } from 'molstar/lib/apps/viewer/app';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { PluginStateObject } from 'molstar/lib/mol-plugin-state/objects';
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms';
import { setSubtreeVisibility } from 'molstar/lib/mol-plugin/behavior/static/state';
import { StateObject, StateObjectSelector } from 'molstar/lib/mol-state';
import { Color } from 'molstar/lib/mol-util/color';
import { StructureElement } from 'molstar/lib/mol-model/structure';
import type { Trajectory } from './api';

type Sel<T extends StateObject = PluginStateObject.Molecule.Model> = StateObjectSelector<T>;

const PDB_ELEMENT_COL = 76;

/** Atoms only (no CONECT): noisy intermediate frames have no real bonds. */
export function framesToPdb(t: Trajectory): string {
  const out: string[] = [];
  t.frames.forEach((frame, f) => {
    out.push(`MODEL     ${String(f + 1).padStart(4)}`);
    frame.forEach((xyz, i) => {
      const el = (t.frame_elements[f]?.[i] ?? t.elements[i] ?? 'C');
      const name = el.padEnd(4);
      const line = `HETATM${String(i + 1).padStart(5)} ${name} LIG L   1    ` +
        `${xyz[0].toFixed(3).padStart(8)}${xyz[1].toFixed(3).padStart(8)}${xyz[2].toFixed(3).padStart(8)}` +
        `  1.00  0.00`;
      out.push(line.padEnd(PDB_ELEMENT_COL - 2) + el.padStart(2));
    });
    out.push('ENDMDL');
  });
  out.push('END');
  return out.join('\n') + '\n';
}

export class PocketViewer {
  private constructor(private viewer: Viewer) {}
  private trajDataRef?: string;
  private trajModel?: Sel;
  private trajStructureRef?: string;
  private finalDataRef?: string;
  private finalStructureRef?: string;
  private lastIndex = 0;
  private refs: { colored?: string; plain?: string; pocket?: string; cartoon?: string; shell?: string; cloud?: string; halo?: string; ligandSticks?: string; ligandSurface?: string } = {};
  layers = { surface: true, potential: true, pocket: true, ligand: true, trajectory: true };
  private frame = 0;
  private busy = false;
  private pending: number | null = null;

  static async create(el: HTMLElement, opts: { spin?: boolean } = {}): Promise<PocketViewer> {
    const viewer = await Viewer.create(el, {
      layoutIsExpanded: false, layoutShowControls: false, layoutShowRemoteState: false, layoutShowSequence: false,
      layoutShowLog: false, layoutShowLeftPanel: false, viewportShowExpand: false, viewportShowSelectionMode: false,
      viewportShowAnimation: false, collapseLeftPanel: true, collapseRightPanel: true,
    });
    // Transparent canvas (CSS provides haze, stars, vignette) + ambient occlusion + emissive bloom.
    try {
      viewer.plugin.canvas3d?.setProps((old) => ({
        transparentBackground: true,
        checkeredTransparentBackground: false,
        renderer: {
          ...old.renderer,
          backgroundColor: Color(0x000000),
          xrayEdgeFalloff: 1.2,
          exposure: 1.05,
        },
        postprocessing: {
          ...old.postprocessing,
          enabled: true,
          occlusion: { name: 'on', params: { samples: 32, multiScale: { name: 'off', params: {} }, radius: 5, bias: 0.8, blurKernelSize: 15, blurDepthBias: 0.5, resolutionScale: 1, color: 0, transparentThreshold: 0.4 } },
          bloom: { name: 'on', params: { strength: 0.35, radius: 0.8, threshold: 0.55, mode: 'emissive', transparency: false } },
          outline: { name: 'off', params: {} },
        },
        ...(opts.spin ? { trackball: { ...old.trackball, animate: { name: 'spin', params: { speed: 0.10, axis: [0, -1, 0] } } } } : {}),
      } as any));
    } catch {
      // If postprocessing fails (unsupported GPU or API mismatch), continue without effects.
    }
    return new PocketViewer(viewer);
  }

  private get plugin() { return this.viewer.plugin; }

  async clearLigand(): Promise<void> {
    const p = this.plugin;
    const refs = [this.trajDataRef, this.finalDataRef, this.trajStructureRef, this.finalStructureRef];
    for (const ref of refs) {
      if (ref && p.state.data.cells.has(ref)) {
        try {
          await PluginCommands.State.RemoveObject(p, { state: p.state.data, ref });
        } catch { /* ignore if already cleaned */ }
      }
    }
    this.trajDataRef = undefined;
    this.trajModel = undefined;
    this.trajStructureRef = undefined;
    this.finalDataRef = undefined;
    this.finalStructureRef = undefined;
  }

  async clear() {
    await this.clearLigand();
    await this.plugin.clear();
    this.refs = {};
  }

  /** Pocket residues with cartoon ribbon and electrostatic surface (APBS red negative, blue positive). */
  async loadPocket(pdbUrl: string, dxUrl: string | null, range = 5): Promise<void> {
    const p = this.plugin;
    const data = await p.builders.data.download({ url: pdbUrl, isBinary: false }, { state: { isGhost: true } });
    const traj = await p.builders.structure.parseTrajectory(data, 'pdb');
    const model = await p.builders.structure.createModel(traj);
    const structure = await p.builders.structure.createStructure(model);
    const polymer = await p.builders.structure.tryCreateComponentStatic(structure, 'polymer');
    if (!polymer) throw new Error('pocket has no polymer component');
    this.refs = {};

    // 1) Protein backbone cartoon ribbon
    try {
      const cartoon = await p.builders.structure.representation.addRepresentation(polymer, {
        type: 'cartoon',
        color: 'uniform',
        colorParams: { value: Color(0x3b82f6) },
        typeParams: { alpha: 0.65 } as any,
      });
      this.refs.cartoon = cartoon.ref;
    } catch { /* cartoon representation fallback */ }

    // 2) Pocket binding site residues as thin sticks
    const sticks = await p.builders.structure.representation.addRepresentation(polymer, {
      type: 'ball-and-stick',
      color: 'element-symbol',
      typeParams: { sizeFactor: 0.14, emissive: 0.05 } as any,
    });
    this.refs.pocket = sticks.ref;

    // 3) Pocket molecular surface (subtle translucent blue fallback)
    const plain = await p.builders.structure.representation.addRepresentation(polymer, {
      type: 'molecular-surface',
      typeParams: { alpha: 0.32, emissive: 0.05, probeRadius: 1.8 } as any,
      color: 'uniform',
      colorParams: { value: Color(0x1e3a8a) },
    });
    this.refs.plain = plain.ref;

    if (dxUrl) {
      const raw = await p.builders.data.download({ url: dxUrl, isBinary: false }, { state: { isGhost: true } });
      const parsed = await p.dataFormats.get('dx')!.parse(p, raw);
      const volume = (parsed.volume || parsed.volumes?.[0]) as Sel<PluginStateObject.Volume.Data> | undefined;
      if (!volume) throw new Error('APBS .dx file contained no volume');
      // Translucent potential-coloured molecular surface (legend gradient)
      const colored = await p.builders.structure.representation.addRepresentation(polymer, {
        type: 'molecular-surface',
        typeParams: {
          alpha: 0.36,
          emissive: 0.08,
          doubleSided: true,
          probeRadius: 1.8,
          resolution: 0.8,
          bumpFrequency: 1.2,
          bumpAmplitude: 0.3,
        } as any,
        color: 'external-volume',
        colorParams: {
          volume: { ref: volume.ref },
          coloring: {
            name: 'absolute-value',
            params: {
              domain: { name: 'custom', params: [-range, range] },
              list: { kind: 'interpolate', colors: [[Color(0xff5a1f), 0], [Color(0xf1ecff), 0.5], [Color(0x2f6bff), 1]] },
            },
          },
          defaultColor: Color(0x3b82f6),
        },
      } as any);
      this.refs.colored = colored.ref;
    }
    this.applyLayers();
  }

  setLayers(l: Partial<PocketViewer['layers']>) {
    this.layers = { ...this.layers, ...l };
    this.applyLayers();
    void this.setFrame(this.frame);
  }

  private applyLayers() {
    const st = this.plugin.state.data, L = this.layers;
    const hide = (ref: string | undefined, hidden: boolean) => { if (ref) setSubtreeVisibility(st, ref, hidden); };
    hide(this.refs.cartoon, !L.surface);
    hide(this.refs.colored, !(L.surface && L.potential));
    hide(this.refs.plain, !(L.surface && (!L.potential || !this.refs.colored)));
    hide(this.refs.pocket, !L.pocket);
  }

  /** Denoising trajectory as a multi-model structure: scrubbing only updates the model index (no reload). */
  async loadTrajectory(t: Trajectory, finalSdfUrl: string | null, initialFrame?: number): Promise<void> {
    await this.clearLigand();
    const p = this.plugin;
    const raw = await p.builders.data.rawData({ data: framesToPdb(t), label: 'denoising' }, { state: { isGhost: true } });
    this.trajDataRef = raw.ref;
    const traj = await p.builders.structure.parseTrajectory(raw, 'pdb');
    this.trajModel = await p.builders.structure.createModel(traj, { modelIndex: 0 });
    const structure = await p.builders.structure.createStructure(this.trajModel);
    this.trajStructureRef = structure.ref;
    await p.builders.structure.representation.addRepresentation(structure, {
      type: 'spacefill', color: 'element-symbol', typeParams: { sizeFactor: 0.28, emissive: 0.8 } as any,
    });
    this.finalStructureRef = undefined;
    this.finalDataRef = undefined;
    if (finalSdfUrl) {
      try {
        const sdf = await p.builders.data.download({ url: finalSdfUrl, isBinary: false }, { state: { isGhost: true } });
        this.finalDataRef = sdf.ref;
        const ftraj = await p.builders.structure.parseTrajectory(sdf, 'sdf');
        const fmodel = await p.builders.structure.createModel(ftraj);
        const fstruct = await p.builders.structure.createStructure(fmodel);
        this.finalStructureRef = fstruct.ref;

        // 1. Crystal-clear Ball-and-Stick: atom spheres with distinct CPK colors, bonds as solid 3D cylinders
        const ligandSticks = await p.builders.structure.representation.addRepresentation(fstruct, {
          type: 'ball-and-stick',
          color: 'element-symbol',
          typeParams: { sizeFactor: 0.35, sizeAspectRatio: 0.8, emissive: 0.08 } as any,
        });
        this.refs.ligandSticks = ligandSticks.ref;

        // 2. Translucent 3D Molecular Surface envelope contouring the exact shape/volume of the molecule
        try {
          const ligandSurface = await p.builders.structure.representation.addRepresentation(fstruct, {
            type: 'molecular-surface',
            color: 'uniform',
            colorParams: { value: Color(0x38bdf8) },
            typeParams: { alpha: 0.28, emissive: 0.08, probeRadius: 1.2, doubleSided: true } as any,
          });
          this.refs.ligandSurface = ligandSurface.ref;
        } catch { /* molecular surface optional */ }

        setSubtreeVisibility(p.state.data, fstruct.ref, true); // hidden until the last frame
      } catch {
        this.finalStructureRef = undefined;
        this.finalDataRef = undefined;
      }
    }
    this.lastIndex = t.frames.length - 1;
    const targetFrame = initialFrame !== undefined ? initialFrame : this.lastIndex;
    await this.setFrame(targetFrame);

    // Smoothly focus camera onto the ligand so it fills the view with its true shape!
    try {
      if (this.finalStructureRef) {
        const cell = p.state.data.cells.get(this.finalStructureRef);
        if (cell && cell.obj && cell.obj.data) {
          p.managers.camera.focusLoci(StructureElement.Loci.all(cell.obj.data), { durationMs: 400, extraRadius: 4 });
        }
      }
    } catch {
      p.managers.camera.reset();
    }
  }

  /** Move to frame i; calls arriving while an update runs are coalesced to the latest. */
  async setFrame(i: number): Promise<void> {
    if (!this.trajModel) return;
    if (this.busy) { this.pending = i; return; }
    this.busy = true;
    try {
      const p = this.plugin;
      await p.build().to(this.trajModel).update(StateTransforms.Model.ModelFromTrajectory, (old) => ({ ...old, modelIndex: i })).commit();
      this.frame = i;
      const L = this.layers;
      const atEnd = (i >= this.lastIndex || !L.trajectory) && !!this.finalStructureRef;
      if (this.finalStructureRef) setSubtreeVisibility(p.state.data, this.finalStructureRef, !(L.ligand && atEnd));
      if (this.trajStructureRef) setSubtreeVisibility(p.state.data, this.trajStructureRef, !(L.ligand && !atEnd));
    } finally {
      this.busy = false;
      if (this.pending !== null) { const n = this.pending; this.pending = null; await this.setFrame(n); }
    }
  }

  /** Standalone ligand preview (used by the small 3D Conformation panel). */
  async loadSdf(url: string): Promise<void> {
    const p = this.plugin;
    const sdf = await p.builders.data.download({ url, isBinary: false }, { state: { isGhost: true } });
    const traj = await p.builders.structure.parseTrajectory(sdf, 'sdf');
    const model = await p.builders.structure.createModel(traj);
    const structure = await p.builders.structure.createStructure(model);

    // Crisp 3D ball and stick
    await p.builders.structure.representation.addRepresentation(structure, {
      type: 'ball-and-stick',
      color: 'element-symbol',
      typeParams: { sizeFactor: 0.38, sizeAspectRatio: 0.8, emissive: 0.08 } as any,
    });

    // 3D molecular shape surface envelope
    try {
      await p.builders.structure.representation.addRepresentation(structure, {
        type: 'molecular-surface',
        color: 'uniform',
        colorParams: { value: Color(0x38bdf8) },
        typeParams: { alpha: 0.28, emissive: 0.08, probeRadius: 1.2, doubleSided: true } as any,
      });
    } catch { /* surface optional */ }

    p.managers.camera.reset();
  }

  /* Camera controls */
  resetCamera() { this.plugin.managers.camera.reset(); }
  zoomIn() {
    const c = this.plugin.canvas3d?.camera;
    if (c) {
      c.setState({ ...c.state, radius: Math.max(5, c.state.radius * 0.8) });
      this.plugin.canvas3d?.requestDraw();
    }
  }
  zoomOut() {
    const c = this.plugin.canvas3d?.camera;
    if (c) {
      c.setState({ ...c.state, radius: Math.min(500, c.state.radius * 1.25) });
      this.plugin.canvas3d?.requestDraw();
    }
  }
  dispose() { this.viewer.dispose(); }
}
