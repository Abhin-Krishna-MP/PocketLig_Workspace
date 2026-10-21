// Nebula renderer: a from-scratch WebGL2 raymarcher (no dependencies) for the main 3D viewport.
// - Pocket atoms are splatted into a 3D density texture (gaussian "blobby" surface).
// - The APBS potential (.dx) is resampled into the same grid and used to colour the surface with the
//   legend gradient (orange-red -> pale white -> blue).
// - The ray marcher renders a translucent glassy skin + soft volumetric body + hot cavity glow + sparkle dust.
// - The ligand / denoising atoms are analytic emissive ball-and-stick, composited behind the front wall.
// API is intentionally identical to PocketViewer so App.tsx can swap it in.
import type { Trajectory } from './api';

type V3 = [number, number, number];
interface Atom { p: V3; el: string }

const EL_RGB: Record<string, V3> = {
  C: [0.80, 0.82, 0.88], N: [0.25, 0.45, 1.0], O: [1.0, 0.25, 0.18], S: [1.0, 0.82, 0.25],
  F: [0.5, 0.9, 0.65], P: [1.0, 0.62, 0.26], Cl: [0.41, 0.82, 0.5], Br: [0.75, 0.44, 0.23], H: [0.9, 0.9, 0.95],
};
const elRgb = (e: string): V3 => EL_RGB[e] ?? [0.75, 0.52, 0.99];
const VDW: Record<string, number> = { H: 1.1, C: 1.7, N: 1.55, O: 1.52, S: 1.8, P: 1.8, F: 1.47, Cl: 1.75, Br: 1.85 };

const MAX_SPH = 512;
const MAX_CAP = 1024;

const VS = `#version 300 es
in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;
out vec4 outColor;

uniform vec2 uRes;
uniform vec3 uCam, uRight, uUp, uFwd, uBMin, uBMax, uTarget, uInvDim;
uniform float uTanF, uR, uCut, uTime, uIso, uHasPot;
uniform vec4 uF;            // x: surface, y: potential colouring, z: ligand, w: sparkle
uniform sampler3D uVol;     // r: density, g: potential (0..1)
uniform sampler2D uSA;      // xyz centre, w radius
uniform sampler2D uSB;      // rgb colour, a emissive
uniform int uN, uNC;
uniform sampler2D uCT;      // capsules: row0 pa+radius, row1 pb+emissive, row2 rgb

float hash13(vec3 p){
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)); p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
vec3 potColor(float g){
  vec3 negC = vec3(1.00, 0.30, 0.08);
  vec3 midC = vec3(0.78, 0.82, 1.00);
  vec3 posC = vec3(0.12, 0.32, 1.00);
  float s = g * 2.0 - 1.0;
  s = sign(s) * pow(abs(s), 0.42);
  return s < 0.0 ? mix(midC, negC, -s) : mix(midC, posC, s);
}
float capIntersect(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra){
  vec3 ba = pb - pa, oa = ro - pa;
  float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
  float a = baba - bard * bard;
  float b = baba * rdoa - baoa * bard;
  float c = baba * oaoa - baoa * baoa - ra * ra * baba;
  float h = b * b - a * c;
  if (h >= 0.0) {
    float t = (-b - sqrt(h)) / a;
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba) return t;
    vec3 oc = (y <= 0.0) ? oa : ro - pb;
    b = dot(rd, oc); c = dot(oc, oc) - ra * ra; h = b * b - c;
    if (h > 0.0) return -b - sqrt(h);
  }
  return -1.0;
}
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float a = hash13(i), b = hash13(i + vec3(1,0,0)), c = hash13(i + vec3(0,1,0)), d = hash13(i + vec3(1,1,0));
  float e = hash13(i + vec3(0,0,1)), g = hash13(i + vec3(1,0,1)), h = hash13(i + vec3(0,1,1)), k = hash13(i + vec3(1,1,1));
  return mix(mix(mix(a,b,f.x), mix(c,d,f.x), f.y), mix(mix(e,g,f.x), mix(h,k,f.x), f.y), f.z);
}
float fbm(vec3 p){ return 0.6 * vnoise(p) + 0.3 * vnoise(p * 2.1 + 5.3) + 0.1 * vnoise(p * 4.3 + 1.7); }
float dens(vec3 p){ return texture(uVol, (p - uBMin) / (uBMax - uBMin)).r; }

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
  vec3 rd = normalize(uFwd + (uRight * uv.x + uUp * uv.y) * 2.0 * uTanF);
  vec3 ro = uCam;
  float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);

  // bounding box
  vec3 inv = 1.0 / rd;
  vec3 ta = (uBMin - ro) * inv, tb = (uBMax - ro) * inv;
  vec3 tmn = min(ta, tb), tmx = max(ta, tb);
  float tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
  float tf = min(min(tmx.x, tmx.y), tmx.z);

  // analytic atoms (spheres) + bonds (capsules): nearest hit, additive glow. Anything in front of the cut plane is ignored.
  float tcCut = (uCut < 1e4) ? dot(uTarget - ro, uFwd) / dot(rd, uFwd) - uCut : -1e9;
  float tL = 1e9; int hit = -1; int capHit = -1; vec3 glow = vec3(0.0);
  for (int i = 0; i < ${MAX_SPH}; i++) {
    if (i >= uN) break;
    vec4 a = texelFetch(uSA, ivec2(i, 0), 0);
    vec4 cb = texelFetch(uSB, ivec2(i, 0), 0);
    vec3 o = ro - a.xyz;
    float b = dot(o, rd);
    float oo = dot(o, o);
    float h = b * b - (oo - a.w * a.w);
    float d2 = max(oo - b * b, 0.0);
    glow += cb.rgb * cb.a * 0.018 * exp(-d2 / (0.6 + a.w * a.w * 5.0)) * step(tcCut, -b);
    if (h > 0.0) { float t = -b - sqrt(h); if (t > max(0.0, tcCut) && t < tL) { tL = t; hit = i; } }
  }
  for (int i = 0; i < ${MAX_CAP}; i++) {
    if (i >= uNC) break;
    vec4 pa = texelFetch(uCT, ivec2(i, 0), 0);
    vec4 pb = texelFetch(uCT, ivec2(i, 1), 0);
    float t = capIntersect(ro, rd, pa.xyz, pb.xyz, pa.w);
    if (t > max(0.0, tcCut) && t < tL) { tL = t; hit = -1; capHit = i; }
  }

  vec3 C = vec3(0.0);
  float T = 1.0;
  vec3 Ld = normalize(-uFwd * 0.8 + uUp * 0.7 - uRight * 0.5);

  if (uF.x > 0.5 && tf > tn) {
    float tcen = dot(uTarget - ro, uFwd) / dot(rd, uFwd);
    float ts = max(tn, tcen - uCut);
    float te = min(tf, tL);
    float dt = max(0.26, (tf - tn) / 200.0);
    float t = ts + dt * dither;
    for (int i = 0; i < 260; i++) {
      if (t > te || T < 0.015) break;
      vec3 p = ro + rd * t;
      vec2 v = texture(uVol, (p - uBMin) / (uBMax - uBMin)).rg;
      float nz = fbm(p * 0.55);
      float d = v.r + 0.20 * (nz - 0.5);            // organic lumps
      if (d < 0.10) { t += dt * 1.6; continue; }

      float body = smoothstep(0.28, 0.9, d);
      float s1 = exp(-pow((d - uIso) / 0.075, 2.0));
      float s2 = 0.7 * exp(-pow((d - 0.80) / 0.06, 2.0));   // inner skin -> nested glassy layers
      float s = clamp(s1 + s2, 0.0, 1.0);
      float rr = length(p - uTarget) / uR;
      float cav = exp(-pow(rr / 0.30, 2.0));

      float g;
      if (uHasPot > 0.5) g = v.g;
      else g = mix(0.06, 0.95, smoothstep(0.16, 0.85, rr + 0.12 * (nz - 0.5)));
      vec3 base = uF.y > 0.5 ? potColor(g) : vec3(0.18, 0.38, 1.0);

      vec3 col = base * 0.34 * mix(1.0, 0.45, smoothstep(0.6, 1.0, d));
      if (s > 0.12) {
        vec3 gr = vec3(
          dens(p + vec3(0.9, 0, 0)) - dens(p - vec3(0.9, 0, 0)),
          dens(p + vec3(0, 0.9, 0)) - dens(p - vec3(0, 0.9, 0)),
          dens(p + vec3(0, 0, 0.9)) - dens(p - vec3(0, 0, 0.9)));
        vec3 n = -normalize(gr + 1e-5);
        n = normalize(n + 0.55 * (vec3(vnoise(p * 1.7), vnoise(p * 1.7 + 9.1), vnoise(p * 1.7 + 4.2)) - 0.5));
        float diff = max(dot(n, Ld), 0.0);
        float fr = pow(1.0 - abs(dot(n, rd)), 2.2);
        float sp = pow(max(dot(reflect(rd, n), Ld), 0.0), 34.0);
        float sp2 = pow(max(dot(reflect(rd, n), normalize(Ld * vec3(-1, 0.4, -0.7))), 0.0), 18.0);
        vec3 surf = base * (0.16 + 0.60 * diff) + vec3(0.92, 0.96, 1.0) * (sp * 0.8 + sp2 * 0.2)
                  + mix(base, vec3(0.55, 0.80, 1.0), 0.6) * fr * 1.05;
        col = mix(col, surf, clamp(s * 1.2, 0.0, 1.0));
      }
      // hot cavity emission (orange/red core glow)
      vec3 em = vec3(1.0, 0.38, 0.07) * cav * (0.30 * body + 1.6 * s) * 1.3;
      // sparkle dust sitting on the skin
      if (uF.w > 0.5) {
        vec3 cell = floor(p * 1.1);
        float h = hash13(cell);
        vec3 jp = (cell + 0.2 + 0.6 * vec3(hash13(cell + 7.1), hash13(cell + 3.7), hash13(cell + 1.3))) / 1.1;
        float sdist = length(p - jp);
        float tw = 0.65 + 0.35 * sin(uTime * 2.0 + h * 40.0);
        em += vec3(0.75, 0.88, 1.0) * step(0.72, h) * smoothstep(0.30, 0.0, sdist) * (0.4 + 2.2 * s) * tw * 0.9;
      }
      float sigma = 0.42 * body + 1.7 * s;
      float a = 1.0 - exp(-sigma * dt);
      C += T * (a * col + em * dt * 0.55);
      T *= (1.0 - a);
      t += dt;
    }
  }

  if (hit >= 0 || capHit >= 0) {
    vec3 p = ro + rd * tL; vec3 n; vec3 colr; float em;
    if (hit >= 0) {
      vec4 A = texelFetch(uSA, ivec2(hit, 0), 0); vec4 B = texelFetch(uSB, ivec2(hit, 0), 0);
      n = normalize(p - A.xyz); colr = B.rgb; em = B.a;
    } else {
      vec4 pa = texelFetch(uCT, ivec2(capHit, 0), 0); vec4 pb = texelFetch(uCT, ivec2(capHit, 1), 0); vec4 cc = texelFetch(uCT, ivec2(capHit, 2), 0);
      vec3 ba = pb.xyz - pa.xyz; float hh = clamp(dot(p - pa.xyz, ba) / dot(ba, ba), 0.0, 1.0);
      n = normalize(p - pa.xyz - ba * hh); colr = cc.rgb; em = pb.w;
    }
    float diff = max(dot(n, Ld), 0.0);
    float sp = pow(max(dot(reflect(rd, n), Ld), 0.0), 60.0);
    float fr = pow(1.0 - max(dot(n, -rd), 0.0), 2.5);
    vec3 lig = colr * (0.38 + 0.8 * diff) + vec3(1.0) * sp * 0.8 + colr * em * 0.55 + vec3(0.6, 0.8, 1.0) * fr * 0.35;
    C += T * lig;
    T *= 0.0;
  }
  C += glow * (0.45 + 0.55 * T);

  float alpha = clamp(1.0 - T, 0.0, 1.0);
  alpha = max(alpha, clamp(max(glow.r, max(glow.g, glow.b)) * 0.8, 0.0, 1.0));
  C = C / (1.0 + 0.22 * C);               // soft tone map (keeps highlights bloomy)
  outColor = vec4(min(C, vec3(1.4)), alpha);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
  return s;
}

/* ───────────── parsers ───────────── */
export function parsePdbAtoms(text: string, cap = 7000): Atom[] {
  const out: Atom[] = [];
  for (const line of text.split('\n')) {
    if (!(line.startsWith('ATOM') || line.startsWith('HETATM'))) continue;
    const x = parseFloat(line.slice(30, 38)), y = parseFloat(line.slice(38, 46)), z = parseFloat(line.slice(46, 54));
    if (Number.isNaN(x + y + z)) continue;
    let el = line.slice(76, 78).trim();
    if (!el) el = line.slice(12, 16).trim().replace(/[0-9]/g, '').slice(0, 1);
    el = el ? el[0].toUpperCase() + el.slice(1).toLowerCase() : 'C';
    if (el === 'H') continue;
    out.push({ p: [x, y, z], el });
    if (out.length >= cap) break;
  }
  return out;
}

export function parseSdf(text: string): { atoms: Atom[]; bonds: [number, number][] } | null {
  const L = text.split('\n');
  if (L.length < 5) return null;
  const na = parseInt(L[3].slice(0, 3)), nb = parseInt(L[3].slice(3, 6));
  if (!(na > 0)) return null;
  const atoms: Atom[] = [], bonds: [number, number][] = [];
  for (let i = 0; i < na; i++) {
    const l = L[4 + i];
    atoms.push({ p: [parseFloat(l.slice(0, 10)), parseFloat(l.slice(10, 20)), parseFloat(l.slice(20, 30))], el: l.slice(31, 34).trim() || 'C' });
  }
  for (let i = 0; i < nb; i++) {
    const l = L[4 + na + i];
    bonds.push([parseInt(l.slice(0, 3)) - 1, parseInt(l.slice(3, 6)) - 1]);
  }
  return { atoms, bonds };
}

interface Dx { n: V3; o: V3; d: V3; data: Float32Array }
export function parseDx(text: string): Dx | null {
  const lines = text.split('\n');
  let n: V3 | null = null, o: V3 = [0, 0, 0]; const d: V3 = [1, 1, 1]; let di = 0, start = -1;
  for (let i = 0; i < lines.length && start < 0; i++) {
    const l = lines[i].trim();
    if (l.startsWith('object 1')) { const m = l.split(/\s+/); n = [+m[m.length - 3], +m[m.length - 2], +m[m.length - 1]]; }
    else if (l.startsWith('origin')) { const m = l.split(/\s+/); o = [+m[1], +m[2], +m[3]]; }
    else if (l.startsWith('delta')) { const m = l.split(/\s+/).slice(1).map(Number); d[di] = m[di]; di++; }
    else if (l.startsWith('object 3')) start = i + 1;
  }
  if (!n || start < 0) return null;
  const total = n[0] * n[1] * n[2];
  const data = new Float32Array(total); let k = 0;
  for (let i = start; i < lines.length && k < total; i++) {
    const l = lines[i];
    if (l.startsWith('attribute') || l.startsWith('object') || l.startsWith('component')) break;
    for (const tok of l.trim().split(/\s+/)) { if (tok && k < total) data[k++] = parseFloat(tok); }
  }
  return k === total ? { n, o, d, data } : null;
}

/* ───────────── viewer ───────────── */
export class NebulaViewer {
  layers = { surface: true, potential: true, pocket: true, ligand: true, trajectory: true, atoms: true, hbonds: true };
  hasFinalMolecule = false;

  private gl!: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;
  private prog!: WebGLProgram;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private volTex: WebGLTexture | null = null;
  private saTex!: WebGLTexture; private sbTex!: WebGLTexture;
  private nSph = 0; private nCap = 0; private ctTex!: WebGLTexture;
  private pocketAtoms: Atom[] = []; private pocketPrims: Prims | null = null;
  private bmin: V3 = [-10, -10, -10]; private bmax: V3 = [10, 10, 10];
  private invDim: V3 = [0.01, 0.01, 0.01];
  private target: V3 = [0, 0, 0]; private R = 12; private hasPot = false; private hasVol = false;
  private yaw = 0.6; private pitch = 0.25; private dist = 30;
  private dragging = false; private lx = 0; private ly = 0;
  private raf = 0; private dirty = true; private t0 = performance.now(); private disposed = false;
  private ro: ResizeObserver;
  private traj?: Trajectory;
  private finalMol: { atoms: Atom[]; bonds: [number, number][] } | null = null;
  private frame = 0;
  private ligandCenter: V3 | null = null;

  private constructor(private el: HTMLElement, private spin: boolean) {
    if ((window as any).__PL_NO_SPIN) this.spin = false;
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab;';
    el.appendChild(this.canvas);
    const gl = this.canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    gl.getExtension('EXT_color_buffer_float');
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
    this.prog = prog; gl.useProgram(prog);
    for (const n of ['uRes', 'uCam', 'uRight', 'uUp', 'uFwd', 'uBMin', 'uBMax', 'uTarget', 'uInvDim', 'uTanF', 'uR', 'uCut', 'uTime', 'uIso', 'uHasPot', 'uF', 'uVol', 'uSA', 'uSB', 'uN', 'uNC', 'uCT'])
      this.u[n] = gl.getUniformLocation(prog, n);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'aPos'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.saTex = this.makeSphTex(); this.sbTex = this.makeSphTex();
    this.ctTex = this.makeSphTex();
    this.uploadSpheres(emptyPrims());

    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => { this.dragging = true; this.lx = e.clientX; this.ly = e.clientY; c.setPointerCapture(e.pointerId); c.style.cursor = 'grabbing'; });
    c.addEventListener('pointerup', (e) => { this.dragging = false; c.releasePointerCapture(e.pointerId); c.style.cursor = 'grab'; });
    c.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.yaw -= (e.clientX - this.lx) * 0.008; this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (e.clientY - this.ly) * 0.008));
      this.lx = e.clientX; this.ly = e.clientY; this.dirty = true;
    });
    c.addEventListener('wheel', (e) => { e.preventDefault(); this.dist = Math.max(this.R * 0.6, Math.min(this.R * 8, this.dist * (1 + Math.sign(e.deltaY) * 0.08))); this.dirty = true; }, { passive: false });
    this.ro = new ResizeObserver(() => { this.dirty = true; });
    this.ro.observe(el);
    let last = 0;
    const loop = (now = 0) => { if (this.disposed) return; this.raf = requestAnimationFrame(loop); if (now - last < 32) return; last = now; if (this.spin && !this.dragging) { this.yaw += 0.006; this.dirty = true; } if (this.dirty) this.render(); };
    loop();
  }

  static async create(el: HTMLElement, opts: { spin?: boolean } = {}): Promise<NebulaViewer> { return new NebulaViewer(el, !!opts.spin); }

  private makeSphTex() {
    const gl = this.gl; const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private uploadSpheres(pr: Prims) {
    const gl = this.gl;
    const n = Math.min(MAX_SPH, pr.sa.length / 4); this.nSph = n;
    const nc = Math.min(MAX_CAP, pr.ca.length / 4); this.nCap = nc;
    const A = new Float32Array(MAX_SPH * 4), B = new Float32Array(MAX_SPH * 4);
    A.set(pr.sa.slice(0, n * 4)); B.set(pr.sb.slice(0, n * 4));
    const CT = new Float32Array(MAX_CAP * 4 * 3);
    CT.set(pr.ca.slice(0, nc * 4), 0); CT.set(pr.cb.slice(0, nc * 4), MAX_CAP * 4); CT.set(pr.cc.slice(0, nc * 4), MAX_CAP * 8);
    gl.bindTexture(gl.TEXTURE_2D, this.saTex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_SPH, 1, 0, gl.RGBA, gl.FLOAT, A);
    gl.bindTexture(gl.TEXTURE_2D, this.sbTex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_SPH, 1, 0, gl.RGBA, gl.FLOAT, B);
    gl.bindTexture(gl.TEXTURE_2D, this.ctTex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_CAP, 3, 0, gl.RGBA, gl.FLOAT, CT);
    this.dirty = true;
  }

  /** Ball-and-stick primitives: atoms -> spheres, bonds -> two half-capsules coloured per atom. */
  private molPrims(atoms: Atom[], bonds: [number, number][], atomR: number, bondR: number, em: number, tint = 1, into?: Prims): Prims {
    const o = into ?? emptyPrims();
    const col = (e: string): V3 => { const c = elRgb(e); return [c[0] * tint, c[1] * tint, c[2] * tint]; };
    for (const at of atoms) { o.sa.push(at.p[0], at.p[1], at.p[2], atomR); const c = col(at.el); o.sb.push(c[0], c[1], c[2], em); }
    for (const [i, j] of bonds) {
      const A = atoms[i], B = atoms[j]; if (!A || !B) continue;
      const m: V3 = [(A.p[0] + B.p[0]) / 2, (A.p[1] + B.p[1]) / 2, (A.p[2] + B.p[2]) / 2];
      const ca = col(A.el), cb = col(B.el);
      o.ca.push(A.p[0], A.p[1], A.p[2], bondR); o.cb.push(m[0], m[1], m[2], em * 0.7); o.cc.push(ca[0], ca[1], ca[2], 0);
      o.ca.push(m[0], m[1], m[2], bondR); o.cb.push(B.p[0], B.p[1], B.p[2], em * 0.7); o.cc.push(cb[0], cb[1], cb[2], 0);
    }
    return o;
  }

  /** Dashed hydrogen-bond lines between ligand N/O and pocket N/O within 3.4 A. */
  private hbondPrims(lig: Atom[], into: Prims) {
    const polar = (e: string) => e === 'N' || e === 'O';
    const red: V3 = [1.0, 0.32, 0.22];
    for (const a of lig) {
      if (!polar(a.el)) continue;
      let best: Atom | null = null, bd = 3.4;
      for (const b of this.pocketAtoms) {
        if (!polar(b.el)) continue;
        const d = Math.hypot(a.p[0] - b.p[0], a.p[1] - b.p[1], a.p[2] - b.p[2]);
        if (d < bd && d > 2.2) { bd = d; best = b; }
      }
      if (!best) continue;
      const n = Math.max(3, Math.round(bd / 0.34));
      for (let k = 0; k < n; k++) {
        const f0 = k / n, f1 = f0 + 0.55 / n;
        const P = (f: number): V3 => [a.p[0] + (best!.p[0] - a.p[0]) * f, a.p[1] + (best!.p[1] - a.p[1]) * f, a.p[2] + (best!.p[2] - a.p[2]) * f];
        const p0 = P(f0), p1 = P(f1);
        into.ca.push(p0[0], p0[1], p0[2], 0.06); into.cb.push(p1[0], p1[1], p1[2], 1.4); into.cc.push(red[0], red[1], red[2], 0);
      }
    }
  }

  /** Pocket residue atoms nearest the ligand/cavity centre, as dim ball-and-stick (bonds inferred from distance). */
  private buildPocketPrims() {
    const c = this.target; const R = 10.5;
    const near = this.pocketAtoms.filter((a) => Math.hypot(a.p[0] - c[0], a.p[1] - c[1], a.p[2] - c[2]) < R)
      .sort((x, y) => Math.hypot(x.p[0] - c[0], x.p[1] - c[1], x.p[2] - c[2]) - Math.hypot(y.p[0] - c[0], y.p[1] - c[1], y.p[2] - c[2]))
      .slice(0, 190);
    const bonds: [number, number][] = [];
    for (let i = 0; i < near.length; i++) for (let j = i + 1; j < near.length; j++) {
      const A = near[i], B = near[j]; const d = Math.hypot(A.p[0] - B.p[0], A.p[1] - B.p[1], A.p[2] - B.p[2]);
      const lim = (A.el === 'S' || B.el === 'S') ? 2.1 : 1.72;
      if (d < lim && d > 0.8) bonds.push([i, j]);
    }
    this.pocketPrims = this.molPrims(near, bonds, 0.34, 0.15, 0.45, 0.9);
  }

  async clear() {
    const gl = this.gl;
    if (this.volTex) { gl.deleteTexture(this.volTex); this.volTex = null; }
    this.hasVol = false; this.hasPot = false; this.traj = undefined; this.finalMol = null; this.hasFinalMolecule = false;
    this.pocketAtoms = []; this.pocketPrims = null; this.uploadSpheres(emptyPrims()); this.ligandCenter = null;
  }

  async loadPocket(pdbUrl: string, dxUrl: string | null, range = 5): Promise<void> {
    const text = await (await fetch(pdbUrl)).text();
    const atoms = parsePdbAtoms(text);
    if (!atoms.length) throw new Error('pocket has no atoms');
    let dx: Dx | null = null;
    if (dxUrl) { try { dx = parseDx(await (await fetch(dxUrl)).text()); } catch { dx = null; } }
    this.pocketAtoms = atoms;
    this.buildVolume(atoms, dx, range);
    this.buildPocketPrims();
    this.refreshSpheres();
  }

  private buildVolume(atoms: Atom[], dx: Dx | null, range: number) {
    const gl = this.gl;
    const mn: V3 = [1e9, 1e9, 1e9], mx: V3 = [-1e9, -1e9, -1e9];
    for (const a of atoms) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], a.p[k]); mx[k] = Math.max(mx[k], a.p[k]); }
    const pad = 6;
    const bmin: V3 = [mn[0] - pad, mn[1] - pad, mn[2] - pad], bmax: V3 = [mx[0] + pad, mx[1] + pad, mx[2] + pad];
    const ext = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
    const vox = Math.max(0.7, Math.max(...ext) / 112);
    const dim: V3 = [Math.ceil(ext[0] / vox), Math.ceil(ext[1] / vox), Math.ceil(ext[2] / vox)];
    // snap bbox to the voxel grid
    const bmax2: V3 = [bmin[0] + dim[0] * vox, bmin[1] + dim[1] * vox, bmin[2] + dim[2] * vox];
    const N = dim[0] * dim[1] * dim[2];
    const sum = new Float32Array(N);
    const cut = 6.4, rv = Math.ceil(cut / vox);
    for (const a of atoms) {
      const sig = 0.85 * (VDW[a.el] ?? 1.7);          // gaussian width scales with atom size
      const inv = 1 / (2 * sig * sig * 0.55);
      const ci = [(a.p[0] - bmin[0]) / vox, (a.p[1] - bmin[1]) / vox, (a.p[2] - bmin[2]) / vox];
      for (let z = Math.max(0, Math.floor(ci[2] - rv)); z <= Math.min(dim[2] - 1, Math.ceil(ci[2] + rv)); z++) {
        const dz = (z + 0.5 - ci[2]) * vox;
        for (let y = Math.max(0, Math.floor(ci[1] - rv)); y <= Math.min(dim[1] - 1, Math.ceil(ci[1] + rv)); y++) {
          const dy = (y + 0.5 - ci[1]) * vox;
          const row = (z * dim[1] + y) * dim[0];
          for (let x = Math.max(0, Math.floor(ci[0] - rv)); x <= Math.min(dim[0] - 1, Math.ceil(ci[0] + rv)); x++) {
            const dxx = (x + 0.5 - ci[0]) * vox;
            const r2 = dxx * dxx + dy * dy + dz * dz;
            if (r2 < cut * cut) sum[row + x] += Math.exp(-r2 * inv);
          }
        }
      }
    }
    const data = new Float32Array(N * 2);
    const samp = (px: number, py: number, pz: number): number => {
      if (!dx) return 0.5;
      const fx = (px - dx.o[0]) / dx.d[0], fy = (py - dx.o[1]) / dx.d[1], fz = (pz - dx.o[2]) / dx.d[2];
      if (fx < 0 || fy < 0 || fz < 0 || fx > dx.n[0] - 1 || fy > dx.n[1] - 1 || fz > dx.n[2] - 1) return 0;
      const x0 = Math.min(dx.n[0] - 2, Math.floor(fx)), y0 = Math.min(dx.n[1] - 2, Math.floor(fy)), z0 = Math.min(dx.n[2] - 2, Math.floor(fz));
      const tx = fx - x0, ty = fy - y0, tz = fz - z0;
      const at = (i: number, j: number, k: number) => dx!.data[(i * dx!.n[1] + j) * dx!.n[2] + k];   // DX: z fastest
      const c00 = at(x0, y0, z0) * (1 - tx) + at(x0 + 1, y0, z0) * tx, c10 = at(x0, y0 + 1, z0) * (1 - tx) + at(x0 + 1, y0 + 1, z0) * tx;
      const c01 = at(x0, y0, z0 + 1) * (1 - tx) + at(x0 + 1, y0, z0 + 1) * tx, c11 = at(x0, y0 + 1, z0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1, z0 + 1) * tx;
      return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz;
    };
    for (let z = 0; z < dim[2]; z++) for (let y = 0; y < dim[1]; y++) for (let x = 0; x < dim[0]; x++) {
      const i = (z * dim[1] + y) * dim[0] + x;
      data[i * 2] = 1 - Math.exp(-sum[i] * 0.55);              // soft-saturated density 0..1
      if (dx) { const v = samp(bmin[0] + (x + 0.5) * vox, bmin[1] + (y + 0.5) * vox, bmin[2] + (z + 0.5) * vox); data[i * 2 + 1] = Math.min(1, Math.max(0, 0.5 + v / (2 * range))); }
      else data[i * 2 + 1] = 0.5;
    }
    if (this.volTex) gl.deleteTexture(this.volTex);
    const tex = gl.createTexture()!; this.volTex = tex;
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG16F, dim[0], dim[1], dim[2], 0, gl.RG, gl.FLOAT, data);
    this.bmin = bmin; this.bmax = bmax2; this.invDim = [1 / dim[0], 1 / dim[1], 1 / dim[2]];
    this.hasVol = true; this.hasPot = !!dx;
    const cx = (mn[0] + mx[0]) / 2, cy = (mn[1] + mx[1]) / 2, cz = (mn[2] + mx[2]) / 2;
    this.target = this.ligandCenter ?? [cx, cy, cz];
    this.R = Math.max(8, Math.max(...ext) * 0.5);
    this.dist = this.R * 1.65;
    this.dirty = true;
  }

  async loadTrajectory(t: Trajectory, finalSdfUrl: string | null, initialFrame?: number): Promise<void> {
    this.traj = t; this.finalMol = null; this.hasFinalMolecule = false;
    if (finalSdfUrl) {
      try { const txt = await (await fetch(finalSdfUrl)).text(); this.finalMol = parseSdf(txt); this.hasFinalMolecule = !!this.finalMol; } catch { this.finalMol = null; }
    }
    const last = t.frames[t.frames.length - 1];
    const c: V3 = [0, 0, 0]; last.forEach((p) => { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; });
    this.ligandCenter = [c[0] / last.length, c[1] / last.length, c[2] / last.length];
    this.target = this.ligandCenter;
    this.buildPocketPrims();
    this.frame = initialFrame !== undefined ? initialFrame : t.frames.length - 1;
    await this.setFrame(this.frame);
  }

  setLayers(l: Partial<NebulaViewer['layers']>) { this.layers = { ...this.layers, ...l }; this.refreshSpheres(); this.dirty = true; }

  async setFrame(i: number): Promise<void> { this.frame = i; this.refreshSpheres(); }

  private refreshSpheres() {
    const out = emptyPrims();
    const L = this.layers;
    const t = this.traj;
    if (L.atoms && this.pocketPrims) { // pocket residues as dim ball-and-stick
      const p = this.pocketPrims;
      out.sa.push(...p.sa); out.sb.push(...p.sb); out.ca.push(...p.ca); out.cb.push(...p.cb); out.cc.push(...p.cc);
    }
    if (t) {
      const atEnd = (this.frame >= t.frames.length - 1 || !L.trajectory) && !!this.finalMol;
      if (L.ligand) {
        if (atEnd && this.finalMol) {
          this.molPrims(this.finalMol.atoms, this.finalMol.bonds, 0.46, 0.2, 0.9, 1, out);
          if (L.hbonds) this.hbondPrims(this.finalMol.atoms, out);
        } else {
          const fr = t.frames[Math.min(this.frame, t.frames.length - 1)];
          const atoms: Atom[] = fr.map((p, k) => ({ p: p as V3, el: t.frame_elements[this.frame]?.[k] ?? t.elements[k] ?? 'C' }));
          this.molPrims(atoms, [], 0.42, 0.1, 1.1, 1, out);
        }
      }
    }
    this.uploadSpheres(out);
  }

  resetCamera() { this.yaw = 0.6; this.pitch = 0.25; this.dist = this.R * 1.65; this.dirty = true; }
  zoomIn() { this.dist = Math.max(this.R * 0.6, this.dist * 0.8); this.dirty = true; }
  zoomOut() { this.dist = Math.min(this.R * 8, this.dist * 1.25); this.dirty = true; }

  private render() {
    const gl = this.gl, c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * 0.85;
    const w = Math.max(2, Math.round(c.clientWidth * dpr)), h = Math.max(2, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.disposed) return;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cam: V3 = [this.target[0] + this.dist * cp * Math.sin(this.yaw), this.target[1] + this.dist * sp, this.target[2] + this.dist * cp * Math.cos(this.yaw)];
    const f = norm(sub(this.target, cam)); const r = norm(cross(f, [0, 1, 0])); const up = cross(r, f);
    const u = this.u;
    gl.useProgram(this.prog);
    gl.uniform2f(u.uRes, w, h);
    gl.uniform3fv(u.uCam, cam); gl.uniform3fv(u.uRight, r); gl.uniform3fv(u.uUp, up); gl.uniform3fv(u.uFwd, f);
    gl.uniform3fv(u.uBMin, this.bmin); gl.uniform3fv(u.uBMax, this.bmax); gl.uniform3fv(u.uTarget, this.target); gl.uniform3fv(u.uInvDim, this.invDim);
    gl.uniform1f(u.uTanF, Math.tan((36 * Math.PI) / 360));
    gl.uniform1f(u.uR, this.R);
    gl.uniform1f(u.uCut, this.layers.pocket ? this.R * 0.38 : 1e5);
    gl.uniform1f(u.uTime, (performance.now() - this.t0) / 1000);
    gl.uniform1f(u.uIso, 0.56);
    gl.uniform1f(u.uHasPot, this.hasPot ? 1 : 0);
    gl.uniform4f(u.uF, this.layers.surface && this.hasVol ? 1 : 0, this.layers.potential ? 1 : 0, 1, 1);
    gl.uniform1i(u.uN, this.nSph); gl.uniform1i(u.uNC, this.nCap);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.volTex); gl.uniform1i(u.uVol, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.saTex); gl.uniform1i(u.uSA, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.sbTex); gl.uniform1i(u.uSB, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.ctTex); gl.uniform1i(u.uCT, 3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.dirty = false;
  }

  dispose() { this.disposed = true; cancelAnimationFrame(this.raf); this.ro.disconnect(); this.canvas.remove(); this.gl.getExtension('WEBGL_lose_context')?.loseContext(); }
}

interface Prims { sa: number[]; sb: number[]; ca: number[]; cb: number[]; cc: number[] }
function emptyPrims(): Prims { return { sa: [], sb: [], ca: [], cb: [], cc: [] }; }
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
