#!/usr/bin/env bash
# =============================================================================
# fast_pipeline.sh — PocketLig End-to-End Pipeline for 10 Curated PDB Targets
# =============================================================================
# Runs 4 phases per target:
#   PHASE 1 (pocketlig env) : pocket extraction + protein cleaning
#   PHASE 2 (pocketforge env): TargetDiff 100-molecule diffusion sampling
#   PHASE 3 (pocketlig env) : SDF export → GNN eval → PoseBusters → GNINA → APBS
#   PHASE 4 (pocketlig env) : artifact validation
#
# Usage:
#   bash fast_pipeline.sh                     # all 4 phases on all 10 targets
#   bash fast_pipeline.sh --steps prepare     # prepare only
#   bash fast_pipeline.sh --targets 5r84,1hvr # subset of targets
#   bash fast_pipeline.sh --resume            # skip already-done targets
#
# The pipeline is idempotent: every step skips if its output already exists.
# Safe to Ctrl-C and rerun; chunks of 10 molecules save atomically via .tmp files.
# =============================================================================

set -euo pipefail

# ── Workspace paths ──────────────────────────────────────────────────────────
WORKSPACE="/home/abhin-krishna-m-p/pocketlig_workspace"
POCKETLIG="$WORKSPACE/pocketlig"
TARGETDIFF="$WORKSPACE/targetdiff"
POCKETLIG_PY="$POCKETLIG/.venv/bin/python"
POCKETFORGE_PY="/home/abhin-krishna-m-p/anaconda3/envs/pocketforge/bin/python"
GNN_CHECKPOINT="$POCKETLIG/runs/gnn_s0/best.pt"
SAMPLING_CONFIG="$TARGETDIFF/configs/sampling.yml"
ARTIFACTS="$POCKETLIG/artifacts"

# ── 10 Curated PDB Targets ───────────────────────────────────────────────────
# Format: "PDB_ID CHAIN LIGAND_RESNAME  # Protein Family — Notes"
declare -a TARGETS=(
  "3erk A SB4"   # MAP Kinase ERK2          — Signal transduction kinase
  "1stp A BTN"   # Streptavidin             — Biotin-binding protein (non-kinase)
  "2rgp A HYZ"   # B-Raf Kinase             — Oncology kinase target (PLX4720 / HYZ)
  "1hvr A XK2"   # HIV-1 Protease           — Viral aspartyl protease
  "4rlu A HCC"   # PARP1 inhibitor target   — DNA damage response (38Z / HCC)
  "1xap A TTB"   # Carbonic Anhydrase II    — Zinc metalloenzyme (AZM / TTB)
  "5r84 A GWS"   # SARS-CoV-2 Mpro          — Viral cysteine protease (GWS)
  "1fin A ATP"   # CDK2 (ATP-bound)         — Cell cycle kinase
  "4f3c A BIG"   # CDK2 alternative inhibitor — Selectivity benchmark (BIG)
  "1err A RAL"   # Estrogen Receptor α      — Nuclear hormone receptor
)

# ── CLI argument parsing ──────────────────────────────────────────────────────
PHASES="prepare,sample,export,eval,electro,validate"
TARGET_FILTER=""
NUM_SAMPLES=100
CHUNK=10
N_FRAMES=50
DEVICE="cuda:0"
RESUME=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)    PHASES="$2";        shift 2 ;;
    --targets)  TARGET_FILTER="$2"; shift 2 ;;
    --samples)  NUM_SAMPLES="$2";   shift 2 ;;
    --chunk)    CHUNK="$2";         shift 2 ;;
    --device)   DEVICE="$2";        shift 2 ;;
    --resume)   RESUME=true;        shift 1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE="$POCKETLIG/pipeline_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================="
echo " PocketLig Fast Pipeline  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
echo " Workspace : $WORKSPACE"
echo " Artifacts : $ARTIFACTS"
echo " Phases    : $PHASES"
echo " Targets   : ${TARGET_FILTER:-all 10}"
echo " Device    : $DEVICE"
echo " Log file  : $LOG_FILE"
echo "=============================================================="

mkdir -p "$ARTIFACTS"

# ── Helper: check if target is in filter ─────────────────────────────────────
should_run() {
  local pdb_id="$1"
  [[ -z "$TARGET_FILTER" ]] && return 0
  [[ ",$TARGET_FILTER," == *",$pdb_id,"* ]] && return 0
  return 1
}

# ── Per-target tracking ───────────────────────────────────────────────────────
declare -A PHASE1_STATUS PHASE2_STATUS PHASE3_STATUS PHASE4_STATUS
PIPELINE_START=$(date +%s)

# =============================================================================
# PHASE 1: Pocket Extraction & Protein Cleaning (pocketlig .venv)
# =============================================================================
if [[ "$PHASES" == *"prepare"* ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " PHASE 1: Pocket Extraction & Protein Cleaning"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for t in "${TARGETS[@]}"; do
    read -r pdb_id chain resname <<< "$t"
    should_run "$pdb_id" || continue

    OUT="$ARTIFACTS/$pdb_id"
    echo ""
    echo "▶ [$pdb_id] Preparing pocket (chain=$chain, ligand=$resname)..."

    if [[ "$RESUME" == true ]] && [[ -f "$OUT/pocket.pdb" ]]; then
      echo "  ↳ pocket.pdb exists — skipping (--resume)"
      PHASE1_STATUS[$pdb_id]="SKIPPED"
      continue
    fi

    T0=$(date +%s)
    if "$POCKETLIG_PY" -m pocketlig.precompute.run_target \
        "$pdb_id" \
        --ligand-resname "$resname" \
        --chain "$chain" \
        --out "$OUT" \
        --steps prepare; then
      T1=$(date +%s)
      echo "  ✓ [$pdb_id] Prepared in $((T1-T0))s → $OUT/pocket.pdb"
      PHASE1_STATUS[$pdb_id]="OK"
    else
      echo "  ✗ [$pdb_id] Preparation FAILED (exit $?)"
      PHASE1_STATUS[$pdb_id]="FAILED"
    fi
  done
fi

# =============================================================================
# PHASE 2: Diffusion Sampling — 100 molecules per target (pocketforge / targetdiff)
# =============================================================================
if [[ "$PHASES" == *"sample"* ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " PHASE 2: TargetDiff Diffusion Sampling (100 mol × 10 targets)"
  echo " Checkpoint: $SAMPLING_CONFIG"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for t in "${TARGETS[@]}"; do
    read -r pdb_id chain resname <<< "$t"
    should_run "$pdb_id" || continue

    OUT="$ARTIFACTS/$pdb_id"
    POCKET="$OUT/pocket.pdb"
    TRAJ="$OUT/trajectories"

    if [[ ! -f "$POCKET" ]]; then
      echo "  ✗ [$pdb_id] pocket.pdb missing — run prepare phase first"
      PHASE2_STATUS[$pdb_id]="NO_POCKET"
      continue
    fi

    mkdir -p "$TRAJ"
    # Count already-done chunks safely without pipefail error
    DONE_CHUNKS=$(find "$TRAJ" -name "frames.npz" 2>/dev/null | wc -l || true)
    if [[ "$RESUME" == true ]] && [[ "$DONE_CHUNKS" -ge "$NUM_SAMPLES" ]]; then
      echo "  ↳ [$pdb_id] $DONE_CHUNKS/$NUM_SAMPLES chunks done — skipping (--resume)"
      PHASE2_STATUS[$pdb_id]="SKIPPED"
      continue
    fi

    echo ""
    echo "▶ [$pdb_id] Sampling $NUM_SAMPLES molecules (chunk=$CHUNK, done=$DONE_CHUNKS)..."
    T0=$(date +%s)

    if (cd "$TARGETDIFF" && PYTHONPATH="$POCKETLIG/src:$TARGETDIFF:." \
       "$POCKETFORGE_PY" \
         "$POCKETLIG/scripts/targetdiff_sample_pocket.py" \
         "$SAMPLING_CONFIG" \
         --pdb_path "$POCKET" \
         --out_dir "$TRAJ" \
         --num_samples "$NUM_SAMPLES" \
         --chunk "$CHUNK" \
         --n_frames "$N_FRAMES" \
         --device "$DEVICE"); then
      T1=$(date +%s)
      N_DONE=$(find "$TRAJ" -name "frames.npz" 2>/dev/null | wc -l || true)
      echo "  ✓ [$pdb_id] Sampling done in $((T1-T0))s — $N_DONE mol_* folders written"
      PHASE2_STATUS[$pdb_id]="OK($N_DONE)"
    else
      echo "  ✗ [$pdb_id] Sampling FAILED (exit $?)"
      PHASE2_STATUS[$pdb_id]="FAILED"
    fi
  done
fi

# =============================================================================
# PHASE 3: Export + Evaluation (pocketlig env)
#   - Export NPZ → frames.json for web viewer
#   - GNN affinity scoring (pocketlig GNN ensemble)
#   - PoseBusters geometric validity checks
#   - GNINA CNN scoring (Vina-score + CNNaffinity)
#   - APBS electrostatics (pdb2pqr → surface_potential.dx)
# =============================================================================
EVAL_STEPS="export"
[[ "$PHASES" == *"eval"* ]]   && EVAL_STEPS="$EVAL_STEPS,eval"
[[ "$PHASES" == *"electro"* ]] && EVAL_STEPS="$EVAL_STEPS,electro"

if [[ "$EVAL_STEPS" != "export" ]] || [[ "$PHASES" == *"export"* ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " PHASE 3: Export + Evaluation (steps: $EVAL_STEPS)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for t in "${TARGETS[@]}"; do
    read -r pdb_id chain resname <<< "$t"
    should_run "$pdb_id" || continue

    OUT="$ARTIFACTS/$pdb_id"
    TRAJ="$OUT/trajectories"
    PDBQT_CACHE="$POCKETLIG/data/pdbqt_cache/${pdb_id}.pdbqt"

    N_TRAJ=$(find "$TRAJ" -name "frames.npz" 2>/dev/null | wc -l || true)
    if [[ "$N_TRAJ" -eq 0 ]]; then
      echo "  ✗ [$pdb_id] no trajectories found — skipping evaluation"
      PHASE3_STATUS[$pdb_id]="NO_TRAJ"
      continue
    fi

    echo ""
    echo "▶ [$pdb_id] Evaluating $N_TRAJ trajectories (steps: $EVAL_STEPS)..."
    T0=$(date +%s)

    ARGS=("$pdb_id" --ligand-resname "$resname" --chain "$chain" --out "$OUT" --steps "$EVAL_STEPS")
    [[ -f "$GNN_CHECKPOINT" ]] && ARGS+=(--gnn "$GNN_CHECKPOINT")
    [[ -f "$PDBQT_CACHE" ]]    && ARGS+=(--receptor-pdbqt "$PDBQT_CACHE")

    if "$POCKETLIG_PY" -m pocketlig.precompute.run_target "${ARGS[@]}"; then
      T1=$(date +%s)
      # Parse n_valid from metrics.json
      N_VALID=$(python3 -c "
import json, pathlib
m = pathlib.Path('$OUT/metrics.json')
if m.exists():
    d = json.loads(m.read_text())
    print(d.get('n_valid', '?'))
else:
    print('?')
" 2>/dev/null || echo "?")
      echo "  ✓ [$pdb_id] Eval done in $((T1-T0))s — n_valid=$N_VALID"
      PHASE3_STATUS[$pdb_id]="OK(valid=$N_VALID)"
    else
      echo "  ✗ [$pdb_id] Evaluation FAILED (exit $?)"
      PHASE3_STATUS[$pdb_id]="FAILED"
    fi
  done
fi

# =============================================================================
# PHASE 4: Artifact Validation
# =============================================================================
if [[ "$PHASES" == *"validate"* ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " PHASE 4: Artifact Validation"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  T0=$(date +%s)
  if "$POCKETLIG_PY" -m pocketlig.precompute.validate_artifacts "$ARTIFACTS"; then
    echo "  ✓ All artifacts validated OK"
  else
    echo "  ✗ Some artifacts have issues — review output above"
  fi
  echo "  Validation completed in $(($(date +%s)-T0))s"
fi

# =============================================================================
# Final Summary
# =============================================================================
PIPELINE_END=$(date +%s)
TOTAL=$((PIPELINE_END - PIPELINE_START))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Pipeline Complete  —  Total time: ${TOTAL}s  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TARGET           | PHASE1   | PHASE2              | PHASE3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for t in "${TARGETS[@]}"; do
  read -r pdb_id chain resname <<< "$t"
  should_run "$pdb_id" || continue
  P1="${PHASE1_STATUS[$pdb_id]:-—}"
  P2="${PHASE2_STATUS[$pdb_id]:-—}"
  P3="${PHASE3_STATUS[$pdb_id]:-—}"
  printf " %-16s | %-8s | %-19s | %s\n" "$pdb_id" "$P1" "$P2" "$P3"
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Full log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  Terminal 1: cd $POCKETLIG && make api    # serve FastAPI backend"
echo "  Terminal 2: cd $POCKETLIG/web && npm install && npm run dev  # React viewer"
echo "  Open:       http://localhost:5173"
