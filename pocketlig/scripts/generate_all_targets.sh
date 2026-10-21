#!/usr/bin/env bash
# =============================================================================
# generate_all_targets.sh — Generate 10 Ligands for All 10 Target Proteins
# =============================================================================
# For each target:
#   1. Samples 10 molecules (chunk=10) via TargetDiff on CPU
#   2. Exports frames.json + final.sdf for the web viewer
#   3. Evaluates GNN affinity, PoseBusters validity, QED, SA score -> metrics.json
#
# Fully resumable: skips any target or chunk already completed.
# =============================================================================

set -uo pipefail

WORKSPACE="/home/abhin-krishna-m-p/pocketlig_workspace"
POCKETLIG="/home/abhin-krishna-m-p/pocketlig"
TARGETDIFF="$WORKSPACE/targetdiff"
POCKETFORGE_PY="/home/abhin-krishna-m-p/anaconda3/envs/pocketforge/bin/python"
POCKETLIG_PY="$POCKETLIG/.venv/bin/python"
GNN_CHECKPOINT="$POCKETLIG/runs/gnn_s0/best.pt"
SAMPLING_CONFIG="$TARGETDIFF/configs/sampling.yml"
ARTIFACTS="$POCKETLIG/artifacts"
LOG_FILE="$POCKETLIG/pipeline_generation.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================================="
echo " Generating 10 Ligands for 10 Targets — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

declare -a TARGETS=(
  "5r84 A GWS"   # SARS-CoV-2 Mpro
  "3erk A SB4"   # MAP Kinase ERK2
  "1fin A ATP"   # CDK2 / ATP
  "1err A RAL"   # Estrogen Receptor α
  "2rgp A HYZ"   # B-Raf Kinase
  "4rlu A HCC"   # PARP1
  "1xap A TTB"   # Carbonic Anhydrase II
  "4f3c A BIG"   # CDK2 Inhibitor
  "1stp A BTN"   # Streptavidin
  "1hvr A XK2"   # HIV-1 Protease (already complete)
)

TOTAL_START=$(date +%s)

for item in "${TARGETS[@]}"; do
  read -r pdb_id chain resname <<< "$item"
  OUT="$ARTIFACTS/$pdb_id"
  POCKET="$OUT/pocket.pdb"
  TRAJ="$OUT/trajectories"
  METRICS="$OUT/metrics.json"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Target: $pdb_id (Chain: $chain, Ligand: $resname)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ! -f "$POCKET" ]]; then
    echo "  [ERROR] pocket.pdb not found for $pdb_id. Skipping."
    continue
  fi

  # Check if already fully evaluated
  N_DONE=$(find "$TRAJ" -name "frames.json" 2>/dev/null | wc -l || true)
  if [[ -f "$METRICS" ]] && [[ "$N_DONE" -ge 10 ]]; then
    echo "  [SKIP] Target $pdb_id already has $N_DONE molecules + metrics.json."
    continue
  fi

  T0=$(date +%s)
  mkdir -p "$TRAJ"

  # Step 1: Diffusion Sampling (10 molecules, chunk=10, 50 frames, device=cpu)
  echo "  ▶ Step 1: TargetDiff Diffusion Sampling (10 molecules)..."
  (
    cd "$TARGETDIFF"
    PYTHONPATH="$POCKETLIG/src:$TARGETDIFF:." \
    "$POCKETFORGE_PY" \
      "$POCKETLIG/scripts/targetdiff_sample_pocket.py" \
      "$SAMPLING_CONFIG" \
      --pdb_path "$POCKET" \
      --out_dir "$TRAJ" \
      --num_samples 10 \
      --chunk 10 \
      --n_frames 50 \
      --device cuda:0
  )

  # Step 2: Export & Evaluation (frames.json, final.sdf, GNN affinity, metrics.json)
  echo "  ▶ Step 2: Export to JSON/SDF & Evaluate Multi-Objective Metrics..."
  PDBQT_CACHE="$POCKETLIG/data/pdbqt_cache/${pdb_id}.pdbqt"
  ARGS=("$pdb_id" --ligand-resname "$resname" --chain "$chain" --out "$OUT" --steps "export,eval")
  [[ -f "$GNN_CHECKPOINT" ]] && ARGS+=(--gnn "$GNN_CHECKPOINT")
  [[ -f "$PDBQT_CACHE" ]]    && ARGS+=(--receptor-pdbqt "$PDBQT_CACHE")

  "$POCKETLIG_PY" -m pocketlig.precompute.run_target "${ARGS[@]}"

  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  echo "  ✓ Target $pdb_id COMPLETE in ${ELAPSED}s ($(date '+%Y-%m-%d %H:%M:%S'))"
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))
echo ""
echo "=============================================================="
echo " All 10 Targets Generation Finished in ${TOTAL_TIME}s"
echo "=============================================================="
