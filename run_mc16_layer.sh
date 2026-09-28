#!/usr/bin/env bash
# Matched-cost 16-seed x 3-control sweep at one layer.
# usage: [CONFIG=...] [PREFIX=...] ./run_mc16_layer.sh <early|mid|late|N> <d_path> <outdir>
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAYER="$1"; DPATH="$2"; OUTDIR="$3"
CONFIG="${CONFIG:-configs/mvp_jobs56_pronouns.yaml}"; PREFIX="${PREFIX:-mcl}"
mkdir -p "$OUTDIR"
exec > >(tee -a "$OUTDIR/master.log") 2>&1
for s in $(seq 1 16); do
  OUT="$OUTDIR/seed${s}.matched_cost_summary.json"
  [ -f "$OUT" ] && { echo "skip seed $s"; continue; }
  NAME="${PREFIX}_${LAYER}_s${s}"
  echo "[$(date -Is)] launch layer=$LAYER seed=$s"
  if ! .venv/bin/python -u scripts/run_causal_matched_cost.py \
    --config "$CONFIG" \
    --source-run runs/mvp_jobs56_pronouns_20260220_232956 \
    --d-path "$DPATH" \
    --layer "$LAYER" --mode project_out --alpha-policy same --intervention-scope non_name \
    --controls orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512 \
    --control-seed 2027 --direction-seed "$s" \
    --target-pll-change auto --auto-target-frac 0.05 --match-tol auto \
    --calibration-mode probe --alpha-probe 5.0 --alpha-min 0 --alpha-max 40 \
    --search-iters 6 --calib-bootstrap-samples 200 --match-check-bootstrap-samples 200 \
    --final-bootstrap-samples 5000 \
    --holdout-frac 0.05 --holdout-seed 1337 \
    --run-name "$NAME" > "$OUTDIR/seed${s}.log" 2>&1; then
    echo "seed $s FAILED"; continue
  fi
  R=$(ls -1dt runs/${NAME}_2* 2>/dev/null | head -1 || true)
  if [ -z "$R" ] || [ ! -f "$R/matched_cost_summary.json" ]; then echo "seed $s FAILED (no summary)"; continue; fi
  cp "$R/matched_cost_summary.json" "$OUT"
  echo "[$(date -Is)] done layer=$LAYER seed=$s"
done
echo "DONE layer $LAYER"
