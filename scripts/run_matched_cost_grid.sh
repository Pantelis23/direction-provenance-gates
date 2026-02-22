#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CFG="${CFG:-configs/mvp_jobs56_pronouns.yaml}"
SRC_RUN="${SRC_RUN:-runs/mvp_jobs56_pronouns_20260220_232956}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.3}"
SEEDS="${SEEDS:-1337 2025 9001}"
LAYERS="${LAYERS:-early mid late}"
CTRLS="${CTRLS:-none,random,shuffled}"
TARGETS="${TARGETS:--0.01 -0.02}"
CAL_ITERS="${CAL_ITERS:-6}"
CAL_BOOT="${CAL_BOOT:-0}"
FINAL_BOOT="${FINAL_BOOT:-10000}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
CAL_MODE="${CAL_MODE:-bisection}"
ALPHA_PROBE="${ALPHA_PROBE:-1.0}"
MATCH_TOL="${MATCH_TOL:-auto}"
MATCH_CHECK_BOOT="${MATCH_CHECK_BOOT:-0}"
FALLBACK_ON_UNMATCHED="${FALLBACK_ON_UNMATCHED:-bisection}"
RUN_TAG="${RUN_TAG:-}"
INTERVENTION_SCOPE="${INTERVENTION_SCOPE:-name}"

name_prefix="mc"
learn_prefix="mc_learn"
if [[ -n "$RUN_TAG" ]]; then
  name_prefix="${name_prefix}_${RUN_TAG}"
  learn_prefix="${learn_prefix}_${RUN_TAG}"
fi

for seed in $SEEDS; do
  for layer in $LAYERS; do
    latest_learn=$(ls -1dt runs/${learn_prefix}_s${seed}_${layer}_* 2>/dev/null | head -1 || true)
    if [[ -n "$latest_learn" && -f "$latest_learn/control_dir.npy" ]]; then
      echo "[learn-skip] seed=$seed layer=$layer use=$latest_learn"
      dpath="$latest_learn/control_dir.npy"
    else
      echo "[learn] seed=$seed layer=$layer"
      learn_run=$(.venv/bin/python scripts/learn_control_dir.py \
        --config "$CFG" \
        --layer "$layer" \
        --holdout-frac "$HOLDOUT_FRAC" \
        --holdout-seed "$seed" \
        --ridge-lambda "$RIDGE_LAMBDA" \
        --target adj \
        --run-name "${learn_prefix}_s${seed}_${layer}" \
        | sed -n 's/^Run complete: //p' | tail -1)
      dpath="$learn_run/control_dir.npy"
    fi

    for tgt in $TARGETS; do
      if [[ "$MATCH_TOL" == "auto" ]]; then
        tol=$(awk -v t="$tgt" 'BEGIN{v=t+0; if(v<0) v=-v; m=0.15*v; if(m<0.002) m=0.002; printf "%.6f", m}')
      else
        tol="$MATCH_TOL"
      fi
      existing=$(ls -1dt runs/${name_prefix}_s${seed}_${layer}_t${tgt}_*/matched_cost_summary.json 2>/dev/null | head -1 || true)
      if [[ -n "$existing" ]]; then
        echo "[match-skip] seed=$seed layer=$layer target=$tgt have=$existing"
        continue
      fi
      echo "[match] seed=$seed layer=$layer target=$tgt tol=$tol"
      .venv/bin/python scripts/run_causal_matched_cost.py \
        --config "$CFG" \
        --source-run "$SRC_RUN" \
        --d-path "$dpath" \
        --layer "$layer" \
        --mode project_out \
        --alpha-policy same \
        --intervention-scope "$INTERVENTION_SCOPE" \
        --filter-template gender_names_jobs \
        --eval-split test \
        --holdout-frac "$HOLDOUT_FRAC" \
        --holdout-seed "$seed" \
        --controls "$CTRLS" \
        --target-logprob-change "$tgt" \
        --calibration-mode "$CAL_MODE" \
        --alpha-min 0.05 \
        --alpha-max 4.0 \
        --alpha-probe "$ALPHA_PROBE" \
        --search-iters "$CAL_ITERS" \
        --calib-bootstrap-samples "$CAL_BOOT" \
        --match-check-bootstrap-samples "$MATCH_CHECK_BOOT" \
        --final-bootstrap-samples "$FINAL_BOOT" \
        --match-tol "$tol" \
        --fallback-on-unmatched "$FALLBACK_ON_UNMATCHED" \
        --run-name "${name_prefix}_s${seed}_${layer}_t${tgt}"
    done
  done
done
