#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(2 7)
fi

for s in "${SEEDS[@]}"; do
  echo "==== seed $s ===="
  env PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_causal_matched_cost.py \
    --config configs/mvp_jobs56_pronouns.yaml \
    --source-run runs/mvp_jobs56_pronouns_20260220_232956 \
    --d-path runs/grid_learn_s1337_late_20260221_081734/control_dir.npy \
    --layer late --mode project_out --alpha-policy same --intervention-scope non_name \
    --controls none,orthogonal_random,abs_marginal_matched_random_orth,shuffled \
    --control-seed 2027 --direction-seed "$s" \
    --target-pll-change auto --auto-target-frac 0.05 --match-tol auto \
    --calibration-mode probe --alpha-probe 5.0 --alpha-min 0 --alpha-max 40 \
    --search-iters 6 --calib-bootstrap-samples 200 --match-check-bootstrap-samples 200 \
    --final-bootstrap-samples 20000 \
    --holdout-frac 0.05 --holdout-seed 1337 \
    --run-name "mc_shuf_matched_mcK16_hiBS_s${s}" \
    2>&1 | tee "runs/mc_shuf_matched_mcK16_hiBS_s${s}.log"

  R="$(ls -1dt runs/mc_shuf_matched_mcK16_hiBS_s${s}_* | head -1)"
  echo "=== seed $s => $R ==="
  jq -r '.results[] |
    [.control,.matched,.matched_alpha,.cost_err,
     (.final_summary.mean_abs_reduction_adj//null),
     ((.final_summary.block_bootstrap_abs_reduction_adj.ci95//[null,null])[0]),
     ((.final_summary.block_bootstrap_abs_reduction_adj.ci95//[null,null])[1])] | @tsv' \
    "$R/matched_cost_summary.json" | column -ts $'\t'
done
