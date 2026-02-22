#!/usr/bin/env bash
set -euo pipefail

cd /home/pantelis/Desktop/Projects/Work/bias_embedding_exp
mkdir -p runs/mc_orth_vs_supportavoid_mc8
MASTER_LOG="runs/mc_orth_vs_supportavoid_mc8/master.log"
touch "$MASTER_LOG"
exec > >(tee -a "$MASTER_LOG") 2>&1
set -x
echo "===== START $(date -Is) ====="

for s in $(seq 1 8); do
  OUT="runs/mc_orth_vs_supportavoid_mc8/seed${s}.matched_cost_summary.json"
  SEED_LOG="runs/mc_orth_vs_supportavoid_mc8/seed${s}.log"
  if [ -f "$OUT" ]; then
    echo "skip seed $s (exists)"
    continue
  fi

  echo "==== seed $s ===="
  : > "$SEED_LOG"
  echo "[$(date -Is)] launch seed=$s" | tee -a "$SEED_LOG"
  set +e
  stdbuf -oL -eL env PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_causal_matched_cost.py \
    --config configs/mvp_jobs56_pronouns.yaml \
    --source-run runs/mvp_jobs56_pronouns_20260220_232956 \
    --d-path runs/grid_learn_s1337_late_20260221_081734/control_dir.npy \
    --layer late --mode project_out --alpha-policy same --intervention-scope non_name \
    --controls orthogonal_random,orthogonal_random_support_avoid_top512 \
    --control-seed 2027 --direction-seed "$s" \
    --target-pll-change auto --auto-target-frac 0.05 --match-tol auto \
    --calibration-mode probe --alpha-probe 5.0 --alpha-min 0 --alpha-max 40 \
    --search-iters 6 --calib-bootstrap-samples 200 --match-check-bootstrap-samples 200 \
    --final-bootstrap-samples 5000 \
    --holdout-frac 0.05 --holdout-seed 1337 \
    --run-name "mc_orth_vs_supportavoid_s${s}" \
    2>&1 | tee -a "$SEED_LOG"
  rc=${PIPESTATUS[0]}
  set -e

  echo "seed $s rc=$rc"
  echo "[$(date -Is)] done seed=$s rc=$rc" | tee -a "$SEED_LOG"
  if [ "$rc" -ne 0 ]; then
    echo "seed $s FAILED (see runs/mc_orth_vs_supportavoid_mc8/seed${s}.log)"
    continue
  fi

  R=$(ls -1dt runs/mc_orth_vs_supportavoid_s${s}_* 2>/dev/null | head -1 || true)
  if [ -z "$R" ] || [ ! -f "$R/matched_cost_summary.json" ]; then
    echo "seed $s FAILED (missing matched_cost_summary.json; latest run dir: ${R:-none})"
    continue
  fi
  cp "$R/matched_cost_summary.json" "$OUT"
  echo "seed $s saved -> $OUT"
done

echo "DONE mc8 sweep"
