#!/usr/bin/env bash
set +e

cd /home/pantelis/Desktop/Projects/Work/bias_embedding_exp || exit 1
mkdir -p runs/mc_shuf_matched_mcK16_seeds

fail_count=0

for s in $(seq 1 16); do
  OUT="runs/mc_shuf_matched_mcK16_seeds/seed${s}.matched_cost_summary.json"
  if [ -f "$OUT" ]; then
    echo "skip $s"
    continue
  fi

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
    --final-bootstrap-samples 2000 --holdout-frac 0.05 --holdout-seed 1337 \
    --run-name "mc_shuf_matched_mcK16_s${s}" \
    2>&1 | tee "runs/mc_shuf_matched_mcK16_seeds/seed${s}.log"

  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "seed $s failed rc=$rc (continuing)"
    fail_count=$((fail_count+1))
    continue
  fi

  R=$(ls -1dt runs/mc_shuf_matched_mcK16_s${s}_* | head -1)
  if [ -f "$R/matched_cost_summary.json" ]; then
    cp "$R/matched_cost_summary.json" "$OUT"
    echo "seed $s saved -> $OUT"
  else
    echo "seed $s missing matched_cost_summary.json"
    fail_count=$((fail_count+1))
  fi
done

echo "DONE. fail_count=$fail_count"
