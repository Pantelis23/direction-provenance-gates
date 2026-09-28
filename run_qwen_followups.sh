#!/usr/bin/env bash
# H5 (docs/prereg_2026-09-28.md, Qwen addendum): Qwen3-0.6B replication.
# Starts after the GPT-2 follow-up queue has finished.
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
CFG=configs/qwen3_0p6b_jobs56_pronouns.yaml
LOG=runs/qwen_followups.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

until grep -q "ALL DONE" runs/followups.log 2>/dev/null; do sleep 60; done
step "GPT-2 queue done; starting Qwen3-0.6B"

declare -A D
for L in 1 14 26; do
  step "learn source direction L$L"
  out=$(.venv/bin/python -u scripts/learn_control_dir.py --config "$CFG" --run-name "qwen_learn_s1337_L$L" \
        --layer "$L" --pool mean --target adj --holdout-frac 0.3 --holdout-seed 1337 --ridge-lambda 1.0 2>&1 | tee -a runs/qwen_learn.log)
  dir=$(printf '%s\n' "$out" | sed -n 's/^Run complete: //p' | tail -1)
  if [ -z "$dir" ] || [ ! -f "$dir/control_dir.npy" ]; then step "learn L$L FAILED"; continue; fi
  D[$L]="$dir/control_dir.npy"
  step "L$L source = ${D[$L]}"
done

for L in 26 14 1; do
  [ -n "${D[$L]:-}" ] || continue
  step "P1 L$L"
  .venv/bin/python -u scripts/run_unit_projection.py --config "$CFG" --run-prefix qp1 --layer "$L" \
    --d-path "${D[$L]}" --out-dir "runs/qwen_p1_L$L" >> "runs/qwen_p1_L$L.log" 2>&1 || step "P1 L$L FAILED"
done

for L in 26 14 1; do
  [ -n "${D[$L]:-}" ] || continue
  step "MC L$L"
  CONFIG="$CFG" PREFIX=qmc ./run_mc16_layer.sh "$L" "${D[$L]}" "runs/qwen_mc16_L$L" >/dev/null 2>&1 || step "MC L$L FAILED"
done
step "ALL DONE"
