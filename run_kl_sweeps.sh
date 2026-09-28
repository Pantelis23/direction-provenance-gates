#!/usr/bin/env bash
# Prereg addendum B: KL-matched sweeps, GPT-2 then Qwen3-0.6B.
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
LOG=runs/kl_sweeps.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
G=configs/mvp_jobs56_pronouns.yaml; Q=configs/qwen3_0p6b_jobs56_pronouns.yaml
for spec in "mid:runs/grid_learn_s1337_mid_20260221_081426/control_dir.npy" \
            "early:runs/grid_learn_s1337_early_20260221_081121/control_dir.npy" \
            "late:runs/grid_learn_s1337_late_20260221_081734/control_dir.npy"; do
  L=${spec%%:*}; D=${spec#*:}
  step "KL GPT-2 $L"
  CONFIG=$G PREFIX=klg ./run_mc16_kl.sh "$L" "$D" "runs/kl_gpt2_$L" >/dev/null 2>&1 || step "KL GPT-2 $L FAILED"
done
for L in 14 1 26; do
  D=$(ls -d runs/qwen_learn_s1337_L${L}_* | head -1)/control_dir.npy
  step "KL Qwen L$L"
  CONFIG=$Q PREFIX=klq ./run_mc16_kl.sh "$L" "$D" "runs/kl_qwen_L$L" >/dev/null 2>&1 || step "KL Qwen L$L FAILED"
done
step "ALL DONE"
