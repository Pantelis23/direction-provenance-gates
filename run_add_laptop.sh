#!/usr/bin/env bash
# Prereg addendum C: additive steering, GPT-2 and Qwen3-0.6B (run on the laptop, WSL2/CUDA).
set -uo pipefail
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
cd ~/bias_embedding_exp && source .venv/bin/activate
LOG=runs/add_sweeps_laptop.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
G=configs/mvp_jobs56_pronouns.yaml; Q=configs/qwen3_0p6b_jobs56_pronouns.yaml
GE=runs/grid_learn_s1337_early_20260221_081121/control_dir.npy
GM=runs/grid_learn_s1337_mid_20260221_081426/control_dir.npy
GL=runs/grid_learn_s1337_late_20260221_081734/control_dir.npy
Q1=$(ls -d runs/qwen_learn_s1337_L1_*)/control_dir.npy
Q14=$(ls -d runs/qwen_learn_s1337_L14_*)/control_dir.npy
Q26=$(ls -d runs/qwen_learn_s1337_L26_*)/control_dir.npy
step "host=$(hostname) torch=$(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
# model|config|layer arg|tag|source|ADD-N alpha|ADD-KL alpha_max|ADD-KL probe
SPECS=(
 "gpt2|$G|early|1|$GE|2.797|128|16"
 "gpt2|$G|mid|6|$GM|3.955|128|16"
 "gpt2|$G|late|11|$GL|8.455|512|64"
 "qwen|$Q|1|1|$Q1|0.221|32|4"
 "qwen|$Q|14|14|$Q14|2.834|128|16"
 "qwen|$Q|26|26|$Q26|8.969|512|64"
)
for s in "${SPECS[@]}"; do
  IFS='|' read -r M C LA TAG D AN AMAX APROBE <<< "$s"
  step "ADD-N $M L$TAG alpha=$AN"
  python -u scripts/run_unit_projection.py --config "$C" --run-prefix "addn_$M" --layer "$LA" --d-path "$D" \
    --mode add --alpha "$AN" --out-dir "runs/addn_${M}_L$TAG" >> "runs/addn_${M}_L$TAG.log" 2>&1 || step "ADD-N $M L$TAG FAILED"
done
for s in "${SPECS[@]}"; do
  IFS='|' read -r M C LA TAG D AN AMAX APROBE <<< "$s"
  step "ADD-KL $M L$TAG alpha_max=$AMAX probe=$APROBE"
  MODE=add ALPHA_MAX=$AMAX ALPHA_PROBE=$APROBE CONFIG=$C PREFIX="addkl_$M" ./run_mc16_kl.sh "$LA" "$D" "runs/addkl_${M}_L$TAG" >/dev/null 2>&1 || step "ADD-KL $M L$TAG FAILED"
  n=$(grep -c "FAILED" "runs/addkl_${M}_L$TAG/master.log" 2>/dev/null || true); [ "${n:-0}" != "0" ] && step "ADD-KL $M L$TAG: $n seed(s) FAILED"
done
step "ALL DONE"
