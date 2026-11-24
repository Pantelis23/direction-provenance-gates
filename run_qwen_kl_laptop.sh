#!/usr/bin/env bash
# Laptop (WSL2, RTX 5080) continuation of prereg addendum B for Qwen3-0.6B.
set -uo pipefail
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_OFFLINE=1
# torch 2.14 routes some ops to Triton kernels that need python3.12-dev headers to build; use ATen instead.
export TORCH_DISABLE_NATIVE_JIT=1
cd ~/bias_embedding_exp
cp /mnt/c/Users/pante/Desktop/bias_exp_transfer/run_mc16_kl.sh ./run_mc16_kl.sh
source .venv/bin/activate
LOG=runs/kl_sweeps_laptop.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
step "host=$(hostname) torch=$(python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))')"
Q=configs/qwen3_0p6b_jobs56_pronouns.yaml
D14=$(ls -d runs/qwen_learn_s1337_L14_*)/control_dir.npy
step "cross-host check: Qwen L14 seed 1 (desktop result exists)"
SEEDS=1 CONFIG=$Q PREFIX=xhq ./run_mc16_kl.sh 14 "$D14" runs/xhost_kl_qwen_L14 >/dev/null 2>&1 || step "cross-host FAILED"
n=$(grep -c "FAILED" runs/xhost_kl_qwen_L14/master.log 2>/dev/null || true); [ "${n:-0}" != "0" ] && { step "cross-host: $n seed(s) FAILED - stopping"; exit 1; }
for L in 14 1 26; do
  D=$(ls -d runs/qwen_learn_s1337_L${L}_*)/control_dir.npy
  step "KL Qwen L$L"
  CONFIG=$Q PREFIX=klq ./run_mc16_kl.sh "$L" "$D" "runs/kl_qwen_L$L" >/dev/null 2>&1 || step "KL Qwen L$L FAILED"
  n=$(grep -c "FAILED" "runs/kl_qwen_L$L/master.log" 2>/dev/null || true); [ "${n:-0}" != "0" ] && step "KL Qwen L$L: $n seed(s) FAILED"
done
step "ALL DONE"
