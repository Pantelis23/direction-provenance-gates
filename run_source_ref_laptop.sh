#!/usr/bin/env bash
# Post hoc reference runs (NOT pre-registered): the learned source direction itself
# (--direction-control none) under every condition its random controls were run in.
set -uo pipefail
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
cd ~/bias_embedding_exp && source .venv/bin/activate
LOG=runs/source_ref_laptop.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
until grep -q "ALL DONE" runs/llama_laptop.log 2>/dev/null; do sleep 30; done
G=configs/mvp_jobs56_pronouns.yaml; Q=configs/qwen3_0p6b_jobs56_pronouns.yaml; LL=configs/llama32_1b_jobs56_pronouns.yaml
d() { ls -d runs/$1 | head -1; }
# model|config|layer arg|tag|source dir|ADD-N alpha|ADD-KL amax|ADD-KL probe|KL amax
SPECS=(
 "gpt2|$G|early|1|$(d 'grid_learn_s1337_early_*')|2.797|128|16|100"
 "gpt2|$G|mid|6|$(d 'grid_learn_s1337_mid_*')|3.955|128|16|100"
 "gpt2|$G|late|11|$(d 'grid_learn_s1337_late_*')|8.455|512|64|100"
 "qwen|$Q|1|1|$(d 'qwen_learn_s1337_L1_*')|0.221|32|4|100"
 "qwen|$Q|14|14|$(d 'qwen_learn_s1337_L14_*')|2.834|128|16|100"
 "qwen|$Q|26|26|$(d 'qwen_learn_s1337_L26_*')|8.969|512|64|100"
 "llama|$LL|1|1|$(d 'llama_learn_s1337_L1_*')|-|-|-|200"
 "llama|$LL|8|8|$(d 'llama_learn_s1337_L8_*')|-|-|-|200"
 "llama|$LL|14|14|$(d 'llama_learn_s1337_L14_*')|-|-|-|200"
)
for s in "${SPECS[@]}"; do
  IFS='|' read -r M C LA TAG SD AN AAM AAP KAM <<< "$s"; D="$SD/control_dir.npy"
  step "SRC $M L$TAG P1"
  python -u scripts/run_unit_projection.py --config "$C" --run-prefix "srcp1_$M" --layer "$LA" --d-path "$D" --controls none --seeds 1-1 --out-dir "runs/src_p1_${M}_L$TAG" >> runs/source_ref.log 2>&1 || step "SRC $M L$TAG P1 FAILED"
  step "SRC $M L$TAG KL"
  SEEDS=1 CONTROLS=none ALPHA_MAX=$KAM CONFIG=$C PREFIX="srckl_$M" ./run_mc16_kl.sh "$LA" "$D" "runs/src_kl_${M}_L$TAG" >/dev/null 2>&1 || step "SRC $M L$TAG KL FAILED"
  if [ "$AN" != "-" ]; then
    step "SRC $M L$TAG ADD-N"
    python -u scripts/run_unit_projection.py --config "$C" --run-prefix "srcaddn_$M" --layer "$LA" --d-path "$D" --controls none --seeds 1-1 --mode add --alpha "$AN" --out-dir "runs/src_addn_${M}_L$TAG" >> runs/source_ref.log 2>&1 || step "SRC $M L$TAG ADD-N FAILED"
    step "SRC $M L$TAG ADD-KL"
    SEEDS=1 CONTROLS=none MODE=add ALPHA_MAX=$AAM ALPHA_PROBE=$AAP CONFIG=$C PREFIX="srcaddkl_$M" ./run_mc16_kl.sh "$LA" "$D" "runs/src_addkl_${M}_L$TAG" >/dev/null 2>&1 || step "SRC $M L$TAG ADD-KL FAILED"
  fi
done
step "ALL DONE"
