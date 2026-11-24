#!/usr/bin/env bash
# Prereg addendum D: Llama-3.2-1B-Instruct, after the additive queue (addendum C) has finished.
set -uo pipefail
export PATH="/usr/lib/wsl/lib:$PATH" HF_HUB_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
cd ~/bias_embedding_exp && source .venv/bin/activate
LOG=runs/llama_laptop.log
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
until grep -qE "ALL DONE" runs/add_sweeps_laptop.log 2>/dev/null; do sleep 60; done
C=configs/llama32_1b_jobs56_pronouns.yaml
step "start; additive queue done"
declare -A D
for L in 1 8 14; do
  out=$(python -u scripts/learn_control_dir.py --config "$C" --run-name "llama_learn_s1337_L$L" --layer "$L" \
        --pool mean --target adj --holdout-frac 0.3 --holdout-seed 1337 --ridge-lambda 1.0 2>&1 | tee -a runs/llama_learn.log)
  dir=$(printf '%s\n' "$out" | sed -n 's/^Run complete: //p' | tail -1)
  [ -n "$dir" ] && [ -f "$dir/control_dir.npy" ] && D[$L]="$dir/control_dir.npy" && step "L$L source = ${D[$L]}" || step "learn L$L FAILED"
done
step "KL cost pilot (seeds 101-102)"
python -u scripts/pilot_kl_cost.py --config "$C" --layer-dirs "1=${D[1]},8=${D[8]},14=${D[14]}" --out-json runs/pilot_kl_llama.json >> runs/llama_pilot.log 2>&1 || step "pilot FAILED"
AMAX=$(python - <<'PY'
import json
r=json.load(open('runs/pilot_kl_llama.json'))['rows']
ok20=all(x['kl']>0.21 for x in r if x['alpha']==20.0)
if ok20: print(100)
else:
    for a in sorted({x['alpha'] for x in r}):
        if all(x['kl']>0.21 for x in r if x['alpha']==a): print(int(2*a)); break
    else: print(200)
PY
)
step "KL alpha_max by addendum-D rule = $AMAX"
for L in 14 8 1; do
  step "P1 L$L"
  python -u scripts/run_unit_projection.py --config "$C" --run-prefix lp1 --layer "$L" --d-path "${D[$L]}" --out-dir "runs/llama_p1_L$L" >> "runs/llama_p1_L$L.log" 2>&1 || step "P1 L$L FAILED"
done
for L in 14 8 1; do
  step "KL L$L"
  ALPHA_MAX=$AMAX CONFIG=$C PREFIX=klll ./run_mc16_kl.sh "$L" "${D[$L]}" "runs/kl_llama_L$L" >/dev/null 2>&1 || step "KL L$L FAILED"
  n=$(grep -c "FAILED" "runs/kl_llama_L$L/master.log" 2>/dev/null || true); [ "${n:-0}" != "0" ] && step "KL L$L: $n seed(s) FAILED"
done
step "ALL DONE"
