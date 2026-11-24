#!/usr/bin/env bash
# Pre-registered follow-ups (docs/prereg_2026-09-28.md), run after the MC16 v2 sweep.
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG=runs/followups.log
JLENS_SRC="${JLENS_SRC:?set JLENS_SRC to a jacobian-lens checkout}"
step() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

until grep -q "DONE mc16 v2 sweep" runs/mc16_v2_randmask/master.log 2>/dev/null; do sleep 60; done
step "v2 sweep done"

step "H1 SE calibration"
.venv/bin/python -u scripts/calibrate_block_se.py --seeds 1,2,3 >> runs/se_calibration.log 2>&1 || step "H1 FAILED"

for spec in "late:runs/grid_learn_s1337_late_20260221_081734/control_dir.npy" "mid:runs/grid_learn_s1337_mid_20260221_081426/control_dir.npy" "early:runs/grid_learn_s1337_early_20260221_081121/control_dir.npy"; do
  layer=${spec%%:*}; d=${spec#*:}
  step "P1 $layer"
  .venv/bin/python -u scripts/run_unit_projection.py --layer "$layer" --d-path "$d" --out-dir "runs/p1_$layer" >> "runs/p1_$layer.log" 2>&1 || step "P1 $layer FAILED"
done

step "J-lens fit (layers 1,6)"
.venv/bin/python -u scripts/jspace_propagation.py --jlens-src "$JLENS_SRC" --layers 1,6 >> runs/jlens_fit.log 2>&1 || step "J-lens FAILED"

step "MC mid"
./run_mc16_layer.sh mid "runs/grid_learn_s1337_mid_20260221_081426/control_dir.npy" runs/mc16_mid >/dev/null 2>&1 || step "MC mid FAILED"
step "MC early"
./run_mc16_layer.sh early "runs/grid_learn_s1337_early_20260221_081121/control_dir.npy" runs/mc16_early >/dev/null 2>&1 || step "MC early FAILED"
step "ALL DONE"
