#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

paths = sorted(Path("runs/mc_shuf_matched_mcK16_seeds").glob("seed*.matched_cost_summary.json"))
print("found_seed_files", len(paths))
assert paths, "no seed summaries found"

controls = ["none", "orthogonal_random", "abs_marginal_matched_random_orth", "shuffled"]
rows = []
for p in paths:
    j = json.loads(p.read_text())
    seed = int(p.stem.split("seed")[1].split(".")[0])
    rec = {"seed": seed}
    for c in controls:
        rr = next((r for r in j.get("results", []) if r.get("control") == c), None)
        fs = (rr or {}).get("final_summary", {}) or {}
        ci = (fs.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None]
        rec[c] = {
            "matched": (rr or {}).get("matched"),
            "abs_red": fs.get("mean_abs_reduction_adj"),
            "ci_lo": ci[0],
            "ci_hi": ci[1],
        }
    rows.append(rec)

def arr(ctrl, key):
    return np.array([r[ctrl][key] for r in rows if r[ctrl][key] is not None], dtype=float)

sh = arr("shuffled", "abs_red")
no = arr("none", "abs_red")

print("n_seeds", len(rows))
print("shuffled_mean_abs_red", float(sh.mean()))
print("shuffled_q05_q50_q95", [float(x) for x in np.quantile(sh, [0.05, 0.5, 0.95])])
print("none_mean_abs_red", float(no.mean()))
print("none_q05_q50_q95", [float(x) for x in np.quantile(no, [0.05, 0.5, 0.95])])

sig_pos = sum(1 for r in rows if r["shuffled"]["ci_lo"] is not None and r["shuffled"]["ci_lo"] > 0)
sig_neg = sum(1 for r in rows if r["shuffled"]["ci_hi"] is not None and r["shuffled"]["ci_hi"] < 0)
print("shuffled_CI_lo_gt_0", sig_pos, "/", len(rows))
print("shuffled_CI_hi_lt_0", sig_neg, "/", len(rows))

for c in controls:
    m = sum(1 for r in rows if r[c]["matched"])
    print(f"{c}_matched_rate", m, "/", len(rows))

B = 20000
rng = np.random.default_rng(0)
boots = np.array([rng.choice(sh, size=sh.size, replace=True).mean() for _ in range(B)], dtype=float)
lo, hi = np.quantile(boots, [0.025, 0.975])
print("seed_bootstrap_CI95_mean_shuffled", float(lo), float(hi))
PY
