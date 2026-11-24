#!/usr/bin/env python3
"""Calibration check for the block-bootstrap SEs used by the heterogeneity test.

Holds each direction FIXED (same control, same direction seed, same matched
alpha) and re-evaluates it on N different held-out block splits
(--holdout-seed). Its true effect is then the same across splits, so if the
per-split block-bootstrap SEs are accurate, Cochran's Q over the N splits
follows chi2(N-1): E[Q/(N-1)] = 1. Q/(N-1) >> 1 means the SEs are too small
and the across-direction Q in analyze_mc_controls.py is inflated.

Splits overlap slightly (each is ~5% of blocks), which makes this test mildly
conservative (Q biased low), so a ratio near 1 is the pass condition and a
ratio well below 1 is also reported.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"Run complete:\s*(runs/\S+)")


def _matched(summary_path: Path, control: str) -> tuple[float, dict]:
    j = json.loads(summary_path.read_text(encoding="utf-8"))
    for rec in j["results"]:
        if rec["control"] == control:
            if not rec.get("matched"):
                raise SystemExit(f"{summary_path}:{control} is not matched")
            return float(rec["matched_alpha"]), j["args"]
    raise SystemExit(f"{control} not in {summary_path}")


def _run(args: dict, control: str, alpha: float, holdout_seed: int, boots: int, name: str) -> dict:
    cmd = [
        sys.executable, str(REPO / "scripts/run_causal.py"),
        "--config", args["config"], "--source-run", args["source_run"], "--d-path", args["d_path"],
        "--layer", str(args["layer"]), "--mode", args["mode"], "--alpha-policy", args["alpha_policy"],
        "--intervention-scope", args["intervention_scope"], "--filter-template", args["filter_template"],
        "--eval-split", args["eval_split"], "--holdout-frac", str(args["holdout_frac"]),
        "--holdout-seed", str(holdout_seed), "--control-seed", str(args["control_seed"]),
        "--direction-seed", str(args["direction_seed"]), "--direction-control", control,
        "--alpha", repr(alpha), "--bootstrap-samples", str(boots), "--run-name", name,
    ]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(out[-3000:])
    m = RUN_RE.search(out)
    if not m:
        raise RuntimeError("no run dir in output")
    s = json.loads((REPO / m.group(1) / "results_causal_summary.json").read_text(encoding="utf-8"))
    eff = s["block_bootstrap_abs_reduction_adj"]
    lo, hi = eff["ci95"]
    return {"holdout_seed": holdout_seed, "mean": float(eff["mean"]), "lo": float(lo), "hi": float(hi),
            "se": (float(hi) - float(lo)) / (2 * 1.959964), "n_blocks": s.get("n_blocks"),
            "direction_sha256": s.get("direction_sha256"), "run_dir": m.group(1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="runs/mc16_v2_randmask")
    ap.add_argument("--seeds", default="1,2,3", help="Direction seeds (seed summaries) to calibrate.")
    ap.add_argument("--controls", default="orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512")
    ap.add_argument("--n-splits", type=int, default=16)
    ap.add_argument("--first-holdout-seed", type=int, default=5000)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--out-json", default="runs/se_calibration.json")
    args = ap.parse_args()

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        summary = REPO / args.sweep_dir / f"seed{seed}.matched_cost_summary.json"
        for control in [c for c in args.controls.split(",") if c]:
            alpha, sargs = _matched(summary, control)
            splits = []
            for k in range(args.n_splits):
                hs = args.first_holdout_seed + k
                r = _run(sargs, control, alpha, hs, args.bootstrap_samples, f"secal_s{seed}_{control}_h{hs}")
                splits.append(r)
                print(f"seed={seed} {control} holdout={hs} mean={r['mean']:+.5f} se={r['se']:.5f}", flush=True)
            shas = {r["direction_sha256"] for r in splits}
            if len(shas) != 1:
                raise SystemExit(f"direction changed across splits for seed {seed} {control}: {shas}")
            x = np.array([r["mean"] for r in splits])
            se = np.array([r["se"] for r in splits])
            w = 1 / se**2
            mu = float(np.sum(w * x) / np.sum(w))
            q = float(np.sum(w * (x - mu) ** 2))
            rows.append({"seed": seed, "control": control, "alpha": alpha, "direction_sha256": shas.pop(),
                         "Q": q, "df": len(x) - 1, "Q_over_df": q / (len(x) - 1),
                         "sd_of_means": float(x.std(ddof=1)), "median_se": float(np.median(se)), "splits": splits})
            print(f"  -> Q/df = {q / (len(x) - 1):.3f}  sd(means)={x.std(ddof=1):.5f}  median se={np.median(se):.5f}", flush=True)

    qs = np.array([r["Q"] for r in rows])
    dfs = np.array([r["df"] for r in rows])
    pooled = float(qs.sum() / dfs.sum())
    # Under calibration, sum(Q) ~ chi2(sum(df)); report the MC 95% band for the pooled ratio.
    sims = np.random.default_rng(0).chisquare(dfs.sum(), size=200_000) / dfs.sum()
    band = [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]
    print(f"\npooled Q/df = {pooled:.3f}  (calibrated 95% band {band[0]:.3f}-{band[1]:.3f}, n={len(rows)} directions)")
    Path(REPO / args.out_json).write_text(json.dumps({"pooled_Q_over_df": pooled, "calibrated_band95": band,
                                                      "directions": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
