#!/usr/bin/env python3
"""P1 condition of the pre-registration: alpha = 1 (a true projection), no cost matching.

Runs run_causal.py for every (seed, control) at one layer and writes one
seed{N}.matched_cost_summary.json-shaped file per seed, so analyze_mc_controls.py
and check_direction_integrity.py can read P1 sweeps exactly like MC sweeps.
Direction vectors are copied into <out_dir>/directions/ (hermetic), as the
matched-cost driver does.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"Run complete:\s*(runs/\S+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--d-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="1-16")
    ap.add_argument("--controls", default="orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512")
    ap.add_argument("--config", default="configs/mvp_jobs56_pronouns.yaml")
    ap.add_argument("--run-prefix", default="p1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mode", choices=["project_out", "add"], default="project_out")
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    out_dir = REPO / args.out_dir
    (out_dir / "directions").mkdir(parents=True, exist_ok=True)

    for seed in range(a, b + 1):
        out = out_dir / f"seed{seed}.matched_cost_summary.json"
        if out.exists():
            print(f"skip seed {seed}")
            continue
        results = []
        for control in [c for c in args.controls.split(",") if c]:
            name = f"{args.run_prefix}_{args.layer}_s{seed}_{control}"
            cmd = [
                sys.executable, str(REPO / "scripts/run_causal.py"),
                "--config", args.config,
                "--source-run", "runs/mvp_jobs56_pronouns_20260220_232956",
                "--d-path", args.d_path, "--layer", args.layer,
                "--mode", args.mode, "--alpha-policy", "same", "--intervention-scope", "non_name",
                "--filter-template", "gender_names_jobs", "--eval-split", "test",
                "--holdout-frac", "0.05", "--holdout-seed", "1337",
                "--control-seed", "2027", "--direction-seed", str(seed),
                "--direction-control", control, "--alpha", repr(args.alpha),
                "--bootstrap-samples", str(args.bootstrap_samples), "--run-name", name,
            ]
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            txt = p.stdout + p.stderr
            m = RUN_RE.search(txt)
            if p.returncode != 0 or not m:
                raise SystemExit(f"run_causal failed for {name}:\n{txt[-3000:]}")
            run_dir = REPO / m.group(1)
            fs = json.loads((run_dir / "results_causal_summary.json").read_text(encoding="utf-8"))
            rec = {"control": control, "matched": True, "matched_alpha": args.alpha,
                   "direction_sha256": fs.get("direction_sha256"),
                   "direction_path": fs.get("direction_path"),
                   "direction_path_control": fs.get("direction_path_control"),
                   "final_summary": fs}
            for key, src, suffix in (("direction_path_hermetic", fs.get("direction_path"), "direction"),
                                     ("direction_path_control_hermetic", fs.get("direction_path_control"), "direction_control")):
                if src:
                    dst = out_dir / "directions" / f"seed{seed}.{control}.{suffix}.npy"
                    shutil.copyfile(REPO / src, dst)
                    rec[key] = str(dst.relative_to(REPO))
            results.append(rec)
            eff = fs.get("block_bootstrap_abs_reduction_adj") or {}
            print(f"layer={args.layer} seed={seed} {control} abs_red={eff.get('mean')} ci={eff.get('ci95')}", flush=True)
        out.write_text(json.dumps({"schema_version": 2, "condition": "P1" if args.mode == "project_out" else f"{args.mode.upper()}-fixed", "mode": args.mode, "layer": args.layer,
                                   "alpha": args.alpha, "results": results}, indent=2), encoding="utf-8")
    print(f"DONE P1 layer {args.layer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
