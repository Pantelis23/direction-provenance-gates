#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


RUN_RE = re.compile(r"Run complete:\s*(runs/\S+)")


def _parse_run_dir(output: str) -> Path:
    m = RUN_RE.search(output)
    if not m:
        raise RuntimeError("Could not parse run directory from command output.")
    return Path(m.group(1))


def _run_cmd(cmd: list[str]) -> Path:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-40:])
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{tail}")
    return _parse_run_dir((proc.stdout or "") + "\n" + (proc.stderr or ""))


def _parse_layers(layer_arg: str) -> list[str]:
    vals = [v.strip() for v in layer_arg.split(",") if v.strip()]
    if not vals:
        raise ValueError("No layers provided.")
    return vals


def _read_tsv_row(path: Path, control: str = "none") -> dict:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for row in rows:
        if row.get("control") == control:
            return row
    raise RuntimeError(f"Control '{control}' not found in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept-name", required=True)
    ap.add_argument("--positive-file", required=True)
    ap.add_argument("--negative-file", required=True)
    ap.add_argument("--layers", default="early,mid,late")
    ap.add_argument("--pool", choices=["mean", "last", "cls"], default="mean")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--run-name", default="concept_layer_sweep")

    # Optional steering-at-matched-cost stage for steerability-per-cost.
    ap.add_argument("--steer", action="store_true")
    ap.add_argument("--source-run", default=None)
    ap.add_argument("--target-pll-change", default="auto")
    ap.add_argument("--auto-target-frac", type=float, default=0.5)
    ap.add_argument("--intervention-scope", choices=["all", "name", "non_name"], default="all")
    ap.add_argument("--controls", default="none,random")
    ap.add_argument("--alpha-min", type=float, default=0.0)
    ap.add_argument("--alpha-max", type=float, default=40.0)
    ap.add_argument("--match-tol", default="auto")
    ap.add_argument("--match-check-bootstrap-samples", type=int, default=200)
    ap.add_argument("--final-bootstrap-samples", type=int, default=5000)
    args = ap.parse_args()

    layers = _parse_layers(args.layers)
    out_dir = Path("runs") / f"{args.run_name}_{args.concept_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for layer in layers:
        extract_cmd = [
            sys.executable,
            "scripts/extract_concept_dir.py",
            "--config",
            args.config,
            "--concept-name",
            args.concept_name,
            "--positive-file",
            args.positive_file,
            "--negative-file",
            args.negative_file,
            "--layer",
            str(layer),
            "--pool",
            args.pool,
            "--holdout-frac",
            str(float(args.holdout_frac)),
            "--holdout-seed",
            str(int(args.holdout_seed)),
            "--ridge-lambda",
            str(float(args.ridge_lambda)),
            "--run-name",
            f"{args.run_name}_extract",
        ]
        extract_run = _run_cmd(extract_cmd)
        meta = json.loads((extract_run / "concept_meta.json").read_text(encoding="utf-8"))

        rec: dict = {
            "layer": str(layer),
            "extract_run": str(extract_run),
            "concept_dir_path": str(extract_run / "concept_dir.npy"),
            "test_auc": meta.get("test_auc"),
            "test_acc": meta.get("test_acc"),
            "train_auc": meta.get("train_auc"),
            "train_acc": meta.get("train_acc"),
        }

        if args.steer:
            if not args.source_run:
                raise ValueError("--steer requires --source-run")
            steer_cmd = [
                sys.executable,
                "scripts/run_causal_matched_cost.py",
                "--config",
                args.config,
                "--source-run",
                args.source_run,
                "--d-path",
                str(extract_run / "concept_dir.npy"),
                "--layer",
                str(layer),
                "--mode",
                "project_out",
                "--alpha-policy",
                "same",
                "--intervention-scope",
                args.intervention_scope,
                "--filter-template",
                "gender_names_jobs",
                "--eval-split",
                "test",
                "--holdout-frac",
                str(float(args.holdout_frac)),
                "--holdout-seed",
                str(int(args.holdout_seed)),
                "--controls",
                args.controls,
                "--alpha-min",
                str(float(args.alpha_min)),
                "--alpha-max",
                str(float(args.alpha_max)),
                "--match-tol",
                str(args.match_tol),
                "--match-check-bootstrap-samples",
                str(int(args.match_check_bootstrap_samples)),
                "--final-bootstrap-samples",
                str(int(args.final_bootstrap_samples)),
                "--run-name",
                f"{args.run_name}_steer_{layer}",
            ]
            steer_target = str(args.target_pll_change).strip().lower()
            if steer_target == "auto":
                steer_cmd += ["--target-pll-change", "auto", "--auto-target-frac", str(float(args.auto_target_frac))]
            else:
                steer_cmd += ["--target-pll-change", str(float(args.target_pll_change))]

            steer_run = _run_cmd(steer_cmd)
            steer_json = json.loads((steer_run / "matched_cost_summary.json").read_text(encoding="utf-8"))
            none_row = _read_tsv_row(steer_run / "matched_cost.tsv", control="none")
            final_cost = float(none_row["final_cost"])
            abs_red = float(none_row["mean_abs_reduction_adj"])
            rec.update(
                {
                    "steer_run": str(steer_run),
                    "steer_target_cost": steer_json.get("target_cost"),
                    "steer_cost_source": steer_json.get("cost_source"),
                    "steer_match_tol_used": steer_json.get("match_tol_used"),
                    "steer_none_matched": (none_row.get("matched") == "True"),
                    "steer_none_final_cost": final_cost,
                    "steer_none_abs_reduction_adj": abs_red,
                    "steerability_per_cost": (abs_red / final_cost) if final_cost > 0 else None,
                }
            )

        rows.append(rec)

    out_json = out_dir / "concept_layer_sweep_summary.json"
    out_json.write_text(json.dumps({"run_id": out_dir.name, "args": vars(args), "rows": rows}, indent=2), encoding="utf-8")

    out_tsv = out_dir / "concept_layer_sweep_summary.tsv"
    keys = sorted({k for r in rows for k in r.keys()})
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Run complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

