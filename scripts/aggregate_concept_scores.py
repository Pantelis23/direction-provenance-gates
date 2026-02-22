#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return None
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if den <= 0:
        return None
    return float(x0.dot(y0) / den)


def _read_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/*/results_causal_summary.json")
    ap.add_argument("--require-rel", action="store_true", help="Keep only runs that contain rel concept metrics.")
    ap.add_argument(
        "--min-schema-version",
        type=int,
        default=0,
        help="Keep only runs with summary.schema_version >= this value.",
    )
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    summary_paths = sorted(Path(p) for p in glob.glob(args.runs_glob))
    if not summary_paths:
        raise SystemExit(f"No summary files matched: {args.runs_glob}")

    run_rows = []
    for sp in summary_paths:
        run_dir = sp.parent
        summary = json.loads(sp.read_text(encoding="utf-8"))
        if not bool(summary.get("concept_score_enabled")):
            continue
        if int(summary.get("schema_version", 0)) < int(args.min_schema_version):
            continue
        if args.require_rel and summary.get("mean_concept_score_rel_delta_pair") is None:
            continue
        jsonl_path = run_dir / "results_causal.jsonl"
        if not jsonl_path.exists():
            continue

        rows = _read_rows(jsonl_path)
        cs_delta = np.array(
            [float(r["concept_score_delta_pair"]) for r in rows if r.get("concept_score_delta_pair") is not None],
            dtype=float,
        )
        cs_abs_delta = np.array(
            [
                float(r["concept_score_abs_delta_pair"])
                for r in rows
                if r.get("concept_score_abs_delta_pair") is not None
            ],
            dtype=float,
        )
        cs_rel_delta = np.array(
            [float(r["concept_score_rel_delta_pair"]) for r in rows if r.get("concept_score_rel_delta_pair") is not None],
            dtype=float,
        )
        cs_abs_rel_delta = np.array(
            [
                float(r["concept_score_abs_rel_delta_pair"])
                for r in rows
                if r.get("concept_score_abs_rel_delta_pair") is not None
            ],
            dtype=float,
        )

        pairs_delta = [
            (float(r["concept_score_delta_pair"]), float(r["pll_abs_reduction_adj"]))
            for r in rows
            if r.get("concept_score_delta_pair") is not None and r.get("pll_abs_reduction_adj") is not None
        ]
        pairs_abs_delta = [
            (float(r["concept_score_abs_delta_pair"]), float(r["pll_abs_reduction_adj"]))
            for r in rows
            if r.get("concept_score_abs_delta_pair") is not None and r.get("pll_abs_reduction_adj") is not None
        ]
        pairs_rel_delta = [
            (float(r["concept_score_rel_delta_pair"]), float(r["pll_abs_reduction_adj"]))
            for r in rows
            if r.get("concept_score_rel_delta_pair") is not None and r.get("pll_abs_reduction_adj") is not None
        ]
        pairs_abs_rel_delta = [
            (float(r["concept_score_abs_rel_delta_pair"]), float(r["pll_abs_reduction_adj"]))
            for r in rows
            if r.get("concept_score_abs_rel_delta_pair") is not None and r.get("pll_abs_reduction_adj") is not None
        ]

        x_delta = np.array([a for a, _ in pairs_delta], dtype=float)
        y_delta = np.array([b for _, b in pairs_delta], dtype=float)
        x_abs_delta = np.array([a for a, _ in pairs_abs_delta], dtype=float)
        y_abs_delta = np.array([b for _, b in pairs_abs_delta], dtype=float)
        x_rel_delta = np.array([a for a, _ in pairs_rel_delta], dtype=float)
        y_rel_delta = np.array([b for _, b in pairs_rel_delta], dtype=float)
        x_abs_rel_delta = np.array([a for a, _ in pairs_abs_rel_delta], dtype=float)
        y_abs_rel_delta = np.array([b for _, b in pairs_abs_rel_delta], dtype=float)

        run_rows.append(
            {
                "run_id": summary.get("run_id"),
                "run_dir": str(run_dir),
                "direction_control": summary.get("direction_control"),
                "intervention_scope": summary.get("intervention_scope"),
                "layer_idx": summary.get("layer_idx"),
                "alpha": summary.get("alpha"),
                "n_rows": summary.get("n_rows"),
                "n_blocks": summary.get("n_blocks"),
                "mean_abs_reduction_adj": summary.get("mean_abs_reduction_adj"),
                "mean_logprob_change_pair": summary.get("mean_logprob_change_pair"),
                "mean_pll_change": summary.get("mean_pll_change"),
                "mean_concept_score_delta_pair": summary.get("mean_concept_score_delta_pair"),
                "mean_concept_score_abs_delta_pair": summary.get("mean_concept_score_abs_delta_pair"),
                "mean_concept_score_rel_delta_pair": summary.get("mean_concept_score_rel_delta_pair"),
                "mean_concept_score_abs_rel_delta_pair": summary.get("mean_concept_score_abs_rel_delta_pair"),
                "std_concept_score_delta_pair": float(np.std(cs_delta, ddof=0)) if cs_delta.size else None,
                "std_concept_score_abs_delta_pair": float(np.std(cs_abs_delta, ddof=0)) if cs_abs_delta.size else None,
                "std_concept_score_rel_delta_pair": float(np.std(cs_rel_delta, ddof=0)) if cs_rel_delta.size else None,
                "std_concept_score_abs_rel_delta_pair": (
                    float(np.std(cs_abs_rel_delta, ddof=0)) if cs_abs_rel_delta.size else None
                ),
                "corr_concept_delta_vs_abs_reduction_adj": _pearson(x_delta, y_delta),
                "corr_concept_abs_delta_vs_abs_reduction_adj": _pearson(x_abs_delta, y_abs_delta),
                "corr_concept_rel_delta_vs_abs_reduction_adj": _pearson(x_rel_delta, y_rel_delta),
                "corr_concept_abs_rel_delta_vs_abs_reduction_adj": _pearson(x_abs_rel_delta, y_abs_rel_delta),
            }
        )

    if not run_rows:
        raise SystemExit("No runs with concept_score_enabled=true were found.")

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in run_rows:
        key = (str(r.get("direction_control")), str(r.get("intervention_scope")))
        by_group[key].append(r)

    group_rows = []
    for (control, scope), recs in sorted(by_group.items()):
        def _m(k):
            vals = [_as_float(x.get(k)) for x in recs]
            vals = [v for v in vals if v is not None]
            return float(np.mean(np.array(vals, dtype=float))) if vals else None

        group_rows.append(
            {
                "direction_control": control,
                "intervention_scope": scope,
                "n_runs": len(recs),
                "mean_of_mean_abs_reduction_adj": _m("mean_abs_reduction_adj"),
                "mean_of_mean_logprob_change_pair": _m("mean_logprob_change_pair"),
                "mean_of_mean_pll_change": _m("mean_pll_change"),
                "mean_of_mean_concept_score_delta_pair": _m("mean_concept_score_delta_pair"),
                "mean_of_std_concept_score_delta_pair": _m("std_concept_score_delta_pair"),
                "mean_corr_concept_delta_vs_abs_reduction_adj": _m("corr_concept_delta_vs_abs_reduction_adj"),
                "mean_of_mean_concept_score_rel_delta_pair": _m("mean_concept_score_rel_delta_pair"),
                "mean_of_std_concept_score_rel_delta_pair": _m("std_concept_score_rel_delta_pair"),
                "mean_corr_concept_rel_delta_vs_abs_reduction_adj": _m(
                    "corr_concept_rel_delta_vs_abs_reduction_adj"
                ),
            }
        )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_prefix = Path(args.out_prefix) if args.out_prefix else Path(f"runs/concept_score_aggregate_{ts}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_json = out_prefix.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "runs_glob": args.runs_glob,
                "require_rel": bool(args.require_rel),
                "min_schema_version": int(args.min_schema_version),
                "run_count": len(run_rows),
                "rows": run_rows,
                "group_summary": group_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_tsv = out_prefix.with_suffix(".tsv")
    fields = sorted({k for r in run_rows for k in r.keys()})
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in run_rows:
            w.writerow(r)

    out_group_tsv = out_prefix.with_name(out_prefix.name + "_group").with_suffix(".tsv")
    g_fields = sorted({k for r in group_rows for k in r.keys()})
    with out_group_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=g_fields, delimiter="\t")
        w.writeheader()
        for r in group_rows:
            w.writerow(r)

    print(f"Run complete: {out_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
