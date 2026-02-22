#!/usr/bin/env python3
"""Analyze shuffled-control outliers across per-seed matched-cost summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_seed(path: Path) -> int:
    # Expected: seed{N}.matched_cost_summary.json
    stem = path.name
    if not stem.startswith("seed"):
        raise ValueError(f"Unexpected filename format: {path}")
    token = stem.split(".", 1)[0]
    return int(token.replace("seed", "", 1))


def _get_shuffled_row(summary: dict[str, Any]) -> dict[str, Any]:
    for rec in summary.get("results", []):
        if rec.get("control") == "shuffled":
            return rec
    raise KeyError("No shuffled control found in results[]")


def _shape_stats_from_summary(rec: dict[str, Any]) -> dict[str, Any]:
    stats = (rec.get("final_summary", {}) or {}).get("direction_shape_stats", {}) or {}
    return {
        "frac_pos": stats.get("frac_pos"),
        "corr_sign_mag": stats.get("corr_sign_mag"),
        "top32_overlap": stats.get("top32_overlap"),
        "top128_overlap": stats.get("top128_overlap"),
        "top512_overlap": stats.get("top512_overlap"),
        "abs_cos": stats.get("abs_cos_to_source"),
        "kurt_excess": stats.get("kurtosis_excess"),
        "top32_mass": stats.get("top32_mass"),
        "top128_mass": stats.get("top128_mass"),
        "top512_mass": stats.get("top512_mass"),
        "l1": stats.get("l1"),
        "l2": stats.get("l2"),
        "linf": stats.get("linf"),
        "sparsity_lt_1e-3": stats.get("sparsity_lt_1e-3"),
    }


def _shape_stats_from_final_run_dir(rec: dict[str, Any]) -> dict[str, Any]:
    run_dir = rec.get("final_run_dir")
    if not run_dir:
        return {}
    path = Path(run_dir) / "results_causal_summary.json"
    if not path.is_file():
        return {}
    stats = (_load_json(path).get("direction_shape_stats", {}) or {})
    return {
        "frac_pos": stats.get("frac_pos"),
        "corr_sign_mag": stats.get("corr_sign_mag"),
        "top32_overlap": stats.get("top32_overlap"),
        "top128_overlap": stats.get("top128_overlap"),
        "top512_overlap": stats.get("top512_overlap"),
        "abs_cos": stats.get("abs_cos_to_source"),
        "kurt_excess": stats.get("kurtosis_excess"),
        "top32_mass": stats.get("top32_mass"),
        "top128_mass": stats.get("top128_mass"),
        "top512_mass": stats.get("top512_mass"),
        "l1": stats.get("l1"),
        "l2": stats.get("l2"),
        "linf": stats.get("linf"),
        "sparsity_lt_1e-3": stats.get("sparsity_lt_1e-3"),
    }


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed-glob",
        default="runs/mc_shuf_matched_mcK16_seeds/seed*.matched_cost_summary.json",
        help="Glob for per-seed matched_cost_summary.json files.",
    )
    ap.add_argument(
        "--out-tsv",
        default="runs/mc_shuf_matched_mcK16_shuffled_seedstats.tsv",
        help="Output TSV path.",
    )
    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.seed_glob))
    if not paths:
        raise SystemExit(f"No files found for --seed-glob: {args.seed_glob}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        seed = _extract_seed(path)
        summary = _load_json(path)
        rec = _get_shuffled_row(summary)
        final_summary = rec.get("final_summary", {}) or {}
        ci95 = (
            (final_summary.get("block_bootstrap_abs_reduction_adj", {}) or {}).get("ci95")
            or [None, None]
        )
        shape = _shape_stats_from_summary(rec)
        if not any(v is not None for v in shape.values()):
            shape = _shape_stats_from_final_run_dir(rec)
        rows.append(
            {
                "seed": seed,
                "matched": rec.get("matched"),
                "alpha": rec.get("matched_alpha"),
                "cost_err": rec.get("cost_err"),
                "abs_red": final_summary.get("mean_abs_reduction_adj"),
                "ci_lo": ci95[0],
                "ci_hi": ci95[1],
                **shape,
            }
        )

    rows.sort(key=lambda r: r["seed"])
    numeric_rows = [r for r in rows if r.get("abs_red") is not None]
    numeric_rows_sorted = sorted(numeric_rows, key=lambda r: float(r["abs_red"]))

    print(f"n_rows={len(rows)}")
    print("most negative (5):")
    for row in numeric_rows_sorted[:5]:
        print(row)
    print("\nmost positive (5):")
    for row in list(reversed(numeric_rows_sorted[-5:])):
        print(row)

    keys = [
        k
        for k in rows[0].keys()
        if k not in {"seed", "matched", "alpha", "cost_err", "abs_red", "ci_lo", "ci_hi"}
    ]
    y = np.array([_float_or_nan(r["abs_red"]) for r in numeric_rows], dtype=float)

    print("\ncorrelations vs abs_red:")
    for key in keys:
        x = np.array([_float_or_nan(r.get(key)) for r in numeric_rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 3:
            continue
        corr = _corr(x[mask], y[mask])
        if corr is None:
            continue
        print(f"{key:20s} corr={corr:+.3f}")

    cols = [
        "seed",
        "matched",
        "alpha",
        "cost_err",
        "abs_red",
        "ci_lo",
        "ci_hi",
        "frac_pos",
        "corr_sign_mag",
        "abs_cos",
        "kurt_excess",
        "top32_mass",
        "top128_mass",
        "top512_mass",
        "top32_overlap",
        "top128_overlap",
        "top512_overlap",
        "l1",
        "l2",
        "linf",
        "sparsity_lt_1e-3",
    ]

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
