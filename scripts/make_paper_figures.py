#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _seed_from_path(p: str) -> int:
    name = Path(p).name
    # seed{N}.matched_cost_summary.json
    tail = name.split("seed", 1)[1]
    return int(tail.split(".", 1)[0])


def _get_ci(x):
    if isinstance(x, dict):
        ci = x.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                return float(ci[0]), float(ci[1])
            except Exception:
                return (float("nan"), float("nan"))
    return (float("nan"), float("nan"))


def _get_mean(x):
    if isinstance(x, dict) and "mean" in x:
        try:
            return float(x["mean"])
        except Exception:
            return float("nan")
    return float("nan")


def _safe_get_shape(fs: dict) -> dict:
    ds = (fs.get("direction_shape_stats") or {})
    return ds if isinstance(ds, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--controls", required=True, help="comma-separated")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    files = sorted(glob.glob(args.seed_glob))
    if not files:
        raise SystemExit(f"no files matched: {args.seed_glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    abs_cos_by_ctrl = {c: [] for c in controls}
    abs_red_by_ctrl = {c: [] for c in controls}
    abs_red_ci_by_ctrl = {c: [] for c in controls}
    per_seed = {}

    for fp in files:
        seed = _seed_from_path(fp)
        j = json.loads(Path(fp).read_text(encoding="utf-8"))
        for rec in j.get("results", []):
            ctrl = str(rec.get("control"))
            if ctrl not in controls:
                continue
            fs = rec.get("final_summary") or {}
            ds = _safe_get_shape(fs)

            eff = fs.get("block_bootstrap_abs_reduction_adj") or {}
            eff_mean = _get_mean(eff)
            ci_lo, ci_hi = _get_ci(eff)

            abs_cos = ds.get("abs_cos_to_source")
            if abs_cos is None:
                abs_cos = ds.get("abs_cos")
            abs_cos = float(abs_cos) if abs_cos is not None else float("nan")

            msm = ds.get("masked_support_mass")
            sif = ds.get("support_intrusion_frac_eps")
            sie = ds.get("support_intrusion_eps")

            alpha = fs.get("alpha")
            alpha = float(alpha) if alpha is not None else float("nan")

            matched = rec.get("matched")
            matched = bool(matched) if matched is not None else False

            rows.append(
                {
                    "seed": seed,
                    "control": ctrl,
                    "matched": matched,
                    "alpha": alpha,
                    "abs_red_mean": eff_mean,
                    "abs_red_ci_lo": ci_lo,
                    "abs_red_ci_hi": ci_hi,
                    "abs_cos_to_source": abs_cos,
                    "masked_support_mass": float(msm) if msm is not None else float("nan"),
                    "support_intrusion_frac_eps": float(sif) if sif is not None else float("nan"),
                    "support_intrusion_eps": float(sie) if sie is not None else float("nan"),
                }
            )

            if np.isfinite(abs_cos):
                abs_cos_by_ctrl[ctrl].append(abs_cos)
            if np.isfinite(eff_mean):
                abs_red_by_ctrl[ctrl].append(eff_mean)
                abs_red_ci_by_ctrl[ctrl].append((ci_lo, ci_hi))

            per_seed.setdefault(seed, {})[ctrl] = eff_mean

    if not rows:
        raise SystemExit("no matching control rows found in summaries")

    csv_path = out_dir / "mc_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["seed"], x["control"])):
            w.writerow(r)

    md_path = out_dir / "mc_table_means.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| seed | " + " | ".join(controls) + " |\n")
        f.write("|---:|" + "|".join(["---:"] * len(controls)) + "|\n")
        for seed in sorted(per_seed.keys()):
            vals = []
            for c in controls:
                v = per_seed[seed].get(c, float("nan"))
                vals.append(f"{v:.6g}" if np.isfinite(v) else "na")
            f.write(f"| {seed} | " + " | ".join(vals) + " |\n")

    plt.figure()
    data = [abs_cos_by_ctrl[c] for c in controls]
    plt.boxplot(data, tick_labels=controls, showfliers=True)
    plt.ylabel("abs_cos_to_source")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_abs_cos_boxplot.png", dpi=200)
    plt.close()

    if len(controls) == 2:
        c0, c1 = controls[0], controls[1]
        deltas = []
        for seed in sorted(per_seed.keys()):
            v0 = per_seed[seed].get(c0)
            v1 = per_seed[seed].get(c1)
            if v0 is None or v1 is None:
                continue
            if np.isfinite(v0) and np.isfinite(v1):
                deltas.append(v1 - v0)
        deltas = np.asarray(deltas, dtype=np.float64)

        plt.figure()
        if deltas.size:
            plt.hist(deltas, bins=12)
            plt.axvline(float(np.mean(deltas)), linestyle="--")
        plt.xlabel(f"delta = {c1} - {c0}")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_paired_delta_hist.png", dpi=200)
        plt.close()

        np.savetxt(out_dir / "paired_deltas.txt", deltas, fmt="%.8g")

    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    print(f"wrote: {out_dir / 'fig_abs_cos_boxplot.png'}")
    if len(controls) == 2:
        print(f"wrote: {out_dir / 'fig_paired_delta_hist.png'}")
        print(f"wrote: {out_dir / 'paired_deltas.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
