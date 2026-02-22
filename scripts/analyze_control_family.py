#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_seed(path: str) -> int:
    name = Path(path).name
    tok = name.split(".", 1)[0]
    return int(tok.replace("seed", "", 1))


def get_ctrl(summary: dict, ctrl: str) -> dict | None:
    for rec in summary.get("results", []):
        if rec.get("control") == ctrl:
            return rec
    return None


def get_abs_red_ci(rec: dict) -> tuple[float | None, float | None, float | None]:
    fs = rec.get("final_summary") or {}
    val = fs.get("mean_abs_reduction_adj")
    ci = ((fs.get("block_bootstrap_abs_reduction_adj") or {}).get("ci95") or [None, None])
    return val, ci[0], ci[1]


def seed_boot_ci_mean(x: np.ndarray, B: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(B)], dtype=float)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", default="runs/mc_shuf_matched_mcK16_seeds/seed*.matched_cost_summary.json")
    ap.add_argument(
        "--controls",
        default="none,orthogonal_random,abs_marginal_matched_random_orth,shuffled",
    )
    ap.add_argument("--out-md", default="runs/mc_shuf_matched_mcK16_family_controls.md")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.seed_glob))
    if not paths:
        raise SystemExit(f"no files matched: {args.seed_glob}")

    ctrls = [c.strip() for c in args.controls.split(",") if c.strip()]
    rows: list[dict] = []
    for path in paths:
        seed = extract_seed(path)
        summ = load(path)
        row: dict = {"seed": seed}
        for ctrl in ctrls:
            rec = get_ctrl(summ, ctrl)
            if rec is None:
                row[ctrl] = None
                continue
            v, lo, hi = get_abs_red_ci(rec)
            row[ctrl] = {"v": v, "lo": lo, "hi": hi, "matched": rec.get("matched")}
        rows.append(row)

    K = len(rows)
    lines: list[str] = [f"# Family control summary (K={K})", ""]
    for ctrl in ctrls:
        vals = []
        sig_pos = 0
        sig_neg = 0
        matched = 0
        for row in rows:
            obj = row.get(ctrl)
            if not obj:
                continue
            if obj.get("matched"):
                matched += 1
            v = obj.get("v")
            lo = obj.get("lo")
            hi = obj.get("hi")
            if v is None:
                continue
            vals.append(float(v))
            if lo is not None and float(lo) > 0:
                sig_pos += 1
            if hi is not None and float(hi) < 0:
                sig_neg += 1

        a = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        lines.append(f"## {ctrl}")
        lines.append(f"- matched_rate: {matched}/{K}")
        if a.size == 0:
            lines.append("- no numeric results")
            lines.append("")
            continue

        lo_m, hi_m = seed_boot_ci_mean(a, B=20000, seed=0)
        q05, q50, q95 = np.quantile(a, [0.05, 0.5, 0.95])
        lines.append(f"- mean_abs_red: {a.mean():+.6f}")
        lines.append(f"- seed_boot_CI95(mean): [{lo_m:+.6f}, {hi_m:+.6f}]")
        lines.append(f"- q05/q50/q95: [{q05:+.6f}, {q50:+.6f}, {q95:+.6f}]")
        lines.append(f"- sig_pos(ci_lo>0): {sig_pos}/{K}")
        lines.append(f"- sig_neg(ci_hi<0): {sig_neg}/{K}")
        lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
