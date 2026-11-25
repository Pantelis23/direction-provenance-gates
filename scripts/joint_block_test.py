#!/usr/bin/env python3
"""Covariance-robust homogeneity test for a sweep of directions on a shared held-out split.

Cochran's Q treats the 16 per-direction estimates as independent, but every
direction is evaluated on the same held-out (template, job) blocks, so their
sampling errors can covary. This test keeps that covariance in its null:

  x[i, b] = mean abs_red of direction i over the name pairs of block b
  theta_i = mean_b x[i, b]                       (= the reported effect)
  T       = sum_i (theta_i - mean_i theta_i)^2   (spread of the effects)
  null    : recentre each direction, r[i, b] = x[i, b] - theta_i, then
            resample BLOCKS jointly for all directions (one shared draw)
  p       = P(T* >= T)

and the same joint resampling applied to Cochran's Q with block-level SEs
recomputed inside each resample (p_joint_Q, the primary, more powerful version).

Because all directions are resampled together, any correlation between them
induced by the shared blocks is reproduced in T*. Also reports the mean
pairwise Pearson correlation of the recentred block vectors across directions
(the quantity that would make Q anticonservative if negative, conservative if
positive) and the implied Q/df under independence for comparison.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BLOCK_KEYS = ("pll_template", "pll_template_str", "pll_job", "pll_article", "pll_baseline_template")


def run_dir_for(rec: dict, root: Path) -> Path:
    if rec.get("final_run_dir"):
        return root / rec["final_run_dir"]
    return root / "runs" / rec["final_summary"]["run_id"]


def block_matrix(seed_glob: str, root: Path, control: str) -> tuple[np.ndarray, list, np.ndarray]:
    vecs, labels, reported = [], [], []
    blocks_ref = None
    for fp in sorted(glob.glob(str(root / seed_glob))):
        for rec in json.loads(Path(fp).read_text())["results"]:
            if rec["control"] != control or not rec.get("matched", True):
                continue
            rows = [json.loads(l) for l in open(run_dir_for(rec, root) / "results_causal.jsonl")]
            rows = [r for r in rows if r.get("split", "test") == "test" and r.get("pll_abs_reduction_adj") is not None]
            by = {}
            for r in rows:
                by.setdefault(tuple(r.get(k) for k in BLOCK_KEYS), []).append(float(r["pll_abs_reduction_adj"]))
            keys = sorted(by, key=str)
            if blocks_ref is None:
                blocks_ref = keys
            if keys != blocks_ref:
                raise SystemExit(f"{fp}: block set differs from the first direction's")
            vecs.append([np.mean(by[k]) for k in keys])
            labels.append(Path(fp).name.split(".")[0])
            reported.append(float(rec["final_summary"]["block_bootstrap_abs_reduction_adj"]["mean"]))
    return np.array(vecs), labels, np.array(reported)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--controls", default="orthogonal_random,orthogonal_random_support_avoid_top512,orthogonal_random_random_mask_top512")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    out = {"seed_glob": args.seed_glob, "controls": {}}
    for c in [x for x in args.controls.split(",") if x]:
        X, labels, reported = block_matrix(args.seed_glob, root, c)
        n, B = X.shape
        theta = X.mean(1)
        if not np.allclose(theta, reported, atol=1e-9):
            raise SystemExit(f"{c}: block means do not reproduce the reported effects (max diff {np.abs(theta - reported).max():.2e})")
        T = float(((theta - theta.mean()) ** 2).sum())
        R = X - theta[:, None]
        idx = rng.integers(0, B, size=(args.n_boot, B))
        Tb = np.empty(args.n_boot)
        for k in range(args.n_boot):
            th = R[:, idx[k]].mean(1)
            Tb[k] = ((th - th.mean()) ** 2).sum()
        p = float((np.sum(Tb >= T) + 1) / (args.n_boot + 1))
        # Studentized version: Cochran's Q with block-level SEs, its null built by the
        # same joint block resampling (SEs recomputed inside each resample).
        se_obs = X.std(1, ddof=1) / np.sqrt(B)
        w = 1 / se_obs**2
        Q_obs = float((w * (theta - (w * theta).sum() / w.sum()) ** 2).sum())
        Qb = np.empty(args.n_boot)
        for k in range(args.n_boot):
            Rk = R[:, idx[k]]
            th = Rk.mean(1)
            se = Rk.std(1, ddof=1) / np.sqrt(B)
            wk = 1 / np.maximum(se, 1e-12) ** 2
            Qb[k] = (wk * (th - (wk * th).sum() / wk.sum()) ** 2).sum()
        p_q = float((np.sum(Qb >= Q_obs) + 1) / (args.n_boot + 1))
        C = np.corrcoef(R)
        off = C[~np.eye(n, dtype=bool)]
        se_ind = R.std(1, ddof=1) / np.sqrt(B)
        q_ind = float((((theta - np.average(theta, weights=1 / se_ind**2)) ** 2) / se_ind**2).sum())
        out["controls"][c] = {"n_directions": n, "n_blocks": B, "Q": Q_obs, "p_joint_Q": p_q, "T": T, "p_joint_T": p,
                              "mean_pairwise_corr": float(off.mean()), "corr_q05_q95": [float(np.quantile(off, 0.05)), float(np.quantile(off, 0.95))],
                              "Q_independent_blocks": q_ind, "Q_over_df_independent": q_ind / (n - 1)}
        print(f"{c[-24:]:24s} n={n} blocks={B} Q/df={Q_obs/(n-1):.2f} p_jointQ={p_q:.5f} p_jointT={p:.5f} mean_corr={off.mean():+.3f} [{np.quantile(off,0.05):+.2f},{np.quantile(off,0.95):+.2f}] Q/df(ind)={q_ind/(n-1):.2f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
