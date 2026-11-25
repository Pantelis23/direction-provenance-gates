#!/usr/bin/env python3
"""Null calibration of the homogeneity tests on real shared-split data.

For a cell, take the recentred block-level matrix R[i, b] (16 directions x 26
shared blocks) from joint_block_test.block_matrix. Synthetic NULL datasets
(all directions share one effect, but keep the real cross-direction covariance
and the real tails) are made by resampling blocks jointly: X* = R[:, idx].
On each, run:
  - chi2-Q   : Cochran's Q with block-level SEs, p from chi2(n-1)
  - joint-Q  : Q, null by joint block resampling with SEs recomputed (nested)
  - joint-T  : unweighted spread, null by joint block resampling (nested)
and report each test's rejection rate at alpha = 0.05 (calibrated = ~0.05).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_block_test import block_matrix  # noqa: E402


def chi2_sf(q: float, df: int, rng: np.random.Generator, n_mc: int = 20000) -> float:
    # survival function via the regularized gamma (series), exact enough for df <= 40
    a, x = df / 2.0, q / 2.0
    if x <= 0:
        return 1.0
    term = 1.0 / a
    s = term
    for k in range(1, 2000):
        term *= x / (a + k)
        s += term
        if term < 1e-14 * s:
            break
    lower = math.exp(-x + a * math.log(x) - math.lgamma(a)) * s
    return max(0.0, 1.0 - lower)


def stats(X: np.ndarray) -> tuple[float, float]:
    n, B = X.shape
    th = X.mean(1)
    se = X.std(1, ddof=1) / np.sqrt(B)
    w = 1 / np.maximum(se, 1e-12) ** 2
    Q = float((w * (th - (w * th).sum() / w.sum()) ** 2).sum())
    T = float(((th - th.mean()) ** 2).sum())
    return Q, T


def joint_p(X: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n, B = X.shape
    Q, T = stats(X)
    R = X - X.mean(1, keepdims=True)
    idx = rng.integers(0, B, size=(n_boot, B))
    qb = np.empty(n_boot)
    tb = np.empty(n_boot)
    for k in range(n_boot):
        qb[k], tb[k] = stats(R[:, idx[k]])
    return float((np.sum(qb >= Q) + 1) / (n_boot + 1)), float((np.sum(tb >= T) + 1) / (n_boot + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, help="NAME=SEED_GLOB")
    ap.add_argument("--control", default="orthogonal_random")
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {}
    for spec in args.cell:
        name, g = spec.split("=", 1)
        X, _, _ = block_matrix(g, Path("."), args.control)
        n, B = X.shape
        R = X - X.mean(1, keepdims=True)
        rej = {"chi2_Q": 0, "joint_Q": 0, "joint_T": 0}
        for _ in range(args.n_sim):
            Xs = R[:, rng.integers(0, B, size=B)]  # synthetic null dataset, real covariance + tails
            Q, _ = stats(Xs)
            rej["chi2_Q"] += chi2_sf(Q, n - 1, rng) < 0.05
            pq, pt = joint_p(Xs, rng, args.n_boot)
            rej["joint_Q"] += pq < 0.05
            rej["joint_T"] += pt < 0.05
        rates = {k: v / args.n_sim for k, v in rej.items()}
        res[name] = {"control": args.control, "n_sim": args.n_sim, "rejection_rate_at_0.05": rates}
        print(f"{name:20s} {args.control[-20:]:20s} null rejection @0.05: chi2-Q={rates['chi2_Q']:.3f} joint-Q={rates['joint_Q']:.3f} joint-T={rates['joint_T']:.3f}", flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
