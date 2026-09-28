#!/usr/bin/env python3
"""Statistics for a matched-cost control sweep (one direction per seed per control).

Reads per-seed summaries directly and reports, per control:
- mean effect with seed-bootstrap CI, per-seed significance counts
- heterogeneity across directions: Cochran's Q (Monte Carlo p-value under the
  fixed-effect null, using each seed's block-bootstrap SE) and the
  DerSimonian-Laird between-direction SD tau
and for every control pair, paired sign-flip permutation tests on the signed
effect and on |effect|, plus a seed-bootstrap CI for the tau ratio.

Optionally compares direction hashes / effects against a reference sweep with
the same seeds (regeneration reproducibility).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np


def _seed(p: str) -> int:
    return int(Path(p).name.split("seed", 1)[1].split(".", 1)[0])


def load(seed_glob: str) -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {}
    for fp in sorted(glob.glob(seed_glob)):
        j = json.loads(Path(fp).read_text(encoding="utf-8"))
        for rec in j.get("results", []):
            fs = rec.get("final_summary") or {}
            eff = fs.get("block_bootstrap_abs_reduction_adj") or {}
            ci = eff.get("ci95") or [np.nan, np.nan]
            ds = fs.get("direction_shape_stats") or {}
            out.setdefault(_seed(fp), {})[str(rec.get("control"))] = {
                "matched": bool(rec.get("matched")),
                "alpha": float(fs.get("alpha", np.nan)),
                "mean": float(eff.get("mean", np.nan)),
                "lo": float(ci[0]),
                "hi": float(ci[1]),
                "se": (float(ci[1]) - float(ci[0])) / (2 * 1.959964),
                "abs_cos": float(ds.get("abs_cos_to_source", np.nan)),
                "sha": rec.get("direction_sha256") or fs.get("direction_sha256"),
                "target_cost": float(j.get("target_cost", np.nan)),
                "max_cost": float((j.get("max_reachable_cost_control") or {}).get(str(rec.get("control")), np.nan)),
            }
    return out


def cochran(x: np.ndarray, se: np.ndarray, rng: np.random.Generator, n_mc: int = 200_000) -> tuple[float, float, float]:
    w = 1.0 / se**2
    mu = np.sum(w * x) / np.sum(w)
    q = float(np.sum(w * (x - mu) ** 2))
    k = x.size
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    # Null: all directions share one effect; each observed mean ~ N(mu, se_i^2).
    sims = rng.normal(size=(n_mc, k)) * se
    smu = (sims * w).sum(1) / w.sum()
    sq = ((sims - smu[:, None]) ** 2 * w).sum(1)
    p = float((np.sum(sq >= q) + 1) / (n_mc + 1))
    return q, p, float(np.sqrt(tau2))


def signflip(d: np.ndarray, rng: np.random.Generator, n: int = 200_000) -> float:
    flips = rng.choice([-1.0, 1.0], size=(n, d.size))
    return float((np.sum(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-15) + 1) / (n + 1))


def tau_of(x: np.ndarray, se: np.ndarray) -> float:
    w = 1.0 / se**2
    mu = np.sum(w * x) / np.sum(w)
    q = np.sum(w * (x - mu) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return float(np.sqrt(max(0.0, (q - (x.size - 1)) / c)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--reference-glob", default=None, help="Earlier sweep with the same seeds, for reproducibility.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.rng_seed)
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    data = load(args.seed_glob)
    seeds = sorted(s for s in data if all(c in data[s] and data[s][c]["matched"] for c in controls))
    res: dict = {"n_seeds_total": len(data), "n_seeds_all_matched": len(seeds), "seeds": seeds, "controls": {}, "pairs": {}}
    print(f"seeds={len(data)} all-matched={len(seeds)}")
    for c in controls:
        n_matched = sum(1 for s in data if c in data[s] and data[s][c]["matched"])
        x = np.array([data[s][c]["mean"] for s in seeds])
        se = np.array([data[s][c]["se"] for s in seeds])
        lo = np.array([data[s][c]["lo"] for s in seeds])
        hi = np.array([data[s][c]["hi"] for s in seeds])
        boot = rng.choice(x, size=(20000, x.size)).mean(1)
        q, qp, tau = cochran(x, se, rng)
        r = {
            "matched": f"{n_matched}/{len(data)}",
            "mean": float(x.mean()),
            "seed_boot_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "mean_abs": float(np.abs(x).mean()),
            "sig_pos": int((lo > 0).sum()),
            "sig_neg": int((hi < 0).sum()),
            "median_within_se": float(np.median(se)),
            "between_sd": float(x.std(ddof=1)),
            "cochran_Q": q,
            "cochran_p_mc": qp,
            "tau": tau,
            "alpha_mean": float(np.mean([data[s][c]["alpha"] for s in seeds])),
            "abs_cos_mean": float(np.mean([data[s][c]["abs_cos"] for s in seeds])),
            "max_cost_median": float(np.median([data[s][c]["max_cost"] for s in seeds])),
        }
        res["controls"][c] = r
        print(f"\n## {c}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    print("\n## paired comparisons (b - a), over all-matched seeds")
    for a, b in itertools.combinations(controls, 2):
        xa = np.array([data[s][a]["mean"] for s in seeds])
        xb = np.array([data[s][b]["mean"] for s in seeds])
        sa = np.array([data[s][a]["se"] for s in seeds])
        sb = np.array([data[s][b]["se"] for s in seeds])
        d = xb - xa
        da = np.abs(xb) - np.abs(xa)
        idx = rng.integers(0, len(seeds), size=(5000, len(seeds)))
        ratios = []
        for ii in idx:
            ta = tau_of(xa[ii], sa[ii])
            ratios.append(tau_of(xb[ii], sb[ii]) / ta if ta > 0 else np.nan)
        ratios = np.asarray(ratios)
        ratios = ratios[np.isfinite(ratios)]
        tb, ta = res["controls"][b]["tau"], res["controls"][a]["tau"]
        r = {
            "mean_delta": float(d.mean()),
            "n_neg": int((d < 0).sum()),
            "n_pos": int((d > 0).sum()),
            "signflip_p": signflip(d, rng),
            "mean_abs_delta": float(da.mean()),
            "signflip_p_abs": signflip(da, rng),
            "tau_ratio": float(tb / ta) if ta > 0 else float("nan"),
            "tau_ratio_boot_ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))] if ratios.size else None,
        }
        res["pairs"][f"{b} - {a}"] = r
        print(f"\n{b} - {a}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    if args.reference_glob:
        ref = load(args.reference_glob)
        rep = {}
        for c in controls:
            common = [s for s in data if s in ref and c in data[s] and c in ref[s]]
            if not common:
                continue
            same_sha = sum(data[s][c]["sha"] == ref[s][c]["sha"] for s in common)
            diffs = [abs(data[s][c]["mean"] - ref[s][c]["mean"]) for s in common]
            rep[c] = {"n": len(common), "same_direction_sha": same_sha, "max_abs_effect_diff": float(max(diffs))}
        res["reproducibility_vs_reference"] = rep
        print("\n## reproducibility vs reference")
        print(json.dumps(rep, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
