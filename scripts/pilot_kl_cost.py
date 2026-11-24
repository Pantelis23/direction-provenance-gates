#!/usr/bin/env python3
"""Cost-only pilot for the KL-matched design (prereg addendum B).

Measures mean next-token KL(original||edited) on the general-text corpus for
pilot directions (seeds disjoint from the main 1-16) at a grid of alphas, per
layer. Used only to choose a fixed per-layer KL target that every control can
reach; it never looks at the bias effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_causal import (  # noqa: E402
    _make_orthogonal_random_direction,
    _make_orthogonal_random_mask_direction,
    _make_orthogonal_support_avoid_direction,
)
from src.metrics.pll import PLLIntervention, PLLScorer  # noqa: E402

MAKERS = {
    "orthogonal_random": lambda s, r: _make_orthogonal_random_direction(s, r),
    "orthogonal_random_support_avoid_top512": lambda s, r: _make_orthogonal_support_avoid_direction(s, r, topk=512),
    "orthogonal_random_random_mask_top512": lambda s, r: _make_orthogonal_random_mask_direction(s, r, topk=512),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer-dirs", required=True, help="comma list LAYER=path/to/control_dir.npy")
    ap.add_argument("--seeds", default="101,102")
    ap.add_argument("--alphas", default="1,5,20,100")
    ap.add_argument("--corpus", default="data/wikitext103_test_paragraphs.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    scorer = PLLScorer(cfg["model"])
    texts = json.loads((REPO / args.corpus).read_text())[: args.n]
    rows = []
    for spec in args.layer_dirs.split(","):
        layer_s, dpath = spec.split("=", 1)
        layer = int(layer_s)
        src = np.load(REPO / dpath).astype(np.float32)
        src = src / np.linalg.norm(src)
        for seed in [int(s) for s in args.seeds.split(",")]:
            for control, make in MAKERS.items():
                v = make(src, np.random.RandomState(seed))
                v = (v / np.linalg.norm(v)).astype(np.float32)
                for a in [float(x) for x in args.alphas.split(",")]:
                    kl = scorer.mean_next_token_kl(texts, PLLIntervention(layer=layer, direction=v, mode="project_out", alpha=a), args.max_tokens)
                    rows.append({"layer": layer, "seed": seed, "control": control, "alpha": a, "kl": kl["mean_kl"]})
                    print(f"L{layer} seed={seed} {control[-24:]:24s} alpha={a:6.1f} KL={kl['mean_kl']:.5f}", flush=True)
    (REPO / args.out_json).write_text(json.dumps({"config": args.config, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
