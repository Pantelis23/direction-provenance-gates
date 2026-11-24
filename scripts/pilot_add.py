#!/usr/bin/env python3
"""Cost-only pilot for the additive-steering conditions (prereg addendum C).

For pilot directions (seeds disjoint from the main 1-16) at each layer:
  - disp: E|h·v| over every position of the general-text corpus, i.e. the size of
    the displacement a unit projection (P1) makes along v. Its median over pilot
    directions fixes the per-layer alpha of ADD-N (matched displacement).
  - KL(original || h + alpha v) on the same corpus over a grid of alphas, to check
    that the 0.10 KL target is reachable and to set alpha_max for ADD-KL.
Never looks at the bias effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_kl_cost import MAKERS  # noqa: E402
from src.metrics.pll import PLLIntervention, PLLScorer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer-dirs", required=True, help="comma list LAYER=path/to/control_dir.npy")
    ap.add_argument("--seeds", default="101,102")
    ap.add_argument("--alphas", default="1,4,16,64,256,1024")
    ap.add_argument("--corpus", default="data/wikitext103_test_paragraphs.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    scorer = PLLScorer(cfg["model"])
    texts = json.loads((REPO / args.corpus).read_text())[: args.n]
    blocks = scorer._resolve_layers()
    rows = []
    for spec in args.layer_dirs.split(","):
        layer_s, dpath = spec.split("=", 1)
        layer = int(layer_s)
        cap: dict = {}
        h = blocks[layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))
        acts = []
        with torch.no_grad():
            for t in texts:
                ids = scorer.tokenizer(t, return_tensors="pt", truncation=True, max_length=args.max_tokens)["input_ids"].to(scorer.device)
                scorer.model(input_ids=ids)
                acts.append(cap["h"][0])
        h.remove()
        H = torch.cat(acts)  # [N, D]
        resid_rms = float(H.norm(dim=-1).pow(2).mean().sqrt())
        src = np.load(REPO / dpath).astype(np.float32)
        src = src / np.linalg.norm(src)
        for seed in [int(s) for s in args.seeds.split(",")]:
            for control, make in MAKERS.items():
                v = make(src, np.random.RandomState(seed))
                v = (v / np.linalg.norm(v)).astype(np.float32)
                disp = float((H @ torch.from_numpy(v).to(H.device)).abs().mean())
                row = {"layer": layer, "seed": seed, "control": control, "disp": disp, "resid_rms": resid_rms, "kl": {}}
                for a in [float(x) for x in args.alphas.split(",")]:
                    row["kl"][a] = scorer.mean_next_token_kl(texts, PLLIntervention(layer=layer, direction=v, mode="add", alpha=a), args.max_tokens)["mean_kl"]
                rows.append(row)
                print(f"L{layer} seed={seed} {control[-24:]:24s} disp={disp:.3f} rms={resid_rms:.1f} "
                      + " ".join(f"a{a:g}:{k:.4f}" for a, k in row["kl"].items()), flush=True)
    (REPO / args.out_json).write_text(json.dumps({"config": args.config, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
