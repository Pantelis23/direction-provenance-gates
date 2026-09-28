#!/usr/bin/env python3
"""H4 of docs/prereg_2026-09-28.md: does a random direction's propagation
through the Jacobian lens predict its effect?

Fits (or loads) a Jacobian lens for GPT-2 at the requested source layers using
anthropics/jacobian-lens, then for every direction in the given sweeps computes
A(v) = ||J_l v|| (v is unit) and the Spearman correlation between A(v) and
|effect| (one-sided permutation p, H4 predicts rho > 0).

    --sweep LAYER=GLOB   repeatable; LAYER is the integer source layer
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens-src", required=True, help="Path to a checkout of anthropics/jacobian-lens.")
    ap.add_argument("--layers", default="1,6")
    ap.add_argument("--corpus", default="data/wikitext103_test_paragraphs.json")
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--lens-path", default="runs/jlens_gpt2.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.jlens_src)
    import jlens
    import transformers

    layers = [int(x) for x in args.layers.split(",")]
    lens_path = REPO / args.lens_path
    meta_path = lens_path.with_suffix(".meta.json")
    if lens_path.exists():
        lens = jlens.JacobianLens.load(str(lens_path))
        print(f"loaded {lens_path}: {lens}")
    else:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        hf = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(args.device).eval()
        model = jlens.from_hf(hf, tok)
        corpus_bytes = (REPO / args.corpus).read_bytes()
        prompts = json.loads(corpus_bytes)[: args.n_prompts]
        lens = jlens.fit(model, prompts=prompts, source_layers=layers, max_seq_len=128,
                         checkpoint_path=str(lens_path) + ".ckpt", checkpoint_every=25)
        lens.save(str(lens_path), dtype=torch.float32)
        meta_path.write_text(json.dumps({
            "model": "gpt2", "source_layers": layers, "n_prompts": len(prompts), "max_seq_len": 128,
            "corpus": args.corpus, "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "lens_sha256": hashlib.sha256(lens_path.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
        Path(str(lens_path) + ".ckpt").unlink(missing_ok=True)
        print(f"fitted and saved {lens_path}: {lens}")

    rng = np.random.default_rng(0)
    res: dict = {"lens": str(args.lens_path), "tests": {}}
    for spec in args.sweep:
        layer_s, pat = spec.split("=", 1)
        layer = int(layer_s)
        J = lens.jacobians[layer].float()
        rows = []
        for fp in sorted(glob.glob(str(REPO / pat))):
            for rec in json.loads(Path(fp).read_text(encoding="utf-8"))["results"]:
                if not rec.get("matched"):
                    continue
                fs = rec.get("final_summary") or {}
                vp = rec.get("direction_path_hermetic") or rec.get("direction_path") or fs.get("direction_path")
                v = torch.from_numpy(np.load(REPO / vp)).float()
                eff = (fs.get("block_bootstrap_abs_reduction_adj") or {}).get("mean")
                rows.append({"file": Path(fp).name, "control": rec["control"],
                             "A": float((J @ v).norm()), "abs_effect": abs(float(eff))})
        x = np.array([r["A"] for r in rows])
        y = np.array([r["abs_effect"] for r in rows])
        rx, ry = _rank(x), _rank(y)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(20000)])
        p_one = float((np.sum(perm >= rho) + 1) / (len(perm) + 1))
        res["tests"][spec] = {"n": len(rows), "rho": rho, "p_one_sided": p_one,
                              "A_mean": float(x.mean()), "A_cv": float(x.std() / x.mean()), "rows": rows}
        print(f"{spec}: n={len(rows)} rho={rho:+.3f} p(one-sided, rho>0)={p_one:.4f} A mean={x.mean():.3f} cv={x.std()/x.mean():.3f}")
    if args.out_json:
        (REPO / args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
