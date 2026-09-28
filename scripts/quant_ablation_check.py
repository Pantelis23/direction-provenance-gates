#!/usr/bin/env python3
"""Exploratory: does quantization undo a weight-orthogonalization ("abliteration") edit?

Applies the rank-1 edit used by heretic (ΔW = -λ v vᵀ W on every o_proj and
down_proj, i.e. the residual-writing matrices) and pushes the result through:

  exact    fp32 W - λ v vᵀ W                           (the intended model)
  lora     bf16(W) + (λ v)(vᵀ W) kept as a separate path (what heretic scores)
  merged   bf16(bf16(W) + ΔW)                          (what heretic saves)
  q8_0     llama.cpp Q8_0 of merged (32-blocks, amax/127)
  q4_0     llama.cpp Q4_0 of merged (32-blocks, signed-extremum/-8)

and the same quantizers applied to the UNEDITED model as a reference. For each
variant it reports the surviving write along v, ‖vᵀ W'‖ / ‖vᵀ W‖ (0 for an exact
λ=1 ablation), pooled over all edited matrices. Directions: random unit
directions (the control distribution), plus directions concentrated on the
residual stream's high-magnitude ("massive activation") coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]


def q8_0(W: torch.Tensor) -> torch.Tensor:
    o, i = W.shape
    b = W.reshape(o, i // 32, 32)
    d = b.abs().amax(-1, keepdim=True) / 127.0
    q = torch.where(d > 0, torch.round(b / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(b)).clamp(-127, 127)
    return (q * d).half().float().reshape(o, i) if False else (q * d.half().float()).reshape(o, i)


def q4_0(W: torch.Tensor) -> torch.Tensor:
    o, i = W.shape
    b = W.reshape(o, i // 32, 32)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = torch.gather(b, -1, idx)
    d = (mx / -8.0).half().float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.round(b * inv + 8.0), 0, 15)
    return ((q - 8.0) * d).reshape(o, i)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"))
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out-json", default="runs/quant_ablation_check.json")
    args = ap.parse_args()

    wpath = next(Path(args.weights).glob("*/model.safetensors")) if Path(args.weights).is_dir() else Path(args.weights)
    f = safe_open(str(wpath), framework="pt")
    keys = [k for k in f.keys() if k.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight"))]
    mats = {k: f.get_tensor(k).float() for k in sorted(keys)}  # stored bf16 -> exact in fp32
    d = next(iter(mats.values())).shape[0]

    # "Massive-activation" coordinates: largest mean |row| magnitude of the embedding matrix
    # is a weights-only proxy; use the residual-writing matrices' row norms summed over layers.
    row_energy = sum(W.pow(2).sum(1) for W in mats.values())
    top = torch.topk(row_energy, 8).indices

    dirs = {}
    g = torch.Generator().manual_seed(0)
    for s in range(args.n_random):
        v = torch.randn(d, generator=g)
        dirs[f"random_{s}"] = v / v.norm()
    for s in range(4):
        v = torch.zeros(d)
        v[top] = torch.randn(len(top), generator=g)
        dirs[f"outlier_{s}"] = v / v.norm()

    ref = {"q8_0": 0.0, "q4_0": 0.0, "bf16": 0.0}
    rows = []
    for name, v in dirs.items():
        num = {k: 0.0 for k in ("exact", "lora", "merged", "q8_0", "q4_0", "orig_q8_0", "orig_q4_0")}
        den = 0.0
        rel_delta = []
        for W in mats.values():
            a = v @ W                              # vᵀW, [d_in]
            den += float(a.pow(2).sum())
            delta = -args.lam * torch.outer(v, a)
            rel_delta.append(float(delta.abs().mean() / W.abs().mean()))
            exact = W + delta
            merged = bf16(W + delta)               # W is already bf16-exact
            outs = {
                "exact": exact,
                "merged": merged,
                "q8_0": q8_0(merged),
                "q4_0": q4_0(merged),
                "orig_q8_0": q8_0(W),
                "orig_q4_0": q4_0(W),
            }
            for k, M in outs.items():
                num[k] += float((v @ M).pow(2).sum())
            # LoRA path as evaluated: bf16 base + separate fp32 low-rank term, so vᵀ(W + BA) in fp32.
            num["lora"] += float((v @ (W + torch.outer(-args.lam * v, a))).pow(2).sum())
        r = {k: float(np.sqrt(num[k] / den)) for k in num}
        r.update({"direction": name, "mean_rel_delta": float(np.mean(rel_delta))})
        rows.append(r)
        print(f"{name:11s} surviving write along v: exact={r['exact']:.4f} lora={r['lora']:.4f} merged={r['merged']:.4f} "
              f"q8_0={r['q8_0']:.4f} q4_0={r['q4_0']:.4f} | unedited q8_0={r['orig_q8_0']:.4f} q4_0={r['orig_q4_0']:.4f} "
              f"| |ΔW|/|W|={r['mean_rel_delta']:.4f}", flush=True)

    def agg(prefix: str) -> dict:
        sel = [r for r in rows if r["direction"].startswith(prefix)]
        return {k: [float(np.mean([r[k] for r in sel])), float(np.min([r[k] for r in sel])), float(np.max([r[k] for r in sel]))]
                for k in ("exact", "lora", "merged", "q8_0", "q4_0", "mean_rel_delta")}

    summary = {"random": agg("random"), "outlier": agg("outlier"), "outlier_coords": top.tolist(), "lam": args.lam,
               "weights": str(wpath), "n_matrices": len(mats)}
    print(json.dumps({k: summary[k] for k in ("random", "outlier")}, indent=1))
    (REPO / args.out_json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
