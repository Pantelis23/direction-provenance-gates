#!/usr/bin/env python3
"""Exploratory: what do weight-orthogonalization edits look like after the
merge and quantization steps a heretic-style pipeline applies?

For unit directions v in the residual stream, edits every o_proj and
down_proj of Qwen3-0.6B with heretic's two update rules (p-e-w/heretic @ 3521f86,
model.py:abliterate):

  none : ΔW = -λ v (vᵀ W)                                   (rank 1)
  full : row-normalize W, apply the rank-1 update, renormalize rows, restore the
         original row norms, and keep a rank-3 randomized SVD (torch.svd_lowrank,
         q=2r+4, niter=6, seeded) of the resulting delta  (heretic's default)

Each edit is evaluated at the stages a user sees: the fp32 edit as optimised
(`exact`, for `full` that is the rank-3 LoRA delta kept separate), the bf16
merge, and llama.cpp-style Q8_0 / Q4_0 of the merge. Reported per stage:

- survive: ‖vᵀW'‖/‖vᵀW‖ pooled over edited matrices (0 = write along v fully removed)
- kl_vs_orig: mean next-token KL(original fp32 ‖ variant) on general text
- for reference, the same quantizers on the UNEDITED model

If quantization interacts badly with the edit, KL(edited+quant) will exceed
KL(edited) + KL(unedited+quant) by a wide margin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from quant_ablation_check import bf16, q4_0, q8_0  # noqa: E402


def full_delta(W: torch.Tensor, v: torch.Tensor, lam: float, r: int, seed: int) -> torch.Tensor:
    norms = W.norm(dim=1, keepdim=True)
    Wn = F.normalize(W, p=2, dim=1)
    A = (v @ Wn).view(1, -1)
    B = (-lam * v).view(-1, 1)
    Wt = F.normalize(Wn + B @ A, p=2, dim=1) * norms
    D = Wt - W
    torch.manual_seed(seed)
    U, S, V = torch.svd_lowrank(D, q=2 * r + 4, niter=6)
    return (U[:, :r] * S[:r]) @ V[:, :r].T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--corpus", default="data/wikitext103_test_paragraphs.json")
    ap.add_argument("--n-texts", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="runs/quant_ablation_functional.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    texts = json.loads((REPO / args.corpus).read_text())[: args.n_texts]
    enc = [tok(t, return_tensors="pt", truncation=True, max_length=args.max_tokens)["input_ids"].to(args.device) for t in texts]
    targets = {n: p for n, p in model.named_parameters() if n.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight"))}
    orig = {n: p.detach().clone() for n, p in targets.items()}

    @torch.no_grad()
    def logprobs() -> list[torch.Tensor]:
        return [F.log_softmax(model(input_ids=x).logits.float(), -1)[0] for x in enc]

    base = logprobs()

    @torch.no_grad()
    def kl_vs_base() -> float:
        tot, n = 0.0, 0
        for b, e in zip(base, logprobs()):
            tot += float((b.exp() * (b - e)).sum(-1).sum())
            n += b.shape[0]
        return tot / n

    @torch.no_grad()
    def load(mats: dict[str, torch.Tensor]) -> None:
        for n, p in targets.items():
            p.copy_(mats[n])

    results: dict = {"reference": {}, "edits": []}
    for qname, qf in (("bf16", bf16), ("q8_0", q8_0), ("q4_0", q4_0)):
        load({n: qf(W) for n, W in orig.items()})
        results["reference"][qname] = kl_vs_base()
        print(f"unedited {qname:5s} KL={results['reference'][qname]:.5f}", flush=True)

    d = next(iter(orig.values())).shape[0]
    g = torch.Generator().manual_seed(0)  # directions drawn on CPU so they do not depend on the device
    for k in range(args.n_dirs):
        v = torch.randn(d, generator=g)
        v = (v / v.norm()).to(args.device)
        for mode in ("none", "full"):
            exact = {}
            for i, (n, W) in enumerate(orig.items()):
                delta = -args.lam * torch.outer(v, v @ W) if mode == "none" else full_delta(W, v, args.lam, args.rank, seed=i)
                exact[n] = W + delta
            stages = {"exact": exact, "merged": {n: bf16(M) for n, M in exact.items()}}
            stages["q8_0"] = {n: q8_0(M) for n, M in stages["merged"].items()}
            stages["q4_0"] = {n: q4_0(M) for n, M in stages["merged"].items()}
            den = sum(float((v @ W).pow(2).sum()) for W in orig.values())
            row = {"dir": k, "mode": mode}
            for sname, mats in stages.items():
                surv = float(np.sqrt(sum(float((v @ M).pow(2).sum()) for M in mats.values()) / den))
                load(mats)
                kl = kl_vs_base()
                row[sname] = {"survive": surv, "kl_vs_orig": kl}
                print(f"dir={k} mode={mode:4s} {sname:6s} survive={surv:.4f} KL={kl:.5f}", flush=True)
            results["edits"].append(row)
    load(orig)
    (REPO / args.out_json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
