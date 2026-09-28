#!/usr/bin/env python3
"""Does anything measurable about a random control direction predict its effect?

The intervention edits the output of the LAST GPT-2 block, right before ln_f,
so the Jacobian lens of Anthropic's J-space paper reduces exactly to the logit
lens there: an edit v is read out as W_U (gamma * centre(v)) / sigma. For every
direction in a sweep (read from the hermetic .npy copies) this computes:

- logit_gender: logit-lens readout of v on male minus female pronoun/noun tokens
- act_contrast: E[(h_male - h_female) . v] over the gender_names_jobs sentences,
  on pre-ln_f states at the edited (non-name) positions
- proj_rms: sqrt(E[(h . v)^2]) at edited positions (how much there is to edit)

and correlates each (Spearman, sign-flip-free permutation p) with the
direction's signed bias change (mean_delta_adj) and its abs_red effect.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import transformers

REPO = Path(__file__).resolve().parents[1]
MALE = [" he", " him", " his", " himself", " man", " men", " boy", " male", " father", " son"]
FEMALE = [" she", " her", " hers", " herself", " woman", " women", " girl", " female", " mother", " daughter"]


def _rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 20000) -> tuple[float, float]:
    rx, ry = _rank(x), _rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return rho, float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n + 1))


def sentences(cfg: dict) -> list[tuple[str, str, str, str]]:
    import yaml  # noqa: F401  (config already parsed by caller)
    out = []
    for tpl in cfg["data"]["pll_templates"]:
        if tpl["name"] != "gender_names_jobs":
            continue
        for t in tpl["templates"]:
            for j in tpl["jobs"]:
                for a, b in tpl["group_pairs"]:
                    fill = lambda n: t.format(name=n, job=j["job"], article=j["article"])  # noqa: E731
                    out.append((fill(a), a, fill(b), b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--config", default="configs/mvp_jobs56_pronouns.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((REPO / args.config).read_text())
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    blocks = model.transformer.h
    ln_f = model.transformer.ln_f
    W_U = model.lm_head.weight.detach().float()  # [V, D]
    gamma = ln_f.weight.detach().float()

    # Pre-ln_f states at non-name positions, per sentence (mean) and pooled (all positions).
    cap = {}
    blocks[args.layer].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))

    def states(text: str, name: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        off = enc.pop("offset_mapping")[0].tolist()
        s, e = text.find(name), text.find(name) + len(name)
        keep = [i for i, (a, b) in enumerate(off) if not (a < e and b > s)]
        with torch.no_grad():
            model(**{k: v.to(args.device) for k, v in enc.items()})
        return cap["h"][0, keep]  # [T_nonname, D]

    sents = sentences(cfg)
    diff_means, all_h = [], []
    for ta, a, tb, b in sents:
        ha, hb = states(ta, a), states(tb, b)
        diff_means.append(ha.mean(0) - hb.mean(0))
        all_h.append(ha)
        all_h.append(hb)
    dmean = torch.stack(diff_means).mean(0)  # [D]
    H = torch.cat(all_h)  # [N, D]

    def ids(words: list[str]) -> list[int]:
        return [tok.encode(w)[0] for w in words if len(tok.encode(w)) == 1]

    male, female = ids(MALE), ids(FEMALE)

    rows = []
    for fp in sorted(glob.glob(str(REPO / args.seed_glob))):
        seed = int(Path(fp).name.split("seed", 1)[1].split(".", 1)[0])
        j = json.loads(Path(fp).read_text(encoding="utf-8"))
        for rec in j["results"]:
            fs = rec.get("final_summary") or {}
            vp = rec.get("direction_path_hermetic") or fs.get("direction_path_hermetic") or rec.get("direction_path") or fs.get("direction_path")
            v = torch.from_numpy(np.load(REPO / vp)).float().to(args.device)
            vc = v - v.mean()
            readout = W_U @ (gamma * vc)
            proj = H @ v
            eff = fs.get("block_bootstrap_abs_reduction_adj") or {}
            rows.append({
                "seed": seed, "control": rec["control"], "matched": bool(rec.get("matched")),
                "alpha": float(fs.get("alpha", np.nan)),
                "abs_red": float(eff.get("mean", np.nan)),
                "delta_adj": float(fs.get("mean_delta_adj", np.nan)),
                "logit_gender": float(readout[male].mean() - readout[female].mean()),
                "act_contrast": float(dmean @ v),
                "proj_rms": float(proj.pow(2).mean().sqrt()),
            })

    rng = np.random.default_rng(0)
    print(f"directions={len(rows)} sentences={len(sents)} male_ids={len(male)} female_ids={len(female)}")
    res = {"rows": rows, "correlations": {}}
    groups = {"all": rows}
    for c in sorted({r["control"] for r in rows}):
        groups[c] = [r for r in rows if r["control"] == c]
    for gname, g in groups.items():
        g = [r for r in g if r["matched"]]
        for target in ("delta_adj", "abs_red"):
            y = np.array([r[target] for r in g])
            for feat in ("logit_gender", "act_contrast", "proj_rms", "alpha"):
                x = np.array([r[feat] for r in g])
                for tf, xx, yy in (("signed", x, y), ("abs", np.abs(x), np.abs(y))):
                    rho, p = spearman(xx, yy, rng)
                    res["correlations"][f"{gname}|{target}|{feat}|{tf}"] = {"n": len(g), "rho": rho, "p": p}
                    flag = " *" if p < 0.05 else ""
                    print(f"{gname:42s} {target:9s} {feat:12s} {tf:6s} n={len(g):2d} rho={rho:+.3f} p={p:.4f}{flag}")
    if args.out_json:
        Path(REPO / args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
