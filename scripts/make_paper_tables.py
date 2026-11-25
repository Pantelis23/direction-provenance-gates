#!/usr/bin/env python3
"""Generate the Section 5 result tables and counts from the analysis outputs.

Reads runs/<cell>_analysis.json (analyze_mc_controls.py) and
runs/<cell>_jointblock.json (joint_block_test.py) and writes markdown snippets
plus a JSON of every count quoted in the paper, so no number is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CTRL = {"orthogonal_random": "orth", "orthogonal_random_support_avoid_top512": "support-avoid",
        "orthogonal_random_random_mask_top512": "random-mask"}
MODELS = {"GPT-2": [(1, "early"), (6, "mid"), (11, "late")], "Qwen": [(1, None), (14, None), (26, None)],
          "Llama": [(1, None), (8, None), (14, None)]}


def cell(cond: str, model: str, L: int, tag: str | None) -> str | None:
    if cond == "P1":
        return {"GPT-2": f"p1_{tag}", "Qwen": f"qwen_p1_L{L}", "Llama": f"llama_p1_L{L}"}[model]
    if cond == "KL":
        return {"GPT-2": f"kl_gpt2_{tag}", "Qwen": f"kl_qwen_L{L}", "Llama": f"kl_llama_L{L}"}[model]
    if cond in ("ADD-N", "ADD-KL") and model != "Llama":
        pre = "addn" if cond == "ADD-N" else "addkl"
        return f"{pre}_{'gpt2' if model == 'GPT-2' else 'qwen'}_L{L}"
    return None


def load(c: str) -> tuple[dict, dict]:
    a = json.loads((REPO / f"runs/{c}_analysis.json").read_text())["controls"]
    j = json.loads((REPO / f"runs/{c}_jointblock.json").read_text())["controls"]
    return a, j


def fp(x: float) -> str:
    return "< 10⁻⁴" if x < 1e-4 else f"{x:.4f}"


def fm(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


def main() -> int:
    out_md: dict[str, str] = {}
    counts: dict = {}
    for cond in ("P1", "KL", "ADD-N", "ADD-KL"):
        rows = ["| model | block | control | mean effect | mean \\|effect\\| | τ | Q/df | p (calibrated) | p (χ², pre-reg.) |" + (" mean α |" if cond != "P1" else ""),
                "|---|---|---|---|---|---|---|---|---|" + ("---|" if cond != "P1" else "")]
        c05 = cpre05 = cbon = cprebon = n = 0
        per_model: dict = {}
        for model, layers in MODELS.items():
            for L, tag in layers:
                c = cell(cond, model, L, tag)
                if c is None:
                    continue
                a, j = load(c)
                for k, short in CTRL.items():
                    r, q = a[k], j[k]
                    n += 1
                    pc, pp = q["p_joint_Q"], r["cochran_p_mc"]
                    c05 += pc < 0.05
                    cpre05 += pp < 0.05
                    per_model.setdefault(model, [0, 0])
                    per_model[model][0] += pc < 0.05
                    per_model[model][1] += 1
                    line = f"| {model} | {L} | {short} | {fm(r['mean'])} | {r['mean_abs']:.4f} | {r['tau']:.4f} | {r['cochran_Q']/15:.2f} | {fp(pc)} | {fp(pp)} |"
                    if cond != "P1":
                        line += f" {r['alpha_mean']:.1f} |"
                    rows.append(line)
        for model, layers in MODELS.items():  # Bonferroni counts over the cells of this condition
            pass
        bon = 0.05 / n
        for model, layers in MODELS.items():
            for L, tag in layers:
                c = cell(cond, model, L, tag)
                if c is None:
                    continue
                a, j = load(c)
                for k in CTRL:
                    cbon += j[k]["p_joint_Q"] < bon
                    cprebon += a[k]["cochran_p_mc"] < bon
        out_md[cond] = "\n".join(rows)
        counts[cond] = {"n_cells": n, "calibrated_p05": c05, "prereg_chi2_p05": cpre05, "bonferroni_alpha": bon,
                        "calibrated_bonferroni": cbon, "prereg_chi2_bonferroni": cprebon,
                        "calibrated_p05_by_model": {m: f"{v[0]}/{v[1]}" for m, v in per_model.items()}}
    # tau ratios KL vs P1 (all models)
    ratios = []
    for model, layers in MODELS.items():
        for L, tag in layers:
            ak, _ = load(cell("KL", model, L, tag))
            ap, _ = load(cell("P1", model, L, tag))
            for k in CTRL:
                ratios.append(ak[k]["tau"] / ap[k]["tau"] if ap[k]["tau"] > 0 else float("inf"))
    counts["tau_KL_over_P1"] = {"n": len(ratios), "greater_than_1": sum(r > 1 for r in ratios),
                                "min_finite": min(r for r in ratios if r != float("inf")),
                                "max_finite": max(r for r in ratios if r != float("inf")),
                                "n_infinite": sum(r == float("inf") for r in ratios)}
    # block-1 P1 detail
    counts["P1_block1_calibrated_p05"] = sum(
        load(cell("P1", m, MODELS[m][0][0], MODELS[m][0][1]))[1][k]["p_joint_Q"] < 0.05 for m in MODELS for k in CTRL)
    cal = json.loads((REPO / "runs/homogeneity_calibration.json").read_text())
    counts["null_calibration"] = cal
    (REPO / "runs/paper_tables.json").write_text(json.dumps({"counts": counts, "tables": out_md}, indent=2))
    print(json.dumps(counts, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
