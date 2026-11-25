#!/usr/bin/env python3
"""Paper figures.

fig_random_directions.(pdf|png): every random direction's effect (abs_red, nats)
  with its 95% block-bootstrap CI, per model (rows) x condition (columns) x block,
  the three controls offset with distinct markers, and the learned source
  direction (post hoc reference run) as a black diamond where available.
fig_fault_matrix.(pdf|png): the Section 4 fault-detection matrix, parsed from
  paper.md so figure and table cannot disagree.

Colours: the validated categorical slots 1-3 (blue, orange, aqua; all-pairs
CVD-safe) for controls, reserved status colours for caught/missed, with text
labels / marker shapes so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
CTRL = [("orthogonal_random", "orth", "#2a78d6", "o"),
        ("orthogonal_random_support_avoid_top512", "support-avoid", "#eb6834", "s"),
        ("orthogonal_random_random_mask_top512", "random-mask", "#1baf7a", "^")]
GOOD, CRIT, NEUTRAL = "#0ca30c", "#d03b3b", "#e8e7e3"
MODELS = [("GPT-2", "gpt2", [(1, "early"), (6, "mid"), (11, "late")]),
          ("Qwen3-0.6B", "qwen", [(1, None), (14, None), (26, None)]),
          ("Llama-3.2-1B", "llama", [(1, None), (8, None), (14, None)])]
CONDS = [("P1", "unit projection (α = 1)"), ("KL", "KL-matched projection"),
         ("ADD-N", "additive, P1-sized (ADD-N)"), ("ADD-KL", "additive, KL-matched")]


def sweep_dir(cond: str, m: str, L: int, tag: str | None) -> str | None:
    if cond == "P1":
        return {"gpt2": f"p1_{tag}", "qwen": f"qwen_p1_L{L}", "llama": f"llama_p1_L{L}"}[m]
    if cond == "KL":
        return {"gpt2": f"kl_gpt2_{tag}", "qwen": f"kl_qwen_L{L}", "llama": f"kl_llama_L{L}"}[m]
    if m == "llama":
        return None
    return f"{'addn' if cond == 'ADD-N' else 'addkl'}_{m}_L{L}"


def source_dir(cond: str, m: str, L: int) -> str:
    return f"src_{ {'P1': 'p1', 'KL': 'kl', 'ADD-N': 'addn', 'ADD-KL': 'addkl'}[cond] }_{m}_L{L}"


def effects(d: str, control: str) -> list[tuple[float, float, float]]:
    out = []
    for f in sorted(glob.glob(str(REPO / "runs" / d / "seed*.matched_cost_summary.json"))):
        for r in json.loads(Path(f).read_text())["results"]:
            if r["control"] == control:
                e = r["final_summary"]["block_bootstrap_abs_reduction_adj"]
                out.append((e["mean"], e["ci95"][0], e["ci95"][1]))
    return out


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_random_directions(out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.2), sharey="col", facecolor=SURF)
    for ci, (cond, ctitle) in enumerate(CONDS):
        for mi, (mname, m, layers) in enumerate(MODELS):
            ax = axes[mi, ci]
            style(ax)
            ax.axhline(0, color=INK2, linewidth=0.8)
            any_data = False
            for li, (L, tag) in enumerate(layers):
                d = sweep_dir(cond, m, L, tag)
                if d is None:
                    continue
                for k, (key, name, col, mk) in enumerate(CTRL):
                    e = effects(d, key)
                    if not e:
                        continue
                    any_data = True
                    x0 = li + (k - 1) * 0.24
                    xs = x0 + rng.uniform(-0.07, 0.07, len(e))
                    for x, (mu, lo, hi) in zip(xs, e):
                        ax.plot([x, x], [lo, hi], color=col, linewidth=0.6, alpha=0.45, solid_capstyle="round")
                    ax.scatter(xs, [v[0] for v in e], s=13, marker=mk, color=col, edgecolor=SURF, linewidth=0.5, zorder=3)
                src = effects(source_dir(cond, m, L), "none")
                if src:
                    mu, lo, hi = src[0]
                    ax.plot([li + 0.42, li + 0.42], [lo, hi], color=INK, linewidth=1.0)
                    ax.scatter([li + 0.42], [mu], s=34, marker="D", color=INK, edgecolor=SURF, linewidth=0.7, zorder=4)
            if not any_data:
                ax.axis("off")
                ax.text(0.5, 0.5, "not run\n(additive edits were\nnot tested on Llama)", transform=ax.transAxes,
                        ha="center", va="center", color=INK2, fontsize=8)
            else:
                ax.set_xticks(range(len(layers)))
                ax.set_xticklabels([f"block {L}" for L, _ in layers])
                ax.set_xlim(-0.6, len(layers) - 0.35)
            if mi == 0:
                ax.set_title(ctitle, fontsize=9, color=INK, pad=6)
            if ci == 0:
                ax.set_ylabel(f"{mname}\nabs_red (nats)", fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker=mk, linestyle="", color=col, markersize=5, label=f"{name} (16 random directions, 95% CI)") for _, name, col, mk in CTRL]
    handles.append(plt.Line2D([], [], marker="D", linestyle="", color=INK, markersize=5, label="learned source direction (post hoc reference)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle("Effects of individual random control directions on the bias metric (abs_red; positive = bias reduced)", fontsize=10.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.985), h_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_random_directions.{ext}", dpi=200, facecolor=SURF)
    plt.close(fig)


def parse_fault_table() -> tuple[list[str], list[str], list[list[str]]]:
    s = (REPO / "paper.md").read_text()
    sec = s[s.index("## 4. Evaluating the gates"):s.index("## 5.")]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[2:]:
        cells = [re.sub(r"\*\*|`", "", c).strip() for c in l.strip("|").split("|")]
        if cells[0].lower().startswith("caught (of"):
            continue
        rows.append(cells)
    return header[1:], [r[0] for r in rows], [r[1:] for r in rows]


def fig_fault_matrix(out: Path) -> None:
    gates, faults, M = parse_fault_table()
    gates = [g.replace("integrity gate, ", "integrity gate\n").replace("CI: regenerate + verify", "CI: regenerate\n+ verify") for g in gates]
    n_r, n_c = len(faults), len(gates)
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * n_r + 1.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            v = v.lower()
            col, txt, tc = (GOOD, "caught", "white") if v == "caught" else (CRIT, "missed", "white") if v == "missed" else (NEUTRAL, v, INK)
            ax.add_patch(plt.Rectangle((j + 0.04, n_r - 1 - i + 0.06), 0.92, 0.88, color=col, linewidth=0))
            ax.text(j + 0.5, n_r - 1 - i + 0.5, txt, ha="center", va="center", fontsize=7.5, color=tc)
    totals = [sum(1 for r in M if r[j].lower() == "caught") for j in range(n_c)]
    for j, t in enumerate(totals):
        ax.text(j + 0.5, -0.45, f"{t} / {n_r - 1} caught", ha="center", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xlim(0, n_c)
    ax.set_ylim(-0.9, n_r)
    ax.set_xticks([j + 0.5 for j in range(n_c)])
    ax.set_xticklabels(gates, fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([n_r - 1 - i + 0.5 for i in range(n_r)])
    ax.set_yticklabels(faults, fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Which check catches which injected fault", fontsize=10, color=INK, pad=34)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.86, bottom=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_fault_matrix.{ext}", dpi=200, facecolor=SURF)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fig_fault_matrix(out)
    fig_random_directions(out)
    print("wrote", sorted(p.name for p in out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
