#!/usr/bin/env python3
"""Build the arXiv LaTeX source (paper/main.tex) and PDF from paper.md.

paper.md stays the single source. This script:
  - lifts title, author line, repo link and abstract into LaTeX front matter
  - drops the hand-written section numbers ("## 5. Results" -> \\section{Results});
    LaTeX numbers sections in the same order, so "§5.3"-style references still match
  - turns each figure (image + following italic "*Figure N. ...*" caption) into one
    numbered LaTeX figure using the vector PDF
  - leaves References as an unnumbered section
then runs pandoc and (if available) tectonic.

    python3 scripts/build_latex.py [--tectonic PATH]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper"

HEADER = r"""
\usepackage[margin=0.95in]{geometry}
\ifPDFTeX
  \usepackage{newtxtext,newtxmath}
\else
  \setmainfont{texgyretermes-regular.otf}[
    BoldFont=texgyretermes-bold.otf,
    ItalicFont=texgyretermes-italic.otf,
    BoldItalicFont=texgyretermes-bolditalic.otf]
  \setsansfont{texgyreheros-regular.otf}[
    Scale=0.92,
    BoldFont=texgyreheros-bold.otf,
    ItalicFont=texgyreheros-italic.otf,
    BoldItalicFont=texgyreheros-bolditalic.otf]
  \setmonofont{lmmono10-regular.otf}[
    Scale=0.88,
    ItalicFont=lmmono10-italic.otf,
    BoldFont=lmmonolt10-bold.otf]
  \setmathfont{texgyretermes-math.otf}
\fi
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\},fontsize=\scriptsize}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize}
\usepackage{xurl}
\usepackage[htt]{hyphenat}
\usepackage{float}
\floatplacement{figure}{htb!}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.85}
\widowpenalty=10000
\clubpenalty=10000
\setlength{\emergencystretch}{3em}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
"""

SUP_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

UNICODE_MATH = [
    ("χ²", r"$\chi^2$"),
    ("r̄", r"$\bar{r}$"),
    (r"J\_ℓ", r"$J_\ell$"),
    ("J_ℓ", r"$J_\ell$"),
    ("²", r"$^2$"),
    ("±", r"$\pm$"),
    ("·", r"$\cdot$"),
    ("×", r"$\times$"),
    ("α", r"$\alpha$"),
    ("λ", r"$\lambda$"),
    ("ρ", r"$\rho$"),
    ("τ", r"$\tau$"),
    ("χ", r"$\chi$"),
    ("‖", r"$\|$"),
    ("ℓ", r"$\ell$"),
    ("←", r"$\leftarrow$"),
    ("−", r"$-$"),
    ("≈", r"$\approx$"),
    ("≤", r"$\le$"),
    ("≥", r"$\ge$"),
]


def tighten_tables(tex: str) -> str:
    """Replace Pandoc's equal-width p{...} columns with natural-width columns on compact tables."""
    def fix_table(m: re.Match) -> str:
        tbl = m.group(0)
        # Keep wrapped paragraph columns for prose-heavy tables (Threat model, Hypotheses)
        if "silent regeneration" in tbl or "block-bootstrap SEs are calibrated" in tbl:
            return tbl
        # Count columns from p{...} specs in \begin{longtable}[]{@{} ... @{}}
        spec_m = re.search(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", tbl, flags=re.S)
        if not spec_m:
            return tbl
        ncols = spec_m.group(1).count(r"\real{")
        if ncols == 0:
            return tbl
        # Choose compact column alignments that fit cleanly on one line per row
        if "empirical p" in tbl:
            # Effect-size reference (8 cols): natural widths + stacked headers, CIs on 1 line
            colspec = "@{}lllrrccr@{}"
        elif "abs-cos to source" in tbl:
            # MC block 11 (8 cols)
            colspec = "@{}lrrrrrrr@{}"
        elif "mean α" in tbl or r"mean $\alpha$" in tbl:
            # KL / ADD-N / ADD-KL (10 cols)
            colspec = "@{}lrlrrrrrrr@{}"
        elif "Q/df" in tbl:
            # P1 (9 cols)
            colspec = "@{}lrlrrrrrr@{}"
        elif "integrity gate, original" in tbl:
            # Fault injection (5 cols)
            colspec = "@{}lcccc@{}"
        elif "joint-bootstrap Q" in tbl:
            # Calibration (3 cols)
            colspec = "@{}lrr@{}"
        elif "cells heterogeneous" in tbl:
            # Pooled evidence (4 cols)
            colspec = "@{}lccc@{}"
        else:
            return tbl
        # Replace column spec
        tbl = tbl[:spec_m.start()] + f"\\begin{{longtable}}[]{{{colspec}}}" + tbl[spec_m.end():]
        # Strip Pandoc's \begin{minipage}[b]{\linewidth}\raggedright ... \end{minipage} in headers
        # for natural-width l/r/c columns (minipages with \linewidth break natural column width!)
        def unwrap_header(hm: re.Match) -> str:
            head = hm.group(1)
            head = re.sub(
                r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*(.*?)\s*\\end\{minipage\}",
                lambda cm: cm.group(1).replace("\n", " "),
                head,
                flags=re.S,
            )
            return r"\toprule\noalign{}" + "\n" + head + "\n" + r"\midrule\noalign{}"
        tbl = re.sub(r"\\toprule\\noalign\{\}(.*?)\\midrule\\noalign\{\}", unwrap_header, tbl, count=1, flags=re.S)
        # Wrap long header labels cleanly with \shortstack where needed
        if "empirical p" in tbl:
            tbl = tbl.replace("source effect {[}95\\% CI{]}", r"\shortstack[r]{source effect\\{[}95\% CI{]}}")
            tbl = tbl.replace("\\% of baseline \\textbar bias\\textbar{}", r"\shortstack[r]{\% of baseline\\\textbar bias\textbar}")
            tbl = tbl.replace("random \\textbar effect\\textbar: median / 90th pct", r"\shortstack{random \textbar effect\textbar:\\median / 90th}")
            tbl = tbl.replace("random directions with \\textbar effect\\textbar{} ≥ source", r"\shortstack{random \textbar effect\textbar{}\\$\ge$ source}")
        elif "abs-cos to source" in tbl:
            tbl = tbl.replace("mean effect {[}seed-bootstrap 95\\% CI{]}", r"\shortstack[r]{mean effect\\{[}95\% CI{]}}")
            tbl = tbl.replace("abs-cos to source", r"\shortstack[r]{abs-cos\\to source}")
        elif "cells heterogeneous" in tbl:
            tbl = tbl.replace("cells heterogeneous at p \\textless{} 0.05 (calibrated)", r"\shortstack{cells heterogeneous\\at p $<$ 0.05 (cal.)}")
        elif "integrity gate, original" in tbl:
            tbl = tbl.replace("integrity gate, original", r"\shortstack{integrity gate,\\original}")
            tbl = tbl.replace("integrity gate, hardened", r"\shortstack{integrity gate,\\hardened}")
            tbl = tbl.replace("CI: regenerate + verify", r"\shortstack{CI: regenerate\\+ verify}")
        elif "Q/df" in tbl:
            tbl = tbl.replace("p (calibrated)", r"p (cal.)")
            tbl = tbl.replace("p (χ², pre-reg.)", r"p ($\chi^2$, pre-reg.)")
        return tbl

    return re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", fix_table, tex, flags=re.S)


def unicode_to_tex(tex: str) -> str:
    """Convert Unicode math/Greek symbols in Pandoc-generated LaTeX to pdfLaTeX-safe TeX."""
    # Preserve verbatim code blocks untouched
    blocks = re.split(r"(\\begin\{Highlighting\}.*?\\end\{Highlighting\})", tex, flags=re.S)
    out = []
    for blk in blocks:
        if blk.startswith(r"\begin{Highlighting}"):
            out.append(blk)
            continue
        # Inside \texttt{...}, replace non-ASCII with ASCII equivalents
        blk = re.sub(
            r"\\texttt\{[^{}]*\}",
            lambda m: m.group(0).replace("−", "-").replace("…", "...").replace("·", "*"),
            blk,
        )
        # Allow line breaks after dots in long inline \texttt{...} expressions
        blk = blk.replace(
            r"\texttt{sha256(d.astype(float32).reshape(-1).tobytes())}",
            r"\texttt{sha256(d.\allowbreak astype(float32).\allowbreak reshape(-1).\allowbreak tobytes())}",
        )
        blk = re.sub(
            r"10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
            lambda m: f"$10^{{-{m.group(1).translate(SUP_DIGITS)}}}$",
            blk,
        )
        for src, dst in UNICODE_MATH:
            blk = blk.replace(src, dst)
        out.append(blk)
    return "".join(out)


def preprocess(md: str) -> tuple[dict, str]:
    lines = md.splitlines()
    assert lines[0].startswith("# "), "first line must be the title"
    meta = {"title": lines[0][2:].strip()}
    body = "\n".join(lines[1:])
    # author line and repo line, before "## Abstract"
    head, rest = body.split("## Abstract", 1)
    head_lines = [l.strip() for l in head.strip().splitlines() if l.strip()]
    meta["author"] = head_lines[0]
    meta["repo"] = next((re.search(r"https?://\S+", l).group(0) for l in head_lines[1:] if "http" in l), "")
    abstract, rest = rest.split("\n## ", 1)
    meta["abstract"] = abstract.strip()
    rest = "## " + rest
    # figures: ![alt](figures/x.png)\n\n*Figure N. caption*  ->  standalone paragraph ![caption](figures/x.pdf)
    def fig(m: re.Match) -> str:
        path = m.group(2).replace(".png", ".pdf")
        cap = re.sub(r"^Figure \d+\.\s*", "", m.group(3).strip())
        return f"![{cap}]({path})"
    rest = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)\s*\n\s*\n\*(Figure \d+\..*?)\*\s*\n", lambda m: fig(m) + "\n\n", rest, flags=re.S)
    # headings: drop manual numbers; demote one level (## -> #)
    rest = re.sub(r"^(#{2,4}) (\d+(?:\.\d+)*\.?) ", lambda m: m.group(1)[1:] + " ", rest, flags=re.M)
    rest = re.sub(r"^## References", "# References {-}", rest, flags=re.M)
    return meta, rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for f in (REPO / "figures").glob("*.pdf"):
        shutil.copy2(f, OUT / "figures" / f.name)
    meta, body = preprocess((REPO / "paper.md").read_text(encoding="utf-8"))
    (OUT / "body.md").write_text(body, encoding="utf-8")
    (OUT / "header.tex").write_text(HEADER, encoding="utf-8")
    abstract_tex = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex"], input=meta["abstract"],
                                  capture_output=True, text=True, check=True).stdout
    (OUT / "abstract.tex").write_text(unicode_to_tex(abstract_tex), encoding="utf-8")
    author = meta["author"] + (r"\\ \small\url{" + meta["repo"] + "}" if meta["repo"] else "")
    cmd = ["pandoc", "body.md", "-f", "markdown+pipe_tables", "-t", "latex", "-s", "-o", "main.tex",
           "-H", "header.tex", "--top-level-division=section", "-N",
           "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "-M", f"title={meta['title']}", "-V", f"author={author}", "-M", "date=",
           "-V", "abstract=\\input{abstract.tex}"]
    subprocess.run(cmd, cwd=OUT, check=True)
    tex = (OUT / "main.tex").read_text(encoding="utf-8")
    # pandoc escapes the \input in the abstract variable; restore it and place the abstract
    tex = tex.replace("\\textbackslash input\\{abstract.tex\\}", "\\input{abstract.tex}")
    tex = tighten_tables(tex)
    tex = unicode_to_tex(tex)
    (OUT / "main.tex").write_text(tex, encoding="utf-8")
    print("wrote", OUT / "main.tex")
    if args.tectonic:
        r = subprocess.run([args.tectonic, "--keep-logs", "main.tex"], cwd=OUT, capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-3000:])
        if r.returncode != 0:
            return r.returncode
        print("wrote", OUT / "main.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
