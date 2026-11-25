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
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{DejaVu Serif}[Scale=0.92]
\setsansfont{DejaVu Sans}[Scale=0.92]
\setmonofont{DejaVu Sans Mono}[Scale=0.80]
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\scriptsize\setlength{\tabcolsep}{3pt}}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\},fontsize=\scriptsize}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize}
\usepackage{xurl}
\usepackage[htt]{hyphenat}
\usepackage{float}
\floatplacement{figure}{htbp}
\setlength{\emergencystretch}{3em}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
"""


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
    # figures: ![alt](figures/x.png)\n\n*Figure N. caption*  ->  ![caption](figures/x.pdf)
    def fig(m: re.Match) -> str:
        path = m.group(2).replace(".png", ".pdf")
        cap = re.sub(r"^Figure \d+\.\s*", "", m.group(3).strip())
        return f"![{cap}]({path})"
    rest = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)\s*\n\s*\n\*(Figure \d+\..*?)\*\s*\n", lambda m: fig(m) + "\n", rest, flags=re.S)
    # headings: drop manual numbers; demote one level (## -> #)
    rest = re.sub(r"^(#{2,4}) (\d+(?:\.\d+)*\.?) ", lambda m: m.group(1)[1:] + " ", rest, flags=re.M)
    rest = re.sub(r"^## References", "# References {-}", rest, flags=re.M)
    # glyphs missing from DejaVu Serif: render as math outside code spans/blocks
    parts = re.split(r"(```.*?```|`[^`]*`)", rest, flags=re.S)
    rest = "".join(pt if pt.startswith("`") else pt.replace("ℓ", r"$\ell$") for pt in parts)
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
    (OUT / "abstract.tex").write_text(abstract_tex, encoding="utf-8")
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
