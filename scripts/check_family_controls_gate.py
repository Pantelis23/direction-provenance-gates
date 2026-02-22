#!/usr/bin/env python3
import re
from pathlib import Path


def get_block(text: str, name: str) -> str:
    m = re.search(rf"## {re.escape(name)}\n(.*?)(?=\n## |\Z)", text, flags=re.S)
    if not m:
        raise SystemExit(f"missing block: {name}")
    return m.group(1)


def parse_ci(block: str) -> tuple[float, float]:
    m = re.search(r"seed_boot_CI95\(mean\): \[([+\-0-9\.eE]+), ([+\-0-9\.eE]+)\]", block)
    if not m:
        raise SystemExit("missing CI")
    return float(m.group(1)), float(m.group(2))


def parse_sig_pos(block: str) -> tuple[int, int]:
    m = re.search(r"sig_pos\(ci_lo>0\): (\d+)/(\d+)", block)
    if not m:
        raise SystemExit("missing sig_pos")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    p = Path("runs/mc_shuf_matched_mcK16_family_controls.md")
    txt = p.read_text(encoding="utf-8")

    b_none = get_block(txt, "none")
    sig, k = parse_sig_pos(b_none)
    if sig != k:
        raise SystemExit(f"none sig_pos expected {k}/{k}, got {sig}/{k}")

    b_sh = get_block(txt, "shuffled")
    lo, hi = parse_ci(b_sh)
    if not (lo < 0 < hi):
        raise SystemExit(f"shuffled mean CI should cross 0, got [{lo},{hi}]")

    b_orth = get_block(txt, "orthogonal_random")
    lo, hi = parse_ci(b_orth)
    if not (lo < 0 < hi):
        raise SystemExit(f"orthogonal_random mean CI should cross 0, got [{lo},{hi}]")

    b_am = get_block(txt, "abs_marginal_matched_random_orth")
    lo, hi = parse_ci(b_am)
    if not (lo < 0 < hi):
        raise SystemExit(f"abs_marginal mean CI should cross 0, got [{lo},{hi}]")

    print("OK: family-controls gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
