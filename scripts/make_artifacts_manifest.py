#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_workspace_relative(path: Path, cwd: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(cwd))
        except Exception:
            return str(p)
    return str(p)


def _resolve_from_cwd(path_str: str, cwd: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (cwd / p)


def _git_meta(cwd: Path) -> dict[str, object] | None:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    out: dict[str, object] = {"rev": rev}
    if dirty is not None:
        out["dirty"] = dirty
    return out


def _collect_patterns(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    # dedup+sort by normalized string path without resolve()
    uniq = sorted({str(Path(p)) for p in files})
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="matched_cost")
    ap.add_argument("--include-figures", action="append", default=[])
    ap.add_argument("--include-tables", action="append", default=[])
    args = ap.parse_args()

    cwd = Path.cwd()
    seed_files = sorted({str(Path(p)) for p in glob.glob(args.seed_glob)})
    if not seed_files:
        raise SystemExit(f"no files matched: {args.seed_glob}")

    fig_files = _collect_patterns(list(args.include_figures))
    table_files = _collect_patterns(list(args.include_tables))

    groups: dict[str, list[str]] = {}
    for p in seed_files:
        key = str(Path(p).parent)
        groups.setdefault(key, []).append(p)

    runs: list[dict[str, object]] = []
    for run_dir in sorted(groups.keys()):
        files = sorted(groups[run_dir])
        run: dict[str, object] = {
            "run_dir": run_dir,
            "kind": str(args.kind),
            "K": len(files),
            "seed_summaries": [],
        }

        for fp in files:
            p = Path(fp)
            summary = json.loads(p.read_text(encoding="utf-8"))
            recs: list[dict[str, object]] = []
            for rec in summary.get("results", []):
                fs = rec.get("final_summary") or {}
                ds = (fs.get("direction_shape_stats") or {})

                d_path_h = rec.get("direction_path_hermetic") or fs.get("direction_path_hermetic")
                d_path_ch = rec.get("direction_path_control_hermetic") or fs.get("direction_path_control_hermetic")

                raw_d = None
                raw_dc = None
                if d_path_h:
                    p_d = _resolve_from_cwd(str(d_path_h), cwd)
                    if p_d.is_file():
                        raw_d = _sha256_raw(p_d)
                if d_path_ch:
                    p_dc = _resolve_from_cwd(str(d_path_ch), cwd)
                    if p_dc.is_file():
                        raw_dc = _sha256_raw(p_dc)

                recs.append(
                    {
                        "control": rec.get("control"),
                        "direction_sha256_semantic": rec.get("direction_sha256") or fs.get("direction_sha256"),
                        "direction_path_hermetic": d_path_h,
                        "direction_path_control_hermetic": d_path_ch,
                        "sha256_raw_direction_npy": raw_d,
                        "sha256_raw_direction_control_npy": raw_dc,
                        "support_stats": {
                            "masked_support_mass": ds.get("masked_support_mass"),
                            "support_intrusion_eps": ds.get("support_intrusion_eps"),
                            "support_intrusion_frac_eps": ds.get("support_intrusion_frac_eps"),
                        },
                    }
                )

            run["seed_summaries"].append(
                {
                    "path": _to_workspace_relative(p, cwd),
                    "sha256_raw": _sha256_raw(p),
                    "records": recs,
                }
            )

        runs.append(run)

    extras: dict[str, object] = {
        "figures": [
            {
                "path": _to_workspace_relative(Path(fp), cwd),
                "sha256_raw": _sha256_raw(_resolve_from_cwd(fp, cwd)),
            }
            for fp in fig_files
        ],
        "tables": [
            {
                "path": _to_workspace_relative(Path(tp), cwd),
                "sha256_raw": _sha256_raw(_resolve_from_cwd(tp, cwd)),
            }
            for tp in table_files
        ],
    }

    out: dict[str, object] = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "runs": runs,
        "extras": extras,
    }
    git = _git_meta(cwd)
    if git is not None:
        out["git"] = git

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"runs={len(runs)} seed_summaries={sum(len(r['seed_summaries']) for r in runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
