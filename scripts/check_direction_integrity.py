#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _resolve_path(path_str: str | None, summary_path: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(str(path_str))
    if p.is_absolute():
        return p if p.is_file() else None

    # Most records are workspace-relative paths.
    p1 = (Path.cwd() / p).resolve()
    if p1.is_file():
        return p1

    # Fallback for unusual relative paths.
    p2 = (summary_path.parent / p).resolve()
    if p2.is_file():
        return p2
    return None


def _sha256_float32_npy(path: Path) -> str:
    arr = np.load(path, allow_pickle=False)
    v = np.asarray(arr, dtype=np.float32).reshape(-1)
    return hashlib.sha256(v.tobytes()).hexdigest()


def _sha256_raw_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_glob_chars(s: str) -> bool:
    return any(c in s for c in ("*", "?", "[", "]"))


def _read_seed_list(spec: str) -> list[str]:
    if spec == "-":
        return [ln.rstrip("\n") for ln in sys.stdin.read().splitlines()]
    return Path(spec).read_text(encoding="utf-8").splitlines()


def _collect_seed_files(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    issues: list[str] = []
    paths: list[Path] = []

    def add_path_str(s: str, origin: str) -> None:
        t = s.strip()
        if not t or t.startswith("#"):
            return
        p = Path(t)
        if not p.is_file():
            if p.exists() and p.is_dir():
                issues.append(f"{origin}: is a directory: {t}")
            else:
                issues.append(f"{origin}: missing file: {t}")
            return
        paths.append(p)

    def add_glob(pat: str, origin: str) -> None:
        matches = sorted(glob.glob(pat))
        if not matches:
            issues.append(f"{origin}: pattern matched no files: {pat}")
            return
        for m in matches:
            add_path_str(m, origin)

    if args.seed_list:
        for line in _read_seed_list(args.seed_list):
            add_path_str(line, f"--seed-list({args.seed_list})")
    elif args.seed_path:
        for sp in args.seed_path:
            if _has_glob_chars(sp):
                add_glob(sp, f"--seed-path({sp})")
            else:
                add_path_str(sp, f"--seed-path({sp})")
    elif args.seed_glob:
        add_glob(args.seed_glob, f"--seed-glob({args.seed_glob})")
    else:
        issues.append("no input provided (use --seed-list / --seed-path / --seed-glob)")

    uniq: dict[str, Path] = {}
    for p in paths:
        k = str(Path(p))
        if k not in uniq:
            uniq[k] = p
    out = [uniq[k] for k in sorted(uniq.keys())]
    return out, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed-list",
        default=None,
        help=(
            "File of paths (or '-' for stdin). Highest precedence. "
            "Relative paths are interpreted from current working directory."
        ),
    )
    ap.add_argument(
        "--seed-path",
        action="append",
        default=[],
        help=(
            "Path or glob. Repeatable. Second precedence. "
            "Relative paths are interpreted from current working directory."
        ),
    )
    ap.add_argument(
        "--seed-glob",
        default=None,
        help="Glob pattern. Lowest precedence.",
    )
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any issue is found.")
    ap.add_argument(
        "--hash-mode",
        choices=["semantic", "raw", "both"],
        default="semantic",
        help=(
            "semantic: compare float32 payload hash to recorded direction_sha256; "
            "raw: compare exact .npy bytes between canonical and alias; "
            "both: do both checks."
        ),
    )
    ap.add_argument(
        "--require-alias",
        action="store_true",
        help="Treat missing alias direction file as an issue.",
    )
    args = ap.parse_args()

    checked = 0
    ok_canon = 0
    ok_alias = 0
    missing_alias = 0
    mismatch_alias = 0
    issues: list[str] = []
    seed_files, input_issues = _collect_seed_files(args)
    issues.extend(input_issues)
    if not seed_files:
        print(
            " ".join(
                [
                    "files=0",
                    "records_checked=0",
                    "ok_canon=0",
                    "ok_alias=0",
                    "missing_alias=0",
                    "mismatch_alias=0",
                    f"issues={len(issues)}",
                ]
            )
        )
        if issues:
            print("issues:")
            for x in issues:
                print(f"- {x}")
        if issues and args.strict:
            return 1
        return 0
    files = [str(p) for p in seed_files]

    for fp in files:
        summary_path = Path(fp)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for rec in summary.get("results", []):
            checked += 1
            ctrl = str(rec.get("control", "unknown"))
            record_id = f"{summary_path}:{ctrl}"

            fs = rec.get("final_summary") or {}
            expected = rec.get("direction_sha256") or fs.get("direction_sha256")
            canon_hermetic = rec.get("direction_path_hermetic") or fs.get("direction_path_hermetic")
            canon_fallback = rec.get("direction_path") or fs.get("direction_path")
            alias_hermetic = rec.get("direction_path_control_hermetic") or fs.get("direction_path_control_hermetic")
            alias_fallback = rec.get("direction_path_control") or fs.get("direction_path_control")

            canonical = _resolve_path(canon_hermetic, summary_path)
            canon_source = "hermetic"
            if canonical is None:
                canonical = _resolve_path(canon_fallback, summary_path)
                canon_source = "fallback"

            alias = _resolve_path(alias_hermetic, summary_path)
            alias_source = "hermetic"
            if alias is None:
                alias = _resolve_path(alias_fallback, summary_path)
                alias_source = "fallback"

            if canonical is None:
                issues.append(
                    f"{record_id}: missing direction file "
                    f"(tried hermetic={canon_hermetic!r}, fallback={canon_fallback!r})"
                )
                continue

            canonical_semantic: str | None = None
            canonical_raw: str | None = None

            if args.hash_mode in {"semantic", "both"}:
                if expected is None:
                    issues.append(f"{record_id}: missing direction_sha256")
                    continue
                try:
                    canonical_semantic = _sha256_float32_npy(canonical)
                except Exception as e:
                    issues.append(f"{record_id}: failed semantic hash {canonical}: {e}")
                    continue
                if canonical_semantic != str(expected):
                    issues.append(
                        f"{record_id}: canonical semantic sha mismatch source={canon_source} file={canonical} "
                        f"expected={expected} observed={canonical_semantic}"
                    )
                    continue
                ok_canon += 1
            else:
                # Raw-only mode still requires canonical raw hash to be computable.
                try:
                    canonical_raw = _sha256_raw_file(canonical)
                except Exception as e:
                    issues.append(f"{record_id}: failed raw hash {canonical}: {e}")
                    continue
                ok_canon += 1

            if args.hash_mode in {"both"} and canonical_raw is None:
                try:
                    canonical_raw = _sha256_raw_file(canonical)
                except Exception as e:
                    issues.append(f"{record_id}: failed raw hash {canonical}: {e}")
                    continue

            if alias is None:
                missing_alias += 1
                if args.require_alias:
                    issues.append(f"{record_id}: missing alias direction file")
                continue

            alias_semantic: str | None = None
            alias_raw: str | None = None

            if args.hash_mode in {"semantic", "both"}:
                try:
                    alias_semantic = _sha256_float32_npy(alias)
                except Exception as e:
                    issues.append(f"{record_id}: failed alias semantic hash {alias}: {e}")
                    mismatch_alias += 1
                    continue
                if canonical_semantic is None:
                    try:
                        canonical_semantic = _sha256_float32_npy(canonical)
                    except Exception as e:
                        issues.append(f"{record_id}: failed canonical semantic hash {canonical}: {e}")
                        mismatch_alias += 1
                        continue
                if alias_semantic != canonical_semantic:
                    issues.append(
                        f"{record_id}: alias semantic mismatch source={alias_source} file={alias} "
                        f"canonical={canonical_semantic} alias={alias_semantic}"
                    )
                    mismatch_alias += 1
                    continue

            if args.hash_mode in {"raw", "both"}:
                try:
                    alias_raw = _sha256_raw_file(alias)
                except Exception as e:
                    issues.append(f"{record_id}: failed alias raw hash {alias}: {e}")
                    mismatch_alias += 1
                    continue
                if canonical_raw is None:
                    try:
                        canonical_raw = _sha256_raw_file(canonical)
                    except Exception as e:
                        issues.append(f"{record_id}: failed canonical raw hash {canonical}: {e}")
                        mismatch_alias += 1
                        continue
                if alias_raw != canonical_raw:
                    issues.append(
                        f"{record_id}: alias raw mismatch source={alias_source} file={alias} "
                        f"canonical={canonical_raw} alias={alias_raw}"
                    )
                    mismatch_alias += 1
                    continue

            ok_alias += 1

    print(
        " ".join(
            [
                f"files={len(files)}",
                f"records_checked={checked}",
                f"ok_canon={ok_canon}",
                f"ok_alias={ok_alias}",
                f"missing_alias={missing_alias}",
                f"mismatch_alias={mismatch_alias}",
                f"issues={len(issues)}",
            ]
        )
    )
    if issues:
        print("issues:")
        for x in issues:
            print(f"- {x}")
    if issues and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
