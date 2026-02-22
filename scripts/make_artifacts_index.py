#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def _sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_rel(path: Path, cwd: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(cwd))
        except Exception:
            return str(p)
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-glob", default="runs/_artifacts*.json")
    ap.add_argument("--out", default="runs/artifacts_index.json")
    args = ap.parse_args()

    cwd = Path.cwd()
    files = sorted({str(Path(p)) for p in glob.glob(args.manifest_glob)})
    if not files:
        raise SystemExit(f"no manifest files matched: {args.manifest_glob}")

    manifests: list[dict[str, object]] = []
    for fp in files:
        p = Path(fp)
        if not p.is_file():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        entry: dict[str, object] = {
            "path": _to_rel(p, cwd),
            "sha256_raw": _sha256_raw(p),
        }
        if "generated_utc" in j:
            entry["generated_utc"] = j.get("generated_utc")
        if "hostname" in j:
            entry["hostname"] = j.get("hostname")
        git = j.get("git")
        if isinstance(git, dict):
            keep_git: dict[str, object] = {}
            if "rev" in git:
                keep_git["rev"] = git.get("rev")
            if "dirty" in git:
                keep_git["dirty"] = git.get("dirty")
            if keep_git:
                entry["git"] = keep_git
        manifests.append(entry)

    out: dict[str, object] = {
        "schema_version": 1,
        "manifests": manifests,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"manifests={len(manifests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
