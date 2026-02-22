#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_from_cwd(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (Path.cwd() / p)


def _approx_eq(a, b, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return a == b
    return abs(af - bf) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.is_file():
        raise SystemExit(f"missing manifest: {mpath}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    issues: list[str] = []
    checked_summaries = 0
    checked_records = 0
    checked_files = 0

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        issues.append("manifest.runs missing or not a list")
        runs = []

    for run in runs:
        for s in run.get("seed_summaries", []):
            spath = str(s.get("path", ""))
            p = _resolve_from_cwd(spath)
            if not p.is_file():
                issues.append(f"missing summary file: {spath}")
                continue
            checked_files += 1
            expected_raw = s.get("sha256_raw")
            observed_raw = _sha256_raw(p)
            if expected_raw != observed_raw:
                issues.append(f"summary sha mismatch: {spath}")
                continue
            checked_summaries += 1

            summary = json.loads(p.read_text(encoding="utf-8"))
            by_control = {str(r.get("control")): r for r in summary.get("results", [])}

            for rec in s.get("records", []):
                ctrl = str(rec.get("control"))
                r = by_control.get(ctrl)
                if r is None:
                    issues.append(f"{spath}:{ctrl}: missing control in summary")
                    continue
                checked_records += 1
                fs = r.get("final_summary") or {}
                got_sem = r.get("direction_sha256") or fs.get("direction_sha256")
                exp_sem = rec.get("direction_sha256_semantic")
                if got_sem != exp_sem:
                    issues.append(f"{spath}:{ctrl}: semantic direction sha mismatch")

                dph = rec.get("direction_path_hermetic")
                if dph:
                    pd = _resolve_from_cwd(str(dph))
                    if not pd.is_file():
                        issues.append(f"{spath}:{ctrl}: missing hermetic direction file: {dph}")
                    else:
                        exp = rec.get("sha256_raw_direction_npy")
                        obs = _sha256_raw(pd)
                        if exp != obs:
                            issues.append(f"{spath}:{ctrl}: raw hash mismatch for direction_path_hermetic")

                dpc = rec.get("direction_path_control_hermetic")
                if dpc:
                    pdc = _resolve_from_cwd(str(dpc))
                    if not pdc.is_file():
                        issues.append(f"{spath}:{ctrl}: missing hermetic control direction file: {dpc}")
                    else:
                        exp = rec.get("sha256_raw_direction_control_npy")
                        obs = _sha256_raw(pdc)
                        if exp != obs:
                            issues.append(f"{spath}:{ctrl}: raw hash mismatch for direction_path_control_hermetic")

                ss = rec.get("support_stats") or {}
                ds = (fs.get("direction_shape_stats") or {})
                for k in ("masked_support_mass", "support_intrusion_eps", "support_intrusion_frac_eps"):
                    if k in ss and not _approx_eq(ss.get(k), ds.get(k)):
                        issues.append(f"{spath}:{ctrl}: support stat mismatch for {k}")

        for key in ("figures", "tables"):
            for entry in run.get(key, []):
                epath = str(entry.get("path", ""))
                p = _resolve_from_cwd(epath)
                if not p.is_file():
                    issues.append(f"missing {key[:-1]} file: {epath}")
                    continue
                checked_files += 1
                exp = entry.get("sha256_raw")
                obs = _sha256_raw(p)
                if exp != obs:
                    issues.append(f"{key[:-1]} sha mismatch: {epath}")

    extras = manifest.get("extras") or {}
    if extras and not isinstance(extras, dict):
        issues.append("manifest.extras is not an object")
        extras = {}

    for key in ("figures", "tables"):
        for entry in extras.get(key, []):
            epath = str(entry.get("path", ""))
            p = _resolve_from_cwd(epath)
            if not p.is_file():
                issues.append(f"missing extra {key[:-1]} file: {epath}")
                continue
            checked_files += 1
            exp = entry.get("sha256_raw")
            obs = _sha256_raw(p)
            if exp != obs:
                issues.append(f"extra {key[:-1]} sha mismatch: {epath}")

    print(
        " ".join(
            [
                f"runs={len(runs)}",
                f"checked_summaries={checked_summaries}",
                f"checked_records={checked_records}",
                f"checked_files={checked_files}",
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
