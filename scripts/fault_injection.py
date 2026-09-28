#!/usr/bin/env python3
"""Fault-injection evaluation of the direction provenance gates.

Copies a clean matched-cost sweep (seed summaries + every direction file they
reference) into a sandbox, pins a ledger on the clean state, then applies one
fault at a time and records which gate detects it:

- integrity: check_direction_integrity.py --hash-mode both --require-alias --strict
- ci_regen:  make_artifacts_manifest.py then check_artifacts_manifest.py on the
             freshly generated manifest (what ci_gates.sh did originally)
- pinned:    check_artifacts_manifest.py against the ledger pinned before the fault
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
PATH_KEYS = ("direction_path", "direction_path_control", "direction_path_hermetic", "direction_path_control_hermetic")


def _rec_paths(rec: dict) -> dict[str, str]:
    fs = rec.get("final_summary") or {}
    out = {}
    for k in PATH_KEYS:
        v = rec.get(k) or fs.get(k)
        if v:
            out[k] = str(v)
    return out


def _unit(seed: int, dim: int) -> np.ndarray:
    v = np.random.RandomState(seed).normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _sem_sha(v: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float32).reshape(-1).tobytes()).hexdigest()


class Sandbox:
    def __init__(self, root: Path, summaries: list[str]):
        self.root = root
        self.summaries = summaries

    def p(self, rel: str) -> Path:
        return self.root / rel

    def load(self, i: int = 0) -> dict:
        return json.loads(self.p(self.summaries[i]).read_text(encoding="utf-8"))

    def save(self, obj: dict, i: int = 0) -> None:
        self.p(self.summaries[i]).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def rec_paths(self, i: int = 0, r: int = 0) -> dict[str, str]:
        return _rec_paths(self.load(i)["results"][r])


def _build_sandbox(src_glob: str, dst: Path) -> Sandbox:
    summaries = sorted(glob.glob(str(REPO / src_glob)))
    if not summaries:
        raise SystemExit(f"no summaries matched {src_glob}")
    rels = []
    for s in summaries:
        rel = str(Path(s).relative_to(REPO))
        rels.append(rel)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst / rel)
        for rec in json.loads(Path(s).read_text(encoding="utf-8")).get("results", []):
            for fp in _rec_paths(rec).values():
                if (REPO / fp).is_file():
                    (dst / fp).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(REPO / fp, dst / fp)
    return Sandbox(dst, rels)


def _run(args: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _gate_integrity(sb: Sandbox, seed_glob: str, extra: list[str]) -> tuple[bool, str]:
    rc, out = _run(
        [str(REPO / "scripts/check_direction_integrity.py"), "--seed-glob", seed_glob,
         "--hash-mode", "both", "--require-alias", "--strict", *extra],
        sb.root,
    )
    return rc != 0, out.splitlines()[0] if out else ""


def _gate_manifest(sb: Sandbox, manifest: Path) -> tuple[bool, str]:
    rc, out = _run([str(REPO / "scripts/check_artifacts_manifest.py"), "--manifest", str(manifest), "--strict"], sb.root)
    return rc != 0, out.splitlines()[0] if out else ""


def _make_manifest(sb: Sandbox, seed_glob: str, out: Path) -> None:
    rc, txt = _run([str(REPO / "scripts/make_artifacts_manifest.py"), "--seed-glob", seed_glob, "--out", str(out)], sb.root)
    if rc != 0:
        raise RuntimeError(f"manifest generation failed:\n{txt}")


# ---------------------------------------------------------------- faults ----
# Each fault mutates the sandbox in place. Docstring = one-line description.


def f_clean(sb: Sandbox) -> None:
    """No fault (false-positive check)."""


def f_regenerated_vector(sb: Sandbox) -> None:
    """Direction silently regenerated (different RNG/code); both hermetic copies overwritten consistently."""
    rp = sb.rec_paths()
    v = _unit(999, np.load(sb.p(rp["direction_path_hermetic"])).size)
    for k in ("direction_path_hermetic", "direction_path_control_hermetic"):
        np.save(sb.p(rp[k]), v)


def f_bitflip(sb: Sandbox) -> None:
    """Single bit flipped in the canonical payload (storage corruption)."""
    p = sb.p(sb.rec_paths()["direction_path_hermetic"])
    b = bytearray(p.read_bytes())
    b[-1] ^= 0x01
    p.write_bytes(bytes(b))


def f_truncated(sb: Sandbox) -> None:
    """Canonical .npy truncated (partial write)."""
    p = sb.p(sb.rec_paths()["direction_path_hermetic"])
    p.write_bytes(p.read_bytes()[:100])


def f_swapped_controls(sb: Sandbox) -> None:
    """Direction files of two controls swapped (wrong file under the right name)."""
    a, b = sb.rec_paths(0, 0), sb.rec_paths(0, 1)
    for k in ("direction_path_hermetic", "direction_path_control_hermetic"):
        pa, pb = sb.p(a[k]), sb.p(b[k])
        tmp = pa.read_bytes()
        pa.write_bytes(pb.read_bytes())
        pb.write_bytes(tmp)


def f_alias_diverged(sb: Sandbox) -> None:
    """Control alias overwritten with a different vector; canonical intact."""
    rp = sb.rec_paths()
    np.save(sb.p(rp["direction_path_control_hermetic"]), _unit(999, np.load(sb.p(rp["direction_path_hermetic"])).size))


def f_hermetic_deleted(sb: Sandbox) -> None:
    """Hermetic copies deleted; only the original (non-hermetic) run-dir files remain."""
    rp = sb.rec_paths()
    for k in ("direction_path_hermetic", "direction_path_control_hermetic"):
        sb.p(rp[k]).unlink()


def f_all_copies_deleted(sb: Sandbox) -> None:
    """Every copy of the direction deleted."""
    for fp in sb.rec_paths().values():
        if sb.p(fp).exists():
            sb.p(fp).unlink()


def f_joint_regeneration(sb: Sandbox) -> None:
    """Direction regenerated AND summary hash rewritten to match (whole run silently redone)."""
    s = sb.load()
    rec = s["results"][0]
    rp = _rec_paths(rec)
    v = _unit(999, np.load(sb.p(rp["direction_path_hermetic"])).size)
    for fp in rp.values():
        if sb.p(fp).exists():
            np.save(sb.p(fp), v)
    rec["direction_sha256"] = _sem_sha(v)
    if rec.get("final_summary"):
        rec["final_summary"]["direction_sha256"] = _sem_sha(v)
    sb.save(s)


def f_non_canonical(sb: Sandbox) -> None:
    """Non-unit-norm vector stored with a self-consistent hash (violates canonicalization contract)."""
    s = sb.load()
    rec = s["results"][0]
    rp = _rec_paths(rec)
    v = (np.load(sb.p(rp["direction_path_hermetic"])) * 3.0).astype(np.float32)
    for fp in rp.values():
        if sb.p(fp).exists():
            np.save(sb.p(fp), v)
    rec["direction_sha256"] = _sem_sha(v)
    if rec.get("final_summary"):
        rec["final_summary"]["direction_sha256"] = _sem_sha(v)
    sb.save(s)


def f_sha_conflict(sb: Sandbox) -> None:
    """Record-level and final_summary-level direction_sha256 disagree."""
    s = sb.load()
    s["results"][0]["final_summary"]["direction_sha256"] = "0" * 64
    sb.save(s)


def f_record_dropped(sb: Sandbox) -> None:
    """One control record removed from a seed summary."""
    s = sb.load()
    s["results"] = s["results"][1:]
    sb.save(s)


def f_empty_results(sb: Sandbox) -> None:
    """Seed summary with an empty results list."""
    s = sb.load()
    s["results"] = []
    sb.save(s)


def f_summary_deleted(sb: Sandbox) -> None:
    """A whole seed summary file missing (glob silently matches fewer files)."""
    sb.p(sb.summaries[-1]).unlink()


FAULTS: list[Callable[[Sandbox], None]] = [
    f_clean,
    f_regenerated_vector,
    f_bitflip,
    f_truncated,
    f_swapped_controls,
    f_alias_diverged,
    f_hermetic_deleted,
    f_all_copies_deleted,
    f_joint_regeneration,
    f_non_canonical,
    f_sha_conflict,
    f_record_dropped,
    f_empty_results,
    f_summary_deleted,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", default="runs/mc_orth_vs_supportavoid_mc4_newartifact/seed*.matched_cost_summary.json")
    ap.add_argument("--integrity-extra", default="", help="Extra args for check_direction_integrity.py (space-separated).")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    extra = args.integrity_extra.split() if args.integrity_extra else []

    rows = []
    for fault in FAULTS:
        with tempfile.TemporaryDirectory(prefix="fi_") as td:
            sb = _build_sandbox(args.seed_glob, Path(td))
            pinned = Path(td) / "pinned_ledger.json"
            _make_manifest(sb, args.seed_glob, pinned)
            fault(sb)
            integ = _gate_integrity(sb, args.seed_glob, extra)
            regen = Path(td) / "regen_ledger.json"
            try:
                _make_manifest(sb, args.seed_glob, regen)
                ci = _gate_manifest(sb, regen)
            except RuntimeError as e:
                ci = (True, f"manifest generation failed: {str(e).splitlines()[-1]}")
            pin = _gate_manifest(sb, pinned)
            rows.append({
                "fault": fault.__name__[2:],
                "description": (fault.__doc__ or "").strip(),
                "integrity": integ[0], "integrity_out": integ[1],
                "ci_regen": ci[0], "ci_regen_out": ci[1],
                "pinned": pin[0], "pinned_out": pin[1],
            })

    def mark(x: bool, clean: bool) -> str:
        if clean:
            return "false alarm" if x else "pass"
        return "caught" if x else "MISSED"

    lines = ["| fault | description | integrity gate | CI (regenerated manifest) | pinned ledger |", "|---|---|---|---|---|"]
    for r in rows:
        c = r["fault"] == "clean"
        lines.append(f"| `{r['fault']}` | {r['description']} | {mark(r['integrity'], c)} | {mark(r['ci_regen'], c)} | {mark(r['pinned'], c)} |")
    md = "\n".join(lines)
    print(md)
    if args.out_md:
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
