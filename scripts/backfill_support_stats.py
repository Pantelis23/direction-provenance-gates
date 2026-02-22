#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_run_causal_module():
    mod_path = Path(__file__).resolve().parent / "run_causal.py"
    spec = importlib.util.spec_from_file_location("run_causal_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_seed_from_filename(path: Path) -> int:
    name = path.name
    # expected: seed{N}.matched_cost_summary.json
    if "seed" not in name:
        raise ValueError(f"cannot parse seed from filename: {name}")
    tail = name.split("seed", 1)[1]
    seed_str = tail.split(".", 1)[0]
    return int(seed_str)


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    return x / (float(np.linalg.norm(x)) + 1e-12)


def _load_exact_direction(fs: dict, summary_path: Path) -> np.ndarray | None:
    # Prefer exact persisted direction from run_causal summary.
    cand_paths: list[str] = []
    for key in ("direction_path", "direction_path_control"):
        p = fs.get(key)
        if p:
            cand_paths.append(str(p))
    for s in cand_paths:
        p = Path(s)
        if not p.is_absolute():
            # Try workspace-relative first.
            if p.exists():
                return np.load(p).astype(np.float32)
            # Fallback: resolve relative to summary file location.
            p2 = (summary_path.parent / p).resolve()
            if p2.exists():
                return np.load(p2).astype(np.float32)
        elif p.exists():
            return np.load(p).astype(np.float32)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-glob", required=True)
    ap.add_argument("--d-path", required=True, help="source control_dir.npy")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--support-control", default="orthogonal_random_support_avoid_top512")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = _load_run_causal_module()
    src = _unit(np.load(args.d_path).astype(np.float32))
    files = sorted(glob.glob(args.seed_glob))
    if not files:
        raise SystemExit(f"no files matched: {args.seed_glob}")

    patched = 0
    checked = 0
    for fp in files:
        path = Path(fp)
        summary = json.loads(path.read_text(encoding="utf-8"))
        seed = _parse_seed_from_filename(path)
        changed = False
        checked += 1

        for rec in summary.get("results", []):
            ctrl = str(rec.get("control"))
            fs = (rec.get("final_summary") or {})
            ds = (fs.get("direction_shape_stats") or {})
            if (
                "masked_support_mass" in ds
                and "support_overlap_frac" in ds
                and "support_intrusion_eps" in ds
                and "support_intrusion_frac_eps" in ds
            ):
                continue

            v = _load_exact_direction(fs, path)
            if v is None:
                rng = np.random.RandomState(seed)
                if ctrl == str(args.support_control):
                    v = rc._make_orthogonal_support_avoid_direction(src, rng, topk=args.k)
                elif ctrl == "orthogonal_random":
                    v = rc._make_orthogonal_random_direction(src, rng)
                else:
                    continue

            v = _unit(v)
            ds.update(rc._support_avoid_stats(v, src, k=args.k))
            fs["direction_shape_stats"] = ds
            rec["final_summary"] = fs
            changed = True

        if changed:
            patched += 1
            if not args.dry_run:
                path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"checked_files={checked}")
    print(f"patched_files={patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
