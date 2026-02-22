#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


RUN_RE = re.compile(r"b6_jobs56_pronouns_k(?P<k>\d+)_pool(?P<pool>\d+)_")
RUN_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")
ROW_ID_SKIP_KEYS = {
    "pll_bias",
    "pll_bias_raw",
    "pll_bias_adj",
    "embed_bias_adj",
    "embed_bias_raw",
    "embed_bias",
    "embed_dir_norm",
    "embed_swap_err_adj",
    "p_value",
    "elapsed_s",
    "seed",
    "rho",
    "p_perm",
}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _centered_rank_vec(values: np.ndarray) -> tuple[np.ndarray, float] | tuple[None, None]:
    if values.shape[0] < 3:
        return None, None
    rv = _ranks(values)
    rv = rv - rv.mean()
    nrm = float(np.linalg.norm(rv))
    if nrm <= 0:
        return None, None
    return rv, nrm


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    rx, nx = _centered_rank_vec(x)
    ry, ny = _centered_rank_vec(y)
    if rx is None or ry is None or nx is None or ny is None:
        return None
    denom = float(nx * ny)
    if denom <= 0:
        return None
    return float(rx.dot(ry) / denom)


def _spearman_prepared_x(rx: np.ndarray, nx: float, y: np.ndarray) -> float | None:
    ry, ny = _centered_rank_vec(y)
    if ry is None or ny is None:
        return None
    denom = float(nx * ny)
    if denom <= 0:
        return None
    return float(rx.dot(ry) / denom)


def _perm_p_two_sided(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float | None:
    rx, nx = _centered_rank_vec(x)
    ry, ny = _centered_rank_vec(y)
    if rx is None or ry is None or nx is None or ny is None:
        return None
    denom = float(nx * ny)
    if denom <= 0:
        return None
    obs = float(rx.dot(ry) / denom)
    rng = np.random.RandomState(seed)
    obs_abs = abs(obs)
    ge = 0
    n = ry.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        stat = float(rx.dot(ry[perm]) / denom)
        if abs(stat) >= obs_abs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def _fisher_combined_p(pvals: list[float]) -> float | None:
    if not pvals:
        return None
    clean = [min(max(float(p), 1e-300), 1.0) for p in pvals if p is not None and math.isfinite(p)]
    if not clean:
        return None
    m = len(clean)
    stat = -2.0 * sum(math.log(p) for p in clean)
    # For df = 2m, chi-square survival has a closed form:
    # sf(x; 2m) = exp(-x/2) * sum_{i=0}^{m-1} (x/2)^i / i!
    z = stat / 2.0
    term = 1.0
    acc = 1.0
    for i in range(1, m):
        term *= z / i
        acc += term
    return float(math.exp(-z) * acc)


def _variant_label(variant_key: str) -> str:
    try:
        v = json.loads(variant_key)
    except Exception:
        return variant_key
    layer = v.get("layer", "unknown")
    pool = v.get("pooling", "unknown")
    norm = "norm1" if bool(v.get("normalize", False)) else "norm0"
    if v.get("whiten_k"):
        return f"{layer}+whiten{v['whiten_k']}[{pool},{norm}]"
    if v.get("center"):
        return f"{layer}+center[{pool},{norm}]"
    return f"{layer}[{pool},{norm}]"


def _run_seed(base_seed: int, run_name: str, variant_key: str) -> int:
    token = f"{run_name}|{variant_key}".encode("utf-8")
    return int((base_seed ^ zlib.crc32(token)) & 0xFFFFFFFF)


def _run_name_to_unix_local(run_name: str) -> int | None:
    m = RUN_TS_RE.search(run_name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    # Treat run-name timestamps as local wall clock.
    local_tz = datetime.now().astimezone().tzinfo
    dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=local_tz)
    return int(dt.timestamp())


def _run_epoch(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime)
    except FileNotFoundError:
        return _run_name_to_unix_local(path.name)


def _stable_hash8_bytes(payload: bytes) -> str:
    return hashlib.blake2s(payload, digest_size=8).hexdigest()


def _stable_hash8_text(text: str) -> str:
    return _stable_hash8_bytes(text.encode("utf-8"))


def _load_group_pairs_from_telemetry(run_dir: Path, template_name: str) -> list[tuple[str, str]]:
    telemetry_path = run_dir / "telemetry.jsonl"
    if not telemetry_path.exists():
        return []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "config":
            continue
        cfg = rec.get("config", {})
        tpls = cfg.get("data", {}).get("pll_templates", {})
        pairs: list[tuple[str, str]] = []
        if isinstance(tpls, list):
            for tpl in tpls:
                if isinstance(tpl, dict) and tpl.get("name") == template_name:
                    for p in tpl.get("group_pairs", []) or []:
                        if isinstance(p, (list, tuple)) and len(p) == 2:
                            pairs.append((str(p[0]), str(p[1])))
        elif isinstance(tpls, dict):
            tpl = tpls.get(template_name) or {}
            for p in tpl.get("group_pairs", []) or []:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    pairs.append((str(p[0]), str(p[1])))
        return pairs
    return []


def _run_has_complete_marker(run_dir: Path) -> bool:
    telemetry_path = run_dir / "telemetry.jsonl"
    if not telemetry_path.exists():
        return False
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") == "run_complete" and rec.get("ok") is True:
            return True
    return False


def _holm_adjust(pvals: list[float | None]) -> list[float | None]:
    out: list[float | None] = [None] * len(pvals)
    items = [(i, p) for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    if not items:
        return out
    m = len(items)
    items_sorted = sorted(items, key=lambda t: float(t[1]))
    running_max = 0.0
    adj = [0.0] * m
    for rank, (_, p) in enumerate(items_sorted):
        val = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, val)
        adj[rank] = running_max
    for rank, (idx, _) in enumerate(items_sorted):
        out[idx] = adj[rank]
    return out


def _pool_perm_mean_rho(
    pool_pairs: list[tuple[np.ndarray, np.ndarray]],
    n_perm: int,
    seed: int,
) -> tuple[float | None, float | None, bool]:
    if not pool_pairs:
        return None, None, False
    prepared = []
    obs_vals = []
    for x, y in pool_pairs:
        rx, nx = _centered_rank_vec(x)
        ry, ny = _centered_rank_vec(y)
        if rx is None or ry is None or nx is None or ny is None:
            continue
        denom = float(nx * ny)
        if denom <= 0:
            continue
        obs = float(rx.dot(ry) / denom)
        obs_vals.append(obs)
        prepared.append((rx, ry, denom, ry.shape[0]))
    if not prepared:
        return None, None, False
    obs_mean = float(np.mean(obs_vals))
    rng = np.random.RandomState(seed)
    ge = 0
    obs_abs = abs(obs_mean)
    for _ in range(n_perm):
        stats = []
        for rx, ry, denom, n in prepared:
            perm = rng.permutation(n)
            stats.append(float(rx.dot(ry[perm]) / denom))
        mean_stat = float(np.mean(stats))
        if abs(mean_stat) >= obs_abs:
            ge += 1
    p = float((ge + 1) / (n_perm + 1))
    return obs_mean, p, ge == 0


def _pool_perm_union_mean_rho(
    pool_maps: list[dict[str, tuple[float, float] | tuple[float, float, tuple[str, str, str, str, str]]]],
    n_perm: int,
    seed: int,
) -> tuple[float | None, float | None, bool]:
    if not pool_maps:
        return None, None, False
    union_keys = sorted(set().union(*[set(pm.keys()) for pm in pool_maps if pm]))
    if not union_keys:
        return None, None, False
    key_to_idx = {rid: i for i, rid in enumerate(union_keys)}
    n_union = len(union_keys)

    prepared = []
    obs_vals = []
    for pm in pool_maps:
        if not pm:
            continue
        pool_keys = sorted(pm.keys())
        if len(pool_keys) < 3:
            continue
        idxs = np.array([key_to_idx[rid] for rid in pool_keys], dtype=int)
        x = np.array([float(pm[rid][0]) for rid in pool_keys], dtype=float)
        y = np.array([float(pm[rid][1]) for rid in pool_keys], dtype=float)
        rx, nx = _centered_rank_vec(x)
        ry, ny = _centered_rank_vec(y)
        if rx is None or ry is None or nx is None or ny is None:
            continue
        denom = float(nx * ny)
        if denom <= 0:
            continue
        obs = float(rx.dot(ry) / denom)
        obs_vals.append(obs)
        prepared.append((rx, ry, denom, idxs))

    if not prepared:
        return None, None, False

    obs_mean = float(np.mean(obs_vals))
    rng = np.random.RandomState(seed)
    ge = 0
    obs_abs = abs(obs_mean)
    # Coupled permutations: one random total order on union IDs -> induced per-pool permutation.
    for _ in range(n_perm):
        u = rng.permutation(n_union)
        stats = []
        for rx, ry, denom, idxs in prepared:
            perm = np.argsort(u[idxs])
            stat = float(rx.dot(ry[perm]) / denom)
            stats.append(stat)
        mean_stat = float(np.mean(stats))
        if abs(mean_stat) >= obs_abs:
            ge += 1

    p = float((ge + 1) / (n_perm + 1))
    return obs_mean, p, ge == 0


def _fisher_z(rho: float) -> float:
    r = float(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12))
    return float(np.arctanh(r))


def _prepare_blocked_pool_entries(
    pool_maps: list[dict[str, tuple[float, float, tuple[str, str, str, str, str]]]],
    block_min_n: int,
) -> tuple[
    list[list[tuple[int, np.ndarray, np.ndarray, float, np.ndarray, int]]],
    list[int],
    list[float],
    list[int],
    list[float],
    list[float],
]:
    per_pool_blocks: list[dict[tuple[str, str, str, str, str], list[tuple[str, float, float]]]] = []
    all_blocks: set[tuple[str, str, str, str, str]] = set()
    for pm in pool_maps:
        bm: dict[tuple[str, str, str, str, str], list[tuple[str, float, float]]] = defaultdict(list)
        for rid, (x, y, bk) in pm.items():
            bm[bk].append((rid, float(x), float(y)))
            all_blocks.add(bk)
        per_pool_blocks.append(bm)

    prepared_by_pool: list[list[tuple[int, np.ndarray, np.ndarray, float, np.ndarray, int]]] = [
        [] for _ in pool_maps
    ]
    block_union_sizes: list[int] = []
    obs_block_rhos: list[float] = []
    obs_block_ns: list[int] = []
    obs_block_rhos_by_block: dict[int, list[float]] = defaultdict(list)
    obs_block_ns_by_block: dict[int, list[int]] = defaultdict(list)

    for bk in sorted(all_blocks):
        union_ids = sorted(set().union(*[set(rid for rid, _, _ in bm.get(bk, [])) for bm in per_pool_blocks]))
        if len(union_ids) < block_min_n:
            continue
        key_to_idx = {rid: i for i, rid in enumerate(union_ids)}
        block_i = len(block_union_sizes)
        block_union_sizes.append(len(union_ids))

        for pi, bm in enumerate(per_pool_blocks):
            items = bm.get(bk, [])
            if len(items) < block_min_n:
                continue
            items = sorted(items, key=lambda t: t[0])
            rids = [rid for rid, _, _ in items]
            idxs = np.array([key_to_idx[rid] for rid in rids], dtype=int)
            x = np.array([x for _, x, _ in items], dtype=float)
            y = np.array([y for _, _, y in items], dtype=float)

            rx, nx = _centered_rank_vec(x)
            ry, ny = _centered_rank_vec(y)
            if rx is None or ry is None or nx is None or ny is None:
                continue
            denom = float(nx * ny)
            if denom <= 0:
                continue

            obs_block_rho = float(rx.dot(ry) / denom)
            obs_block_rhos.append(obs_block_rho)
            obs_block_ns.append(len(items))
            obs_block_rhos_by_block[block_i].append(obs_block_rho)
            obs_block_ns_by_block[block_i].append(len(items))
            prepared_by_pool[pi].append((block_i, rx, ry, denom, idxs, len(items)))

    prepared_by_pool = [p for p in prepared_by_pool if p]
    unique_block_rhos: list[float] = []
    unique_block_ns: list[float] = []
    for block_i in range(len(block_union_sizes)):
        rhos = obs_block_rhos_by_block.get(block_i, [])
        ns = obs_block_ns_by_block.get(block_i, [])
        if not rhos:
            continue
        unique_block_rhos.append(float(np.mean(np.array(rhos, dtype=float))))
        unique_block_ns.append(float(np.mean(np.array(ns, dtype=float))) if ns else float(block_union_sizes[block_i]))
    return prepared_by_pool, block_union_sizes, obs_block_rhos, obs_block_ns, unique_block_rhos, unique_block_ns


def _pool_block_stat_fisher_z(
    entries: list[tuple[int, np.ndarray, np.ndarray, float, np.ndarray, int]],
    u_list: list[np.ndarray],
    block_counts: np.ndarray | None = None,
) -> float | None:
    num = 0.0
    den = 0.0
    for block_i, rx, ry, denom, idxs, n in entries:
        mult = int(block_counts[block_i]) if block_counts is not None else 1
        if mult <= 0:
            continue
        perm = np.argsort(u_list[block_i][idxs])
        rho = float(rx.dot(ry[perm]) / denom)
        # Fisher-z aggregation with per-block weight n-3 (clamped for tiny blocks).
        w = float(mult * max(n - 3, 1))
        num += w * _fisher_z(rho)
        den += w
    if den <= 0:
        return None
    return float(np.tanh(num / den))


def _pool_perm_union_blocked_mean_rho(
    pool_maps: list[dict[str, tuple[float, float, tuple[str, str, str, str, str]]]],
    n_perm: int,
    seed: int,
    block_min_n: int = 3,
    blocked_bootstrap_samples: int = 0,
    blocked_bootstrap_seed: int | None = None,
) -> tuple[float | None, float | None, bool, dict]:
    if not pool_maps:
        return None, None, False, {}

    (
        prepared_by_pool,
        block_union_sizes,
        obs_block_rhos,
        obs_block_ns,
        unique_block_rhos,
        _unique_block_ns,
    ) = _prepare_blocked_pool_entries(pool_maps, block_min_n)
    if not prepared_by_pool:
        return None, None, False, {}

    obs_vals: list[float] = []
    u_id = [np.arange(n, dtype=int) for n in block_union_sizes]
    for entries in prepared_by_pool:
        s = _pool_block_stat_fisher_z(entries, u_id)
        if s is not None and math.isfinite(s):
            obs_vals.append(s)
    if not obs_vals:
        return None, None, False, {}
    obs_mean = float(np.mean(obs_vals))
    obs_abs = abs(obs_mean)

    meta = {
        "blocked_blocks_used": int(len(unique_block_rhos)),
        "blocked_rows_used": int(sum(block_union_sizes)),
        "blocked_median_block_n": (float(np.median(np.array(block_union_sizes, dtype=float))) if block_union_sizes else None),
        "blocked_poolblock_entries_used": int(len(obs_block_rhos)),
        "blocked_poolblock_rows_used": int(sum(obs_block_ns)),
        "blocked_sign_frac_pos": (
            float(np.mean(np.array(unique_block_rhos, dtype=float) > 0.0)) if unique_block_rhos else None
        ),
        "blocked_sign_frac_neg": (
            float(np.mean(np.array(unique_block_rhos, dtype=float) < 0.0)) if unique_block_rhos else None
        ),
        "blocked_abs_rho_gt_0p3_frac": (
            float(np.mean(np.abs(np.array(unique_block_rhos, dtype=float)) > 0.3)) if unique_block_rhos else None
        ),
        "blocked_rho_p10": (
            float(np.percentile(np.array(unique_block_rhos, dtype=float), 10.0)) if unique_block_rhos else None
        ),
        "blocked_rho_p50": (
            float(np.percentile(np.array(unique_block_rhos, dtype=float), 50.0)) if unique_block_rhos else None
        ),
        "blocked_rho_p90": (
            float(np.percentile(np.array(unique_block_rhos, dtype=float), 90.0)) if unique_block_rhos else None
        ),
        "blocked_bootstrap_samples": int(max(0, blocked_bootstrap_samples)),
        "blocked_bootstrap_n_valid": 0,
        "blocked_bootstrap_seed_used": None,
        "rho_block_boot_ci95": None,
    }

    if blocked_bootstrap_samples > 0 and block_union_sizes:
        seed_used = blocked_bootstrap_seed if blocked_bootstrap_seed is not None else ((seed ^ 0xA5A5A5A5) & 0xFFFFFFFF)
        meta["blocked_bootstrap_seed_used"] = int(seed_used)
        boot_rng = np.random.RandomState(seed_used)
        n_blocks = len(block_union_sizes)
        boot_stats: list[float] = []
        for _ in range(blocked_bootstrap_samples):
            sample = boot_rng.randint(0, n_blocks, size=n_blocks)
            counts = np.bincount(sample, minlength=n_blocks).astype(int, copy=False)
            pool_stats: list[float] = []
            for entries in prepared_by_pool:
                s = _pool_block_stat_fisher_z(entries, u_id, block_counts=counts)
                if s is not None and math.isfinite(s):
                    pool_stats.append(s)
            if pool_stats:
                boot_stats.append(float(np.mean(pool_stats)))
        if boot_stats:
            lo, hi = np.percentile(np.array(boot_stats, dtype=float), [2.5, 97.5])
            meta["rho_block_boot_ci95"] = [float(lo), float(hi)]
        meta["blocked_bootstrap_n_valid"] = int(len(boot_stats))

    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(n_perm):
        u_list = [rng.permutation(n) for n in block_union_sizes]
        stats = []
        for entries in prepared_by_pool:
            s = _pool_block_stat_fisher_z(entries, u_list)
            if s is not None and math.isfinite(s):
                stats.append(s)
        if not stats:
            continue
        mean_stat = float(np.mean(stats))
        if abs(mean_stat) >= obs_abs:
            ge += 1

    p = float((ge + 1) / (n_perm + 1))
    return obs_mean, p, ge == 0, meta


def _block_key(rec: dict) -> tuple[str, str, str, str, str]:
    return (
        str(rec.get("pll_template") or ""),
        str(rec.get("pll_template_str") or ""),
        str(rec.get("pll_job") or ""),
        str(rec.get("pll_article") or ""),
        str(rec.get("pll_baseline_template") or ""),
    )


def _record_base_id(rec: dict) -> str:
    rid = rec.get("pair_id") or rec.get("row_id")
    if rid is None:
        stable_payload = {k: rec.get(k) for k in sorted(rec.keys()) if k not in ROW_ID_SKIP_KEYS}
        rid = _stable_hash8_text(json.dumps(stable_payload, sort_keys=True, separators=(",", ":")))
    return str(rid)


def _record_row_ids(rec: dict, group_pairs: list[tuple[str, str]]) -> list[str]:
    rid_str = _record_base_id(rec)
    if group_pairs:
        return [f"{rid_str}|{a_name}|{b_name}" for a_name, b_name in group_pairs]
    return [rid_str]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/b6_jobs56_pronouns_k*_pool*_*")
    ap.add_argument("--filter-template", default="gender_names_jobs")
    ap.add_argument("--perm-samples", type=int, default=5000)
    ap.add_argument("--perm-seed", type=int, default=123)
    ap.add_argument("--pool-perm-samples", type=int, default=5000)
    ap.add_argument("--pool-perm-null", choices=["induced", "blocked"], default="induced")
    ap.add_argument("--block-min-n", type=int, default=3)
    ap.add_argument("--blocked-bootstrap-samples", type=int, default=0)
    ap.add_argument("--blocked-bootstrap-seed", type=int, default=None)
    ap.add_argument("--overlap-threshold", type=float, default=1.01)
    ap.add_argument("--require-pair-rows", dest="require_pair_rows", action="store_true", default=True)
    ap.add_argument("--allow-legacy-rows", dest="require_pair_rows", action="store_false")
    ap.add_argument("--min-row-frac", type=float, default=0.9)
    ap.add_argument("--require-full-variants", dest="require_full_variants", action="store_true", default=True)
    ap.add_argument("--allow-partial-variants", dest="require_full_variants", action="store_false")
    ap.add_argument("--require-complete", dest="require_complete", action="store_true", default=True)
    ap.add_argument("--allow-incomplete", dest="require_complete", action="store_false")
    ap.add_argument("--out-prefix", default="runs/b6_aggregate_none")
    ap.add_argument("--start-ts", type=int, default=None, help="Optional unix timestamp to include only runs with mtime >= start-ts.")
    args = ap.parse_args()

    run_dirs = sorted(Path(".").glob(args.runs_glob))
    if args.start_ts is not None:
        filtered = []
        for p in run_dirs:
            run_epoch = _run_epoch(p)
            if run_epoch is not None and run_epoch >= args.start_ts:
                filtered.append(p)
        run_dirs = filtered
    long_rows: list[dict] = []
    run_meta_rows: list[dict] = []
    kv_pool_pairs: dict[tuple[int, str], list[dict]] = {}

    for run_dir in run_dirs:
        m = RUN_RE.search(run_dir.name)
        if not m:
            continue
        k = int(m.group("k"))
        pool = int(m.group("pool"))
        run_complete = _run_has_complete_marker(run_dir)
        if args.require_complete and not run_complete:
            continue
        aligned_path = run_dir / "results_embed_aligned.jsonl"
        aligned_pairs_path = run_dir / "results_embed_aligned_pairs.jsonl"
        if aligned_pairs_path.exists():
            source_path = aligned_pairs_path
            has_pair_rows = True
        elif aligned_path.exists() and not args.require_pair_rows:
            source_path = aligned_path
            has_pair_rows = False
        else:
            continue
        group_pairs = _load_group_pairs_from_telemetry(run_dir, args.filter_template) if not has_pair_rows else []

        by_variant: dict[str, list[tuple[float, float]]] = {}
        by_variant_map: dict[str, dict[str, tuple[float, float, tuple[str, str, str, str, str]]]] = {}
        job_ids: set[str] = set()
        base_row_ids: set[str] = set()
        row_ids: set[str] = set()
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pll_template") != args.filter_template:
                continue
            if has_pair_rows:
                rec_base_id = str(rec.get("row_id") or _record_base_id(rec))
                rec_row_ids = [rec_base_id]
            else:
                rec_base_id = _record_base_id(rec)
                rec_row_ids = _record_row_ids(rec, group_pairs)
            base_row_ids.add(rec_base_id)
            for rid in rec_row_ids:
                row_ids.add(rid)
            jid = f"{rec.get('pll_job')}|{rec.get('pll_article')}"
            job_ids.add(jid)
            if rec.get("pll_bias") is None or rec.get("embed_bias_adj") is None:
                continue
            x_val = float(rec["pll_bias"])
            y_val = float(rec["embed_bias_adj"])
            block_key = _block_key(rec)
            vkey = json.dumps(rec.get("variant"), sort_keys=True, separators=(",", ":"))
            by_variant.setdefault(vkey, []).append((x_val, y_val))
            vmap = by_variant_map.setdefault(vkey, {})
            map_ids = rec_row_ids if not has_pair_rows else [rec_base_id]
            for map_id in map_ids:
                if map_id not in vmap:
                    vmap[map_id] = (x_val, y_val, block_key)

        run_meta_rows.append(
            {
                "run": run_dir.name,
                "k": k,
                "pool": pool,
                "run_ts": _run_epoch(run_dir),
                "run_complete": run_complete,
                "aligned_source": source_path.name,
                "variant_count": len(by_variant),
                "variant_keys": sorted(by_variant.keys()),
                "job_count": len(job_ids),
                "job_hash": _stable_hash8_text("\n".join(sorted(job_ids))),
                "base_row_count": len(base_row_ids),
                "base_row_hash": _stable_hash8_text("\n".join(sorted(base_row_ids))),
                "row_count": len(row_ids),
                "row_hash": _stable_hash8_text("\n".join(sorted(row_ids))),
                "job_ids": sorted(job_ids),
                "base_row_ids": sorted(base_row_ids),
                "row_ids": sorted(row_ids),
            }
        )

        for vkey, pairs in by_variant.items():
            x = np.array([p[0] for p in pairs], dtype=float)
            y = np.array([p[1] for p in pairs], dtype=float)
            rho = _spearman(x, y)
            p_perm = _perm_p_two_sided(
                x,
                y,
                args.perm_samples,
                _run_seed(args.perm_seed, run_dir.name, vkey),
            )
            p_perm_floor = 1.0 / (args.perm_samples + 1.0)
            long_rows.append(
                {
                    "run": run_dir.name,
                    "k": k,
                    "pool": pool,
                    "variant_key": vkey,
                    "variant_label": _variant_label(vkey),
                    "n": int(x.shape[0]),
                    "rho": rho,
                    "p_perm": p_perm,
                    "p_perm_is_floor": bool(p_perm is not None and p_perm <= (p_perm_floor + 1e-15)),
                    "filter_template": args.filter_template,
                    "perm_samples": args.perm_samples,
                }
            )
            kv_pool_pairs.setdefault((k, vkey), []).append(
                {
                    "run": run_dir.name,
                    "pool": pool,
                    "x": x,
                    "y": y,
                    "rid_map": by_variant_map.get(vkey, {}),
                }
            )

    # Drop incomplete runs: low row coverage or missing variants for a given k.
    run_keep: dict[str, bool] = {}
    by_k_row_counts: dict[int, list[int]] = {}
    by_k_variant_counts: dict[int, list[int]] = {}
    for meta in run_meta_rows:
        by_k_row_counts.setdefault(int(meta["k"]), []).append(int(meta.get("row_count", 0)))
        by_k_variant_counts.setdefault(int(meta["k"]), []).append(int(meta.get("variant_count", 0)))
    k_row_median = {k: float(np.median(v)) for k, v in by_k_row_counts.items() if v}
    k_variant_max = {k: max(v) for k, v in by_k_variant_counts.items() if v}

    for meta in run_meta_rows:
        k = int(meta["k"])
        row_count = int(meta.get("row_count", 0))
        variant_count = int(meta.get("variant_count", 0))
        median_row = k_row_median.get(k, 0.0)
        max_variants = int(k_variant_max.get(k, variant_count))
        keep_row = True if median_row <= 0 else (row_count >= args.min_row_frac * median_row)
        keep_variant = (not args.require_full_variants) or (variant_count >= max_variants)
        keep = bool(keep_row and keep_variant)
        meta["row_median_k"] = median_row
        meta["max_variant_count_k"] = max_variants
        meta["keep"] = keep
        run_keep[str(meta["run"])] = keep

    run_meta_rows = [m for m in run_meta_rows if bool(m.get("keep", False))]
    long_rows = [r for r in long_rows if run_keep.get(str(r.get("run")), False)]
    for kv in list(kv_pool_pairs.keys()):
        kept = [rec for rec in kv_pool_pairs[kv] if run_keep.get(str(rec.get("run")), False)]
        if kept:
            kv_pool_pairs[kv] = kept
        else:
            del kv_pool_pairs[kv]

    # Aggregate by (k, variant)
    by_kv: dict[tuple[int, str], list[dict]] = {}
    for row in long_rows:
        by_kv.setdefault((row["k"], row["variant_key"]), []).append(row)

    # Pool overlap diagnostics per k
    by_k_pool_base_rows: dict[int, dict[int, set[str]]] = {}
    by_k_pool_expanded_rows: dict[int, dict[int, set[str]]] = {}
    for meta in run_meta_rows:
        by_k_pool_base_rows.setdefault(meta["k"], {})[meta["pool"]] = set(meta["base_row_ids"])
        by_k_pool_expanded_rows.setdefault(meta["k"], {})[meta["pool"]] = set(meta["row_ids"])

    def _mean_jaccard(pool_map: dict[int, set[str]]) -> float | None:
        pools = sorted(pool_map.keys())
        jac = []
        for i in range(len(pools)):
            for j in range(i + 1, len(pools)):
                a = pool_map[pools[i]]
                b = pool_map[pools[j]]
                den = len(a | b)
                if den == 0:
                    continue
                jac.append(len(a & b) / den)
        return float(np.mean(jac)) if jac else None

    k_overlap_base: dict[int, float | None] = {}
    k_overlap_expanded: dict[int, float | None] = {}
    for k, pool_map in by_k_pool_base_rows.items():
        k_overlap_base[k] = _mean_jaccard(pool_map)
    for k, pool_map in by_k_pool_expanded_rows.items():
        k_overlap_expanded[k] = _mean_jaccard(pool_map)

    k_overlap: dict[int, float | None] = {}
    for k in set(k_overlap_base.keys()) | set(k_overlap_expanded.keys()):
        # Backward-compatible alias defaults to expanded overlap.
        k_overlap[k] = k_overlap_expanded.get(k)

    summary_rows: list[dict] = []
    for (k, vkey), rows in sorted(by_kv.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rhos = [r["rho"] for r in rows if r["rho"] is not None and math.isfinite(r["rho"])]
        pvals = [r["p_perm"] for r in rows if r["p_perm"] is not None and math.isfinite(r["p_perm"])]
        if rhos:
            rho_mean = float(np.mean(rhos))
            rho_std = float(np.std(rhos))
            rho_min = float(np.min(rhos))
            rho_max = float(np.max(rhos))
            sign = 0 if rho_mean == 0 else (1 if rho_mean > 0 else -1)
            sign_consistency = float(np.mean([1.0 if (0 if r == 0 else (1 if r > 0 else -1)) == sign else 0.0 for r in rhos]))
        else:
            rho_mean = rho_std = rho_min = rho_max = sign_consistency = None
        summary_rows.append(
            {
                "k": k,
                "variant_key": vkey,
                "variant_label": _variant_label(vkey),
                "n_pools": len(rows),
                "n_min": min(r["n"] for r in rows) if rows else None,
                "n_max": max(r["n"] for r in rows) if rows else None,
                "rho_mean": rho_mean,
                "rho_std": rho_std,
                "rho_min": rho_min,
                "rho_max": rho_max,
                "p_fisher": _fisher_combined_p(pvals),
                "p_fisher_note": "heuristic_if_overlap_high",
                "rho_poolperm_mean": None,
                "p_poolperm": None,
                "p_poolperm_is_floor": None,
                "pool_perm_mode": None,
                "pool_perm_n_used": None,
                "p_holm_k": None,
                "sign_consistency": sign_consistency,
                "pool_overlap_base_jaccard_mean": k_overlap_base.get(k),
                "pool_overlap_expanded_jaccard_mean": k_overlap_expanded.get(k),
                "pool_overlap_jaccard_mean": k_overlap.get(k),
                "blocked_blocks_used": None,
                "blocked_rows_used": None,
                "blocked_median_block_n": None,
                "blocked_sign_frac_pos": None,
                "blocked_sign_frac_neg": None,
                "blocked_abs_rho_gt_0p3_frac": None,
                "blocked_rho_p10": None,
                "blocked_rho_p50": None,
                "blocked_rho_p90": None,
                "blocked_bootstrap_samples": None,
                "rho_block_boot_ci95": None,
            }
        )

    # Pooled permutation p-value per (k, variant)
    summary_idx = {(int(r["k"]), r["variant_key"]): i for i, r in enumerate(summary_rows)}
    for key, pools in kv_pool_pairs.items():
        if key not in summary_idx:
            continue
        idx = summary_idx[key]
        overlap = summary_rows[idx].get("pool_overlap_expanded_jaccard_mean")
        if overlap is not None and overlap > args.overlap_threshold and len(pools) > 1:
            pools_used = [sorted(pools, key=lambda r: int(r["pool"]))[0]]
            pool_mode = f"single_pool_overlap>{args.overlap_threshold:g}"
        else:
            pools_used = pools
            pool_mode = "all_pools"
        if len(pools_used) > 1:
            if args.pool_perm_null == "blocked":
                pooled_rho, pooled_p, pooled_floor, blocked_meta = _pool_perm_union_blocked_mean_rho(
                    [rec.get("rid_map", {}) for rec in pools_used],
                    args.pool_perm_samples,
                    _run_seed(args.perm_seed, f"k{key[0]}", key[1]),
                    block_min_n=args.block_min_n,
                    blocked_bootstrap_samples=args.blocked_bootstrap_samples,
                    blocked_bootstrap_seed=(
                        args.blocked_bootstrap_seed
                        if args.blocked_bootstrap_seed is not None
                        else _run_seed(args.perm_seed, f"k{key[0]}|blocked_boot", key[1])
                    ),
                )
                pool_mode = f"{pool_mode}+union_perm_blocked"
            else:
                pooled_rho, pooled_p, pooled_floor = _pool_perm_union_mean_rho(
                    [rec.get("rid_map", {}) for rec in pools_used],
                    args.pool_perm_samples,
                    _run_seed(args.perm_seed, f"k{key[0]}", key[1]),
                )
                blocked_meta = {}
                pool_mode = f"{pool_mode}+union_perm"
            if pooled_rho is None and pooled_p is None:
                pooled_rho, pooled_p, pooled_floor = _pool_perm_mean_rho(
                    [(rec["x"], rec["y"]) for rec in pools_used],
                    args.pool_perm_samples,
                    _run_seed(args.perm_seed, f"k{key[0]}", key[1]),
                )
                blocked_meta = {}
                pool_mode = f"{pool_mode}+fallback_independent"
        else:
            pooled_rho, pooled_p, pooled_floor = _pool_perm_mean_rho(
                [(rec["x"], rec["y"]) for rec in pools_used],
                args.pool_perm_samples,
                _run_seed(args.perm_seed, f"k{key[0]}", key[1]),
            )
            blocked_meta = {}
        summary_rows[idx]["rho_poolperm_mean"] = pooled_rho
        summary_rows[idx]["p_poolperm"] = pooled_p
        summary_rows[idx]["p_poolperm_is_floor"] = pooled_floor
        summary_rows[idx]["pool_perm_mode"] = pool_mode
        summary_rows[idx]["pool_perm_n_used"] = len(pools_used)
        for k_meta, v_meta in blocked_meta.items():
            summary_rows[idx][k_meta] = v_meta

    # Holm correction within each k across variants, on pooled permutation p
    by_k_idx: dict[int, list[int]] = {}
    for i, row in enumerate(summary_rows):
        by_k_idx.setdefault(int(row["k"]), []).append(i)
    for k, idxs in by_k_idx.items():
        pvals = [summary_rows[i].get("p_poolperm") for i in idxs]
        padj = _holm_adjust(pvals)
        for i, adj in zip(idxs, padj):
            summary_rows[i]["p_holm_k"] = adj

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")

    out_json.write_text(
        json.dumps(
            {
                "filter_template": args.filter_template,
                "perm_samples": args.perm_samples,
                "perm_seed": args.perm_seed,
                "pool_perm_samples": args.pool_perm_samples,
                "pool_perm_null": args.pool_perm_null,
                "block_min_n": args.block_min_n,
                "blocked_bootstrap_samples": args.blocked_bootstrap_samples,
                "blocked_bootstrap_seed": args.blocked_bootstrap_seed,
                "overlap_threshold": args.overlap_threshold,
                "require_pair_rows": args.require_pair_rows,
                "min_row_frac": args.min_row_frac,
                "require_full_variants": args.require_full_variants,
                "require_complete": args.require_complete,
                "runs_glob": args.runs_glob,
                "start_ts": args.start_ts,
                "runs_count": len(run_dirs),
                "runs_kept_count": len(run_meta_rows),
                "rows": long_rows,
                "run_meta": run_meta_rows,
                "pool_overlap_base_by_k": k_overlap_base,
                "pool_overlap_expanded_by_k": k_overlap_expanded,
                "pool_overlap_by_k": k_overlap,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "k",
                "pool",
                "variant_key",
                "variant_label",
                "n",
                "rho",
                "p_perm",
                "p_perm_is_floor",
                "filter_template",
                "perm_samples",
            ],
        )
        w.writeheader()
        for row in long_rows:
            w.writerow(row)

    md_lines = [
        "# B6 Aggregate",
        "",
        f"- filter_template: `{args.filter_template}`",
        f"- perm_samples: `{args.perm_samples}`",
        f"- pool_perm_samples: `{args.pool_perm_samples}`",
        f"- pool_perm_null: `{args.pool_perm_null}`",
        f"- block_min_n: `{args.block_min_n}`",
        f"- blocked_bootstrap_samples: `{args.blocked_bootstrap_samples}`",
        f"- blocked_bootstrap_seed: `{args.blocked_bootstrap_seed}`",
        f"- overlap_threshold: `{args.overlap_threshold}`",
        f"- require_pair_rows: `{args.require_pair_rows}`",
        f"- min_row_frac: `{args.min_row_frac}`",
        f"- require_full_variants: `{args.require_full_variants}`",
        f"- require_complete: `{args.require_complete}`",
        f"- runs_glob: `{args.runs_glob}`",
        f"- start_ts: `{args.start_ts}`",
        f"- runs_kept: `{len(run_meta_rows)}` of `{len(run_dirs)}`",
        "",
        "| k | variant | n_pools | rho_mean±std | rho_range | p_poolperm | holm_p(k) | fisher_p(heur) | sign_consistency | overlap_base | overlap_expanded | block_sign(+/-) | block_rho p50 [p10,p90] | block_|rho|>0.3 | block_boot_ci95 | pool_perm_mode |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        rho_ms = (
            f"{row['rho_mean']:+.4f}±{row['rho_std']:.4f}"
            if row["rho_mean"] is not None and row["rho_std"] is not None
            else "NA"
        )
        rho_range = (
            f"[{row['rho_min']:+.4f}, {row['rho_max']:+.4f}]"
            if row["rho_min"] is not None and row["rho_max"] is not None
            else "NA"
        )
        if row["p_poolperm"] is None:
            ppool = "NA"
        elif bool(row.get("p_poolperm_is_floor")):
            ppool = f"<=1/({args.pool_perm_samples}+1)"
        else:
            ppool = f"{row['p_poolperm']:.6g}"
        fisher = f"{row['p_fisher']:.6g}" if row["p_fisher"] is not None else "NA"
        pholm = f"{row['p_holm_k']:.6g}" if row["p_holm_k"] is not None else "NA"
        sign_cons = f"{row['sign_consistency']:.3f}" if row["sign_consistency"] is not None else "NA"
        overlap_base = (
            f"{row['pool_overlap_base_jaccard_mean']:.3f}"
            if row["pool_overlap_base_jaccard_mean"] is not None
            else "NA"
        )
        overlap_expanded = (
            f"{row['pool_overlap_expanded_jaccard_mean']:.3f}"
            if row["pool_overlap_expanded_jaccard_mean"] is not None
            else "NA"
        )
        block_sign = (
            f"{row['blocked_sign_frac_pos']:.3f}/{row['blocked_sign_frac_neg']:.3f}"
            if row.get("blocked_sign_frac_pos") is not None and row.get("blocked_sign_frac_neg") is not None
            else "NA"
        )
        block_rho_dist = (
            f"{row['blocked_rho_p50']:+.4f} [{row['blocked_rho_p10']:+.4f},{row['blocked_rho_p90']:+.4f}]"
            if row.get("blocked_rho_p50") is not None
            and row.get("blocked_rho_p10") is not None
            and row.get("blocked_rho_p90") is not None
            else "NA"
        )
        block_spike = (
            f"{row['blocked_abs_rho_gt_0p3_frac']:.3f}" if row.get("blocked_abs_rho_gt_0p3_frac") is not None else "NA"
        )
        block_ci = (
            f"[{row['rho_block_boot_ci95'][0]:+.4f},{row['rho_block_boot_ci95'][1]:+.4f}]"
            if isinstance(row.get("rho_block_boot_ci95"), list) and len(row.get("rho_block_boot_ci95")) == 2
            else "NA"
        )
        pool_mode = row.get("pool_perm_mode") or "NA"
        md_lines.append(
            f"| {row['k']} | `{row['variant_label']}` | {row['n_pools']} | {rho_ms} | {rho_range} | {ppool} | {pholm} | {fisher} | {sign_cons} | {overlap_base} | {overlap_expanded} | {block_sign} | {block_rho_dist} | {block_spike} | {block_ci} | {pool_mode} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("wrote", out_json)
    print("wrote", out_csv)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
