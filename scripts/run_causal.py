#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.pll import PLLIntervention, PLLScorer
from src.telemetry.logger import TelemetryLogger
from src.utils.config import load_config
from src.utils.seed import set_seed


def _stable_hash8_text(text: str) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=8).hexdigest()


def _runtime_meta() -> dict[str, str]:
    return {
        "utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hostname": socket.gethostname(),
    }


def _article(job_entry: Any) -> tuple[str, str]:
    if isinstance(job_entry, dict):
        job = job_entry.get("job")
        article = job_entry.get("article")
    else:
        job = str(job_entry)
        article = None
    if not job:
        raise ValueError("job entry missing job name")
    if article is None:
        article = "an" if str(job)[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return str(job), str(article)


def _a_an(article: str) -> str:
    return "n" if article.lower().startswith("an") else ""


def _fill(tpl_str: str, name: str, job: str | None = None, article: str | None = None) -> str:
    job_val = job or ""
    article_val = article or ("an" if job_val[:1].lower() in {"a", "e", "i", "o", "u"} else "a")
    return tpl_str.format(
        name=name,
        group=name,
        subj=name,
        job=job_val,
        article=article_val,
        a_an=_a_an(article_val),
    )


def _resolve_groups(tpl: dict) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    pairs = tpl.get("group_pairs", [])
    if pairs:
        out = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("group_pairs must contain [A, B] pairs")
            out.append((str(pair[0]), str(pair[1])))
        return [a for a, _ in out], [b for _, b in out], out
    group_a = tpl.get("groupA", [])
    group_b = tpl.get("groupB", [])
    if isinstance(group_a, str):
        group_a = [group_a]
    if isinstance(group_b, str):
        group_b = [group_b]
    if len(group_a) != len(group_b):
        raise ValueError("groupA/groupB must have the same length for paired intervention.")
    pairs = list(zip([str(v) for v in group_a], [str(v) for v in group_b]))
    return [a for a, _ in pairs], [b for _, b in pairs], pairs


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    for rank, idx in enumerate(order, start=1):
        r[idx] = float(rank)
    return r


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    den = da * db
    return num / den if den > 0 else 0.0


def _spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def _make_orthogonal_random_direction(source: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        u = rng.normal(size=src.shape).astype(np.float32)
        return u / (float(np.linalg.norm(u)) + 1e-12)
    d = src / (src_norm + 1e-12)
    for _ in range(6):
        u = rng.normal(size=src.shape).astype(np.float32)
        u = u - float(np.dot(u, d)) * d
        un = float(np.linalg.norm(u))
        if un > 1e-8:
            return (u / un).astype(np.float32)
    # Extremely rare fallback: deterministic basis vector orthogonalization.
    basis = np.zeros_like(src, dtype=np.float32)
    basis[0] = 1.0
    if abs(float(np.dot(basis, d))) > 0.9 and src.size > 1:
        basis[0] = 0.0
        basis[1] = 1.0
    u = basis - float(np.dot(basis, d)) * d
    un = float(np.linalg.norm(u))
    if un <= 1e-8:
        u = rng.normal(size=src.shape).astype(np.float32)
        un = float(np.linalg.norm(u))
    return (u / (un + 1e-12)).astype(np.float32)


def _make_orthogonal_support_avoid_direction(
    source: np.ndarray,
    rng: np.random.RandomState,
    topk: int = 512,
) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        src_unit = src / (src_norm + 1e-12)
        return _make_orthogonal_random_direction(src_unit, rng)
    d = src / (src_norm + 1e-12)
    n = int(d.size)
    k = int(min(max(int(topk), 1), n - 1))
    if k <= 0 or k >= n:
        return _make_orthogonal_random_direction(d, rng)

    mask = np.zeros(n, dtype=bool)
    idx = np.argpartition(np.abs(d), n - k)[-k:]
    mask[idx] = True

    # Sample in complement of source top-k support.
    u = rng.normal(size=d.shape).astype(np.float32)
    u[mask] = 0.0

    # Enforce orthogonality in the same subspace.
    d2 = d.copy()
    d2[mask] = 0.0
    denom = float(np.dot(d2, d2))
    if denom <= 1e-12:
        return _make_orthogonal_random_direction(d, rng)

    u = u - (float(np.dot(u, d2)) / denom) * d2
    u[mask] = 0.0

    un = float(np.linalg.norm(u))
    if un <= 1e-8:
        return _make_orthogonal_random_direction(d, rng)
    return (u / (un + 1e-12)).astype(np.float32)


def _orthogonalize_to_source(vec: np.ndarray, source_unit: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    d = np.asarray(source_unit, dtype=np.float32)
    v = v - float(np.dot(v, d)) * d
    vn = float(np.linalg.norm(v))
    if vn <= 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / vn).astype(np.float32)


def _rank_match_values(u: np.ndarray, vals: np.ndarray) -> np.ndarray:
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    vv = np.asarray(vals, dtype=np.float32).reshape(-1)
    if uu.size != vv.size:
        raise ValueError("rank_match_values requires arrays with same length.")
    out = np.empty_like(uu, dtype=np.float32)
    out[np.argsort(uu)] = np.sort(vv)
    return out


def _make_marginal_matched_random_direction(source: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        src_unit = src / (src_norm + 1e-12)
        return _make_orthogonal_random_direction(src_unit, rng)
    d = src / (src_norm + 1e-12)
    # Sample from the empirical signed marginal with replacement (not a permutation).
    v = rng.choice(d, size=d.size, replace=True).astype(np.float32)
    v = _orthogonalize_to_source(v, d)
    if float(np.linalg.norm(v)) <= 1e-8:
        return _make_orthogonal_random_direction(d, rng)
    return v.astype(np.float32)


def _make_abs_marginal_matched_random_direction(source: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        src_unit = src / (src_norm + 1e-12)
        return _make_orthogonal_random_direction(src_unit, rng)
    d = src / (src_norm + 1e-12)
    u = np.abs(rng.normal(size=src.shape).astype(np.float32))
    mags = np.sort(np.abs(d).astype(np.float32))
    vmag = _rank_match_values(u, mags)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d.size).astype(np.float32)
    v = vmag * signs
    v = _orthogonalize_to_source(v, d)
    if float(np.linalg.norm(v)) <= 1e-8:
        return _make_orthogonal_random_direction(d, rng)
    return v.astype(np.float32)


def _make_abs_marginal_source_signs_direction(source: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        src_unit = src / (src_norm + 1e-12)
        return _make_orthogonal_random_direction(src_unit, rng)
    d = src / (src_norm + 1e-12)
    mags = np.sort(np.abs(d).astype(np.float32))
    u = np.abs(rng.normal(size=src.shape).astype(np.float32))
    vmag = _rank_match_values(u, mags)
    signs = np.sign(d).astype(np.float32)
    signs[signs == 0.0] = 1.0
    signs = signs[rng.permutation(signs.size)]
    v = vmag * signs
    v = _orthogonalize_to_source(v, d)
    if float(np.linalg.norm(v)) <= 1e-8:
        return _make_orthogonal_random_direction(d, rng)
    return v.astype(np.float32)


def _make_orthogonalized_shuffled_direction(source: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    src_norm = float(np.linalg.norm(src))
    if src.ndim != 1 or src.size < 2 or src_norm <= 1e-12:
        src_unit = src / (src_norm + 1e-12)
        return _make_orthogonal_random_direction(src_unit, rng)
    d = src / (src_norm + 1e-12)
    v = rng.permutation(d).astype(np.float32)
    v = _orthogonalize_to_source(v, d)
    if float(np.linalg.norm(v)) <= 1e-8:
        return _make_orthogonal_random_direction(d, rng)
    return v.astype(np.float32)


def _topk_abs_idx(v: np.ndarray, k: int) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32).reshape(-1)
    n = int(vv.size)
    kk = int(min(max(int(k), 1), n))
    return np.argpartition(np.abs(vv), n - kk)[-kk:]


def _support_avoid_stats(vec: np.ndarray, source: np.ndarray, k: int = 512) -> dict[str, float | int]:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    # Scale-aware, float32-safe epsilon to avoid brittle absolute thresholds.
    maxabs = float(np.max(np.abs(v))) if v.size else 0.0
    eps = float(np.finfo(np.float32).eps) * 8.0 * (maxabs + 1e-12)
    if v.size <= 0 or src.size <= 0:
        return {
            "masked_support_mass": 0.0,
            "support_overlap_frac": 0.0,
            "support_overlap_count": 0,
            "support_intrusion_eps": float(eps),
            "support_intrusion_frac_eps": 0.0,
            "support_intrusion_count_eps": 0,
        }
    kk = int(min(max(int(k), 1), v.size, src.size))
    idx_src = _topk_abs_idx(src, kk)
    idx_v = _topk_abs_idx(v, kk)
    denom = float(np.sum(np.abs(v))) + 1e-12
    masked_support_mass = float(np.sum(np.abs(v[idx_src])) / denom)
    intrusion = np.abs(v[idx_src]) > eps
    support_intrusion_count_eps = int(np.sum(intrusion))
    support_intrusion_frac_eps = float(np.mean(intrusion))
    overlap = np.intersect1d(idx_src, idx_v, assume_unique=False)
    support_overlap_count = int(overlap.size)
    support_overlap_frac = float(support_overlap_count) / float(kk)
    return {
        "masked_support_mass": masked_support_mass,
        "support_overlap_frac": support_overlap_frac,
        "support_overlap_count": support_overlap_count,
        "support_intrusion_eps": float(eps),
        "support_intrusion_frac_eps": support_intrusion_frac_eps,
        "support_intrusion_count_eps": support_intrusion_count_eps,
    }


def _direction_shape_stats(vec: np.ndarray, source: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    av = np.abs(v)
    l1 = float(np.sum(av))
    l2 = float(np.linalg.norm(v))
    linf = float(np.max(av)) if av.size > 0 else 0.0
    topk = {}
    denom = l1 + 1e-12
    src_abs = np.abs(src)

    def _topk_overlap(k: int) -> float:
        kk = min(k, av.size, src_abs.size)
        if kk <= 0:
            return 0.0
        idx_v = np.argpartition(av, av.size - kk)[-kk:]
        idx_s = np.argpartition(src_abs, src_abs.size - kk)[-kk:]
        return float(len(set(idx_v.tolist()) & set(idx_s.tolist())) / float(kk))

    for k in (32, 128, 512):
        kk = min(k, av.size)
        if kk <= 0:
            topk[f"top{k}_mass"] = 0.0
            topk[f"top{k}_overlap"] = 0.0
            continue
        vals = np.partition(av, av.size - kk)[-kk:]
        topk[f"top{k}_mass"] = float(np.sum(vals) / denom)
        topk[f"top{k}_overlap"] = _topk_overlap(k)
    mv = float(np.mean(v))
    sv = float(np.std(v))
    if sv <= 1e-12:
        kurt = 0.0
    else:
        z = (v - mv) / sv
        kurt = float(np.mean(z**4) - 3.0)
    sparsity_1e3 = float(np.mean(av < 1e-3)) if av.size > 0 else 0.0
    abs_cos = float(np.dot(np.abs(v), np.abs(src)) / ((float(np.linalg.norm(np.abs(v))) * float(np.linalg.norm(np.abs(src)))) + 1e-12))
    frac_pos = float(np.mean(v > 0.0)) if v.size > 0 else 0.0
    sign_v = np.sign(v).astype(np.float32)
    mag_v = av.astype(np.float32)
    sign_std = float(np.std(sign_v))
    mag_std = float(np.std(mag_v))
    if sign_std <= 1e-12 or mag_std <= 1e-12:
        corr_sign_mag = 0.0
    else:
        corr_sign_mag = float(np.mean((sign_v - float(np.mean(sign_v))) * (mag_v - float(np.mean(mag_v)))) / (sign_std * mag_std))
    out = {
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "kurtosis_excess": kurt,
        "sparsity_lt_1e-3": sparsity_1e3,
        "abs_cos_to_source": abs_cos,
        "frac_pos": frac_pos,
        "corr_sign_mag": corr_sign_mag,
    }
    out.update(_support_avoid_stats(v, src, k=512))
    out.update(topk)
    return out


def _resolve_layer_index(layer_value: Any, n_layers: int) -> int:
    if isinstance(layer_value, int):
        if layer_value < 0:
            layer_value = n_layers + layer_value
        return max(0, min(int(layer_value), n_layers - 1))
    if isinstance(layer_value, str):
        lv = layer_value.lower()
        if lv == "early":
            return min(1, n_layers - 1)
        if lv == "mid":
            return max(1, min(n_layers - 1, n_layers // 2))
        if lv == "late":
            return n_layers - 1
    return n_layers - 1


def _block_bootstrap_ci(values: dict[str, list[float]], n_boot: int, seed: int) -> tuple[float | None, list[float] | None]:
    if not values:
        return None, None
    block_keys = sorted(values.keys())
    block_means = {k: float(np.mean(np.array(values[k], dtype=float))) for k in block_keys if values[k]}
    if not block_means:
        return None, None
    obs = float(np.mean(list(block_means.values())))
    if n_boot <= 0:
        return obs, None
    rng = np.random.RandomState(seed)
    keys = list(block_means.keys())
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        boot.append(float(np.mean([block_means[k] for k in sample])))
    lo, hi = np.percentile(np.array(boot, dtype=float), [2.5, 97.5])
    return obs, [float(lo), float(hi)]


def _load_direction(
    source_run: Path,
    d_path: Path | None,
    variant_key: str | None,
    variant_hash: str | None,
) -> tuple[np.ndarray, dict]:
    if d_path is not None:
        arr = np.load(d_path)
        return np.asarray(arr, dtype=np.float32), {"path": str(d_path), "source": "direct_path"}

    index_path = source_run / "bias_directions.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing bias direction index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not index:
        raise ValueError("bias_directions.json is empty or invalid.")

    picked = None
    if variant_key is not None and variant_key in index:
        picked = index[variant_key]
    elif variant_hash is not None:
        for _k, info in index.items():
            if str(info.get("variant_hash")) == variant_hash:
                picked = info
                break
    elif len(index) == 1:
        picked = next(iter(index.values()))

    if picked is None:
        raise ValueError("Could not resolve direction. Provide --variant-key or --variant-hash.")

    vec_path = source_run / picked["path"]
    if not vec_path.exists():
        raise FileNotFoundError(f"Direction file missing: {vec_path}")
    arr = np.load(vec_path)
    return np.asarray(arr, dtype=np.float32), picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-run", required=True, help="Run directory containing bias_directions.json")
    ap.add_argument("--d-path", default=None, help="Optional direct .npy direction path")
    ap.add_argument("--variant-key", default=None)
    ap.add_argument("--variant-hash", default=None)
    ap.add_argument("--mode", choices=["project_out", "flip", "add"], default="flip")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--alpha-policy", choices=["same", "opposite", "only_a", "only_b"], default="same")
    ap.add_argument("--intervention-scope", choices=["all", "name", "non_name"], default="all")
    ap.add_argument("--layer", default=None, help="Layer index or alias (early/mid/late). Default from variant.")
    ap.add_argument("--filter-template", default="gender_names_jobs")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--eval-split", choices=["test", "train", "all"], default="test")
    ap.add_argument(
        "--direction-control",
        choices=[
            "none",
            "random",
            "shuffled",
            "orthogonal_random",
            "orthogonal_random_support_avoid_top512",
            "marginal_matched_random_orth",
            "signed_resample_marginal_orth",
            "abs_marginal_matched_random_orth",
            "abs_marginal_source_signs_orth",
        ],
        default="none",
    )
    ap.add_argument(
        "--direction-seed",
        type=int,
        default=None,
        help="Seed used for control-direction construction (defaults to control-seed).",
    )
    ap.add_argument("--control-seed", type=int, default=None, help="Seed for random/shuffled control direction.")
    ap.add_argument("--log-concept-score", action="store_true", help="Log concept score shifts in rows/summary.")
    ap.add_argument(
        "--concept-d-path",
        default=None,
        help="Optional concept direction .npy for score logging (defaults to intervention direction).",
    )
    ap.add_argument("--concept-score-pool", choices=["mean", "last", "cls"], default="mean")
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--run-name", default="causal_intervention")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_cfg = cfg.get("run", {})
    set_seed(int(run_cfg.get("seed", 42)))

    output_root = Path(run_cfg.get("output_dir", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.run_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetryLogger(run_dir / "telemetry.jsonl")
    telemetry.log({"type": "run_start", "run_id": run_id})
    telemetry.log({"type": "config", "config": cfg, "args": vars(args)})

    source_run = Path(args.source_run)
    d_arr, d_info = _load_direction(
        source_run=source_run,
        d_path=Path(args.d_path) if args.d_path else None,
        variant_key=args.variant_key,
        variant_hash=args.variant_hash,
    )
    d_arr = np.asarray(d_arr, dtype=np.float32)
    d_norm = float(np.linalg.norm(d_arr))
    if d_norm <= 0:
        raise ValueError("Loaded direction has zero norm.")
    d_hat = d_arr / d_norm
    source_d_hat = np.asarray(d_hat, dtype=np.float32).copy()

    control_seed = int(args.control_seed) if args.control_seed is not None else int(args.holdout_seed)
    direction_seed = int(args.direction_seed) if args.direction_seed is not None else int(control_seed)
    rng = np.random.RandomState(direction_seed)
    if args.direction_control == "random":
        d_hat = rng.normal(size=d_hat.shape).astype(np.float32)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "shuffled":
        d_hat = _make_orthogonalized_shuffled_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "orthogonal_random":
        d_hat = _make_orthogonal_random_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "orthogonal_random_support_avoid_top512":
        d_hat = _make_orthogonal_support_avoid_direction(source_d_hat, rng, topk=512)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "marginal_matched_random_orth":
        d_hat = _make_marginal_matched_random_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "signed_resample_marginal_orth":
        d_hat = _make_marginal_matched_random_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "abs_marginal_matched_random_orth":
        d_hat = _make_abs_marginal_matched_random_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)
    elif args.direction_control == "abs_marginal_source_signs_orth":
        d_hat = _make_abs_marginal_source_signs_direction(source_d_hat, rng)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-12)

    d_hat = np.asarray(d_hat, dtype=np.float32).reshape(-1)
    d_hat = d_hat / (float(np.linalg.norm(d_hat)) + 1e-12)
    direction_cos_to_source = float(
        np.dot(d_hat, source_d_hat) / ((float(np.linalg.norm(d_hat)) * float(np.linalg.norm(source_d_hat))) + 1e-12)
    )
    direction_shape_stats = _direction_shape_stats(d_hat, source_d_hat)

    # Persist canonical direction vector and keep control-specific alias for compatibility.
    direction_path = run_dir / "direction.npy"
    direction_path_control = run_dir / f"control_direction_{args.direction_control}.npy"
    np.save(direction_path, d_hat)
    np.save(direction_path_control, d_hat)
    direction_sha256 = hashlib.sha256(d_hat.tobytes()).hexdigest()

    concept_d_hat = None
    concept_direction_sha256 = None
    if args.log_concept_score:
        if args.concept_d_path:
            concept_arr = np.asarray(np.load(Path(args.concept_d_path)), dtype=np.float32)
        else:
            concept_arr = np.asarray(d_hat, dtype=np.float32)
        c_norm = float(np.linalg.norm(concept_arr))
        if c_norm <= 1e-12:
            raise ValueError("Concept direction has near-zero norm.")
        concept_d_hat = concept_arr / c_norm
        concept_direction_sha256 = hashlib.sha256(np.asarray(concept_d_hat, dtype=np.float32).tobytes()).hexdigest()

    scorer = PLLScorer(cfg.get("model", {}))
    n_layers = len(scorer._resolve_layers())
    layer_cfg = args.layer if args.layer is not None else d_info.get("variant", {}).get("layer", "late")
    if isinstance(layer_cfg, str) and layer_cfg.isdigit():
        layer_cfg = int(layer_cfg)
    layer_idx = _resolve_layer_index(layer_cfg, n_layers)
    if args.log_concept_score:
        # Force hidden-state layout resolution once so summary always records it.
        probe = f"__layout_probe__|seed={args.holdout_seed}|layer={layer_idx}|pool={args.concept_score_pool}"
        _ = scorer.score_sentence(
            probe,
            return_hidden=True,
            hidden_layer=layer_idx,
            hidden_pool=args.concept_score_pool,
        )

    telemetry.log(
        {
            "type": "causal_setup",
            "source_run": str(source_run),
            "direction_info": d_info,
            "direction_control": args.direction_control,
            "control_seed": control_seed,
            "direction_seed": direction_seed,
            "direction_dim": int(d_hat.shape[0]),
            "direction_norm": float(np.linalg.norm(d_hat)),
            "direction_cos_to_source": direction_cos_to_source,
            "direction_sha256": direction_sha256,
            "direction_path": str(direction_path),
            "direction_path_control": str(direction_path_control),
            "direction_shape_stats": direction_shape_stats,
            "log_concept_score": bool(args.log_concept_score),
            "concept_direction_path": str(args.concept_d_path) if args.concept_d_path else None,
            "concept_direction_sha256": concept_direction_sha256,
            "concept_score_pool": args.concept_score_pool,
            "layer_idx": int(layer_idx),
            "mode": args.mode,
            "alpha": float(args.alpha),
            "alpha_policy": args.alpha_policy,
            "intervention_scope": args.intervention_scope,
            "holdout_frac": float(args.holdout_frac),
            "eval_split": args.eval_split,
        }
    )

    pll_templates = cfg.get("data", {}).get("pll_templates", [])
    out_rows = []
    block_delta_adj: dict[str, list[float]] = {}
    block_abs_reduction_adj: dict[str, list[float]] = {}
    logprob_pair_change: list[float] = []
    pll_change_values: list[float] = []
    concept_score_delta_values: list[float] = []
    concept_score_abs_delta_values: list[float] = []
    concept_score_rel_delta_values: list[float] = []
    concept_score_abs_rel_delta_values: list[float] = []
    hidden_layout_logged = False
    name_mask_cache: dict[tuple[str, str], np.ndarray] = {}
    token_mask_warned: set[tuple[str, str, str]] = set()
    token_mask_logged: set[tuple[str, str, str]] = set()

    out_path = run_dir / "results_causal.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for tpl in pll_templates:
            tpl_name = str(tpl.get("name", "unnamed"))
            if tpl_name != args.filter_template:
                continue
            templates = tpl.get("templates", [])
            jobs = tpl.get("jobs", [])
            adjust = bool(tpl.get("adjust_by_baseline", False))
            baseline_templates = tpl.get("baseline_templates", [])
            baseline_template = tpl.get("baseline_template")
            _, _, pairs = _resolve_groups(tpl)
            if not pairs:
                continue

            for t_idx, tpl_str in enumerate(templates):
                base_tpl = None
                if adjust:
                    base_tpl = baseline_templates[t_idx] if baseline_templates else baseline_template
                    if not base_tpl:
                        raise ValueError(f"Missing baseline template for {tpl_name}/{tpl_str}")

                for job_entry in jobs:
                    job, article = _article(job_entry)
                    block_key_obj = {
                        "pll_template": tpl_name,
                        "pll_template_str": tpl_str,
                        "pll_job": job,
                        "pll_article": article,
                        "pll_baseline_template": base_tpl if adjust else None,
                    }
                    block_key = json.dumps(block_key_obj, sort_keys=True, separators=(",", ":"))
                    h = int(hashlib.blake2s((block_key + f"|{args.holdout_seed}").encode("utf-8"), digest_size=8).hexdigest(), 16)
                    is_test = (h % 10000) < int(args.holdout_frac * 10000.0)
                    if args.eval_split == "test" and not is_test:
                        continue
                    if args.eval_split == "train" and is_test:
                        continue

                    for pair_idx, (name_a, name_b) in enumerate(pairs):
                        text_a = _fill(tpl_str, name_a, job=job, article=article)
                        text_b = _fill(tpl_str, name_b, job=job, article=article)
                        base_a_text = _fill(base_tpl, name_a) if adjust else None
                        base_b_text = _fill(base_tpl, name_b) if adjust else None
                        need_hidden = bool(args.log_concept_score and concept_d_hat is not None)

                        a_base = scorer.score_sentence(
                            text_a,
                            return_hidden=need_hidden,
                            hidden_layer=layer_idx if need_hidden else None,
                            hidden_pool=args.concept_score_pool,
                        )
                        b_base = scorer.score_sentence(
                            text_b,
                            return_hidden=need_hidden,
                            hidden_layer=layer_idx if need_hidden else None,
                            hidden_pool=args.concept_score_pool,
                        )

                        alpha_a, alpha_b = float(args.alpha), float(args.alpha)
                        if args.alpha_policy == "opposite":
                            alpha_b = -alpha_b
                        elif args.alpha_policy == "only_a":
                            alpha_b = 0.0
                        elif args.alpha_policy == "only_b":
                            alpha_a = 0.0

                        def _build_intervention(text: str, name: str, alpha_val: float) -> PLLIntervention:
                            iv = PLLIntervention(
                                layer=layer_idx,
                                direction=d_hat,
                                mode=args.mode,
                                alpha=alpha_val,
                            )
                            if args.intervention_scope in {"name", "non_name"}:
                                key = (text, name)
                                name_mask = name_mask_cache.get(key)
                                if name_mask is None:
                                    name_mask = scorer.token_mask_for_substring(text, name)
                                    name_mask_cache[key] = name_mask
                                name_found = bool(np.any(name_mask))
                                if args.intervention_scope == "name":
                                    mask = name_mask
                                else:
                                    # non_name is only valid if name span was found.
                                    # If not found, force no-op to avoid accidental full-sequence edits.
                                    mask = np.logical_not(name_mask) if name_found else np.zeros_like(name_mask)
                                log_key = (text, name, args.intervention_scope)
                                # Avoid editing special tokens (BOS/EOS/PAD/etc.) across any masked scope.
                                try:
                                    ids_t = scorer.tokenizer(text, return_tensors="pt", truncation=True).get("input_ids")
                                    if ids_t is not None:
                                        ids = ids_t.squeeze(0).tolist()
                                        special = set(getattr(scorer.tokenizer, "all_special_ids", []) or [])
                                        if special:
                                            for i, tid in enumerate(ids):
                                                if tid in special and i < mask.shape[0]:
                                                    mask[i] = False
                                except Exception:
                                    pass
                                # Always attach a mask. For name scope, empty name mask becomes a no-op.
                                iv.token_mask = mask
                                if log_key not in token_mask_logged:
                                    name_mask_sum = int(np.sum(name_mask))
                                    name_mask_frac = float(name_mask_sum / max(1, name_mask.shape[0]))
                                    mask_sum = int(np.sum(mask))
                                    mask_frac = float(mask_sum / max(1, mask.shape[0]))
                                    telemetry.log(
                                        {
                                            "type": "name_token_mask",
                                            "scope": args.intervention_scope,
                                            "text": text,
                                            "name": name,
                                            "name_mask_sum": name_mask_sum,
                                            "name_mask_len": int(name_mask.shape[0]),
                                            "name_mask_frac": name_mask_frac,
                                            "mask_sum": mask_sum,
                                            "mask_len": int(mask.shape[0]),
                                            "mask_frac": mask_frac,
                                        }
                                    )
                                    token_mask_logged.add(log_key)
                                if not name_found and log_key not in token_mask_warned:
                                    telemetry.log(
                                        {
                                            "type": "warning",
                                            "what": "empty_name_token_mask",
                                            "scope": args.intervention_scope,
                                            "text": text,
                                            "name": name,
                                        }
                                    )
                                    token_mask_warned.add(log_key)
                            return iv

                        int_a_main = _build_intervention(text_a, name_a, alpha_a)
                        int_b_main = _build_intervention(text_b, name_b, alpha_b)

                        a_int = scorer.score_sentence(
                            text_a,
                            intervention=int_a_main,
                            return_hidden=need_hidden,
                            hidden_layer=layer_idx if need_hidden else None,
                            hidden_pool=args.concept_score_pool,
                        )
                        b_int = scorer.score_sentence(
                            text_b,
                            intervention=int_b_main,
                            return_hidden=need_hidden,
                            hidden_layer=layer_idx if need_hidden else None,
                            hidden_pool=args.concept_score_pool,
                        )
                        pll_change_values.append(float(a_base.avg_logprob - a_int.avg_logprob))
                        pll_change_values.append(float(b_base.avg_logprob - b_int.avg_logprob))

                        pll_bias_base_raw = a_base.avg_logprob - b_base.avg_logprob
                        pll_bias_int_raw = a_int.avg_logprob - b_int.avg_logprob
                        pair_mean_logprob_change = (a_int.avg_logprob + b_int.avg_logprob) - (
                            a_base.avg_logprob + b_base.avg_logprob
                        )

                        if adjust:
                            ba_base = scorer.score_sentence(base_a_text)
                            bb_base = scorer.score_sentence(base_b_text)
                            int_a_base = _build_intervention(base_a_text, name_a, alpha_a)
                            int_b_base = _build_intervention(base_b_text, name_b, alpha_b)
                            ba_int = scorer.score_sentence(base_a_text, intervention=int_a_base)
                            bb_int = scorer.score_sentence(base_b_text, intervention=int_b_base)
                            pll_change_values.append(float(ba_base.avg_logprob - ba_int.avg_logprob))
                            pll_change_values.append(float(bb_base.avg_logprob - bb_int.avg_logprob))
                            pll_bias_base_adj = (a_base.avg_logprob - ba_base.avg_logprob) - (
                                b_base.avg_logprob - bb_base.avg_logprob
                            )
                            pll_bias_int_adj = (a_int.avg_logprob - ba_int.avg_logprob) - (
                                b_int.avg_logprob - bb_int.avg_logprob
                            )
                        else:
                            pll_bias_base_adj = None
                            pll_bias_int_adj = None

                        concept_score_delta_a = None
                        concept_score_delta_b = None
                        concept_score_delta_pair = None
                        concept_score_abs_delta_pair = None
                        if args.log_concept_score and concept_d_hat is not None:
                            assert a_base.pooled_hidden is not None
                            assert a_int.pooled_hidden is not None
                            assert b_base.pooled_hidden is not None
                            assert b_int.pooled_hidden is not None
                            if not hidden_layout_logged:
                                layout_info = getattr(scorer, "_last_hidden_layout", None)
                                if isinstance(layout_info, dict):
                                    telemetry.log({"type": "hidden_layout", **layout_info, "layer_idx": int(layer_idx)})
                                hidden_layout_logged = True
                            va_base = np.asarray(a_base.pooled_hidden, dtype=np.float32)
                            va_int = np.asarray(a_int.pooled_hidden, dtype=np.float32)
                            vb_base = np.asarray(b_base.pooled_hidden, dtype=np.float32)
                            vb_int = np.asarray(b_int.pooled_hidden, dtype=np.float32)
                            sa_base = float(np.dot(va_base, concept_d_hat))
                            sa_int = float(np.dot(va_int, concept_d_hat))
                            sb_base = float(np.dot(vb_base, concept_d_hat))
                            sb_int = float(np.dot(vb_int, concept_d_hat))
                            concept_score_delta_a = sa_int - sa_base
                            concept_score_delta_b = sb_int - sb_base
                            concept_score_delta_pair = 0.5 * (concept_score_delta_a + concept_score_delta_b)
                            concept_score_abs_delta_pair = 0.5 * (
                                abs(concept_score_delta_a) + abs(concept_score_delta_b)
                            )
                            base_pair_norm = 0.5 * (
                                float(np.linalg.norm(va_base)) + float(np.linalg.norm(vb_base))
                            )
                            rel_den = max(1e-12, base_pair_norm)
                            concept_score_rel_delta_pair = concept_score_delta_pair / rel_den
                            concept_score_abs_rel_delta_pair = concept_score_abs_delta_pair / rel_den
                            concept_score_delta_values.append(float(concept_score_delta_pair))
                            concept_score_abs_delta_values.append(float(concept_score_abs_delta_pair))
                            concept_score_rel_delta_values.append(float(concept_score_rel_delta_pair))
                            concept_score_abs_rel_delta_values.append(float(concept_score_abs_rel_delta_pair))
                        else:
                            concept_score_rel_delta_pair = None
                            concept_score_abs_rel_delta_pair = None

                        pair_id = _stable_hash8_text(f"{name_a}\0{name_b}\0{pair_idx}")
                        row_id = _stable_hash8_text(
                            json.dumps(
                                {
                                    **block_key_obj,
                                    "pair_id": pair_id,
                                    "group_a": name_a,
                                    "group_b": name_b,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        rec = {
                            **block_key_obj,
                            "pair_id": pair_id,
                            "row_id": row_id,
                            "group_a": name_a,
                            "group_b": name_b,
                            "split": "test" if is_test else "train",
                            "mode": args.mode,
                            "alpha": float(args.alpha),
                            "alpha_policy": args.alpha_policy,
                            "layer_idx": int(layer_idx),
                            "direction_control": args.direction_control,
                            "pll_bias_base_raw": pll_bias_base_raw,
                            "pll_bias_int_raw": pll_bias_int_raw,
                            "pll_bias_delta_raw": pll_bias_int_raw - pll_bias_base_raw,
                            "pll_abs_reduction_raw": abs(pll_bias_base_raw) - abs(pll_bias_int_raw),
                            "pll_pair_mean_logprob_change": pair_mean_logprob_change,
                            "pll_bias_base_adj": pll_bias_base_adj,
                            "pll_bias_int_adj": pll_bias_int_adj,
                            "pll_bias_delta_adj": (
                                (pll_bias_int_adj - pll_bias_base_adj)
                                if pll_bias_base_adj is not None and pll_bias_int_adj is not None
                                else None
                            ),
                            "pll_abs_reduction_adj": (
                                (abs(pll_bias_base_adj) - abs(pll_bias_int_adj))
                                if pll_bias_base_adj is not None and pll_bias_int_adj is not None
                                else None
                            ),
                            "concept_score_delta_a": concept_score_delta_a,
                            "concept_score_delta_b": concept_score_delta_b,
                            "concept_score_delta_pair": concept_score_delta_pair,
                            "concept_score_abs_delta_pair": concept_score_abs_delta_pair,
                            "concept_score_rel_delta_pair": concept_score_rel_delta_pair,
                            "concept_score_abs_rel_delta_pair": concept_score_abs_rel_delta_pair,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        out_rows.append(rec)
                        logprob_pair_change.append(float(pair_mean_logprob_change))
                        telemetry.log({"type": "causal_pair", **rec})

                        if rec["pll_bias_delta_adj"] is not None:
                            block_delta_adj.setdefault(block_key, []).append(float(rec["pll_bias_delta_adj"]))
                            block_abs_reduction_adj.setdefault(block_key, []).append(float(rec["pll_abs_reduction_adj"]))

    base_adj = [float(r["pll_bias_base_adj"]) for r in out_rows if r.get("pll_bias_base_adj") is not None]
    int_adj = [float(r["pll_bias_int_adj"]) for r in out_rows if r.get("pll_bias_int_adj") is not None]
    delta_adj = [float(r["pll_bias_delta_adj"]) for r in out_rows if r.get("pll_bias_delta_adj") is not None]
    abs_red_adj = [float(r["pll_abs_reduction_adj"]) for r in out_rows if r.get("pll_abs_reduction_adj") is not None]

    sign_flip = 0
    sign_total = 0
    for b, i in zip(base_adj, int_adj):
        if abs(b) <= 1e-12 or abs(i) <= 1e-12:
            continue
        sign_total += 1
        if (b > 0 and i < 0) or (b < 0 and i > 0):
            sign_flip += 1

    corr_base_int_adj = _spearman(base_adj, int_adj) if len(base_adj) >= 3 else None
    obs_delta, ci_delta = _block_bootstrap_ci(block_delta_adj, args.bootstrap_samples, args.holdout_seed ^ 0x1234)
    obs_abs_red, ci_abs_red = _block_bootstrap_ci(
        block_abs_reduction_adj, args.bootstrap_samples, args.holdout_seed ^ 0x5678
    )

    summary = {
        "schema_version": 3,
        "runtime_meta": _runtime_meta(),
        "run_id": run_id,
        "source_run": str(source_run),
        "direction_info": d_info,
        "direction_control": args.direction_control,
        "control_seed": int(control_seed),
        "direction_seed": int(direction_seed),
        "direction_path": str(direction_path),
        "direction_path_control": str(direction_path_control),
        "direction_sha256": direction_sha256,
        "direction_dtype": str(np.asarray(d_hat).dtype),
        "direction_shape": list(np.asarray(d_hat).shape),
        "direction_dim": int(d_hat.shape[0]),
        "direction_norm": float(np.linalg.norm(d_hat)),
        "direction_cos_to_source": direction_cos_to_source,
        "direction_shape_stats": direction_shape_stats,
        "hidden_cache_max": int(getattr(scorer, "_hidden_cache_max", 0)),
        "hidden_layout": (getattr(scorer, "_last_hidden_layout", None) if args.log_concept_score else None),
        "layer_idx": int(layer_idx),
        "mode": args.mode,
        "alpha": float(args.alpha),
        "alpha_policy": args.alpha_policy,
        "intervention_scope": args.intervention_scope,
        "filter_template": args.filter_template,
        "eval_split": args.eval_split,
        "holdout_frac": float(args.holdout_frac),
        "n_rows": len(out_rows),
        "n_blocks": len(block_delta_adj),
        "mean_delta_adj": float(np.mean(np.array(delta_adj, dtype=float))) if delta_adj else None,
        "mean_abs_reduction_adj": float(np.mean(np.array(abs_red_adj, dtype=float))) if abs_red_adj else None,
        "mean_logprob_change_pair": (
            float(np.mean(np.array(logprob_pair_change, dtype=float))) if logprob_pair_change else None
        ),
        "mean_pll_change": (
            float(np.mean(np.array(pll_change_values, dtype=float))) if pll_change_values else None
        ),
        "mean_concept_score_delta_pair": (
            float(np.mean(np.array(concept_score_delta_values, dtype=float))) if concept_score_delta_values else None
        ),
        "mean_concept_score_abs_delta_pair": (
            float(np.mean(np.array(concept_score_abs_delta_values, dtype=float)))
            if concept_score_abs_delta_values
            else None
        ),
        "mean_concept_score_rel_delta_pair": (
            float(np.mean(np.array(concept_score_rel_delta_values, dtype=float))) if concept_score_rel_delta_values else None
        ),
        "mean_concept_score_abs_rel_delta_pair": (
            float(np.mean(np.array(concept_score_abs_rel_delta_values, dtype=float)))
            if concept_score_abs_rel_delta_values
            else None
        ),
        "concept_score_enabled": bool(args.log_concept_score),
        "concept_score_pool": args.concept_score_pool if args.log_concept_score else None,
        "concept_direction_sha256": concept_direction_sha256,
        "corr_base_vs_int_adj_spearman": corr_base_int_adj,
        "sign_flip_rate_adj": (float(sign_flip / sign_total) if sign_total > 0 else None),
        "block_bootstrap_samples": int(args.bootstrap_samples),
        "block_bootstrap_delta_adj": {"mean": obs_delta, "ci95": ci_delta},
        "block_bootstrap_abs_reduction_adj": {"mean": obs_abs_red, "ci95": ci_abs_red},
    }

    (run_dir / "results_causal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    telemetry.log({"type": "causal_summary", **summary})
    telemetry.log({"type": "run_end", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.log({"type": "run_complete", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.close()
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
