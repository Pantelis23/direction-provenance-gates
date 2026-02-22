#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.pll import PLLScorer
from src.telemetry.logger import TelemetryLogger
from src.utils.config import load_config
from src.utils.seed import set_seed


def _resolve_groups(tpl: dict) -> list[tuple[str, str]]:
    pairs = tpl.get("group_pairs", [])
    if pairs:
        out = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("group_pairs must contain [A, B] pairs")
            out.append((str(pair[0]), str(pair[1])))
        return out

    group_a = tpl.get("groupA", [])
    group_b = tpl.get("groupB", [])
    if isinstance(group_a, str):
        group_a = [group_a]
    if isinstance(group_b, str):
        group_b = [group_b]
    if len(group_a) != len(group_b):
        raise ValueError("groupA/groupB must have the same length for paired training.")
    return list(zip([str(v) for v in group_a], [str(v) for v in group_b]))


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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        return 0.0
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if den <= 0:
        return 0.0
    return float(np.dot(x0, y0) / den)


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(a.shape[0], dtype=float)
    return r


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default="learn_control_dir")
    ap.add_argument("--filter-template", default="gender_names_jobs")
    ap.add_argument("--layer", default="late")
    ap.add_argument("--pool", choices=["mean", "last", "cls"], default="mean")
    ap.add_argument("--target", choices=["adj", "raw"], default="adj")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
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

    scorer = PLLScorer(cfg.get("model", {}))
    n_layers = len(scorer._resolve_layers())
    layer_cfg = args.layer
    if isinstance(layer_cfg, str) and layer_cfg.isdigit():
        layer_cfg = int(layer_cfg)
    layer_idx = _resolve_layer_index(layer_cfg, n_layers)

    hidden_cache: dict[tuple[str, int, str], np.ndarray] = {}

    def _hidden(text: str) -> np.ndarray:
        key = (text, layer_idx, args.pool)
        if key in hidden_cache:
            return hidden_cache[key]
        vec = scorer.extract_hidden(text=text, layer=layer_idx, pool=args.pool)
        hidden_cache[key] = vec
        return vec

    rows = []
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    pll_templates = cfg.get("data", {}).get("pll_templates", [])
    for tpl in pll_templates:
        tpl_name = str(tpl.get("name", "unnamed"))
        if tpl_name != args.filter_template:
            continue

        templates = tpl.get("templates", [])
        jobs = tpl.get("jobs", [])
        adjust = bool(tpl.get("adjust_by_baseline", False))
        baseline_templates = tpl.get("baseline_templates", [])
        baseline_template = tpl.get("baseline_template")
        pairs = _resolve_groups(tpl)
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
                if is_test:
                    continue  # train split only

                for pair_idx, (name_a, name_b) in enumerate(pairs):
                    text_a = _fill(tpl_str, name_a, job=job, article=article)
                    text_b = _fill(tpl_str, name_b, job=job, article=article)

                    a = scorer.score_sentence(text_a)
                    b = scorer.score_sentence(text_b)
                    y_raw = float(a.avg_logprob - b.avg_logprob)
                    y_adj = None

                    if adjust:
                        base_a_text = _fill(base_tpl, name_a)
                        base_b_text = _fill(base_tpl, name_b)
                        ba = scorer.score_sentence(base_a_text)
                        bb = scorer.score_sentence(base_b_text)
                        y_adj = float((a.avg_logprob - ba.avg_logprob) - (b.avg_logprob - bb.avg_logprob))

                    y_val = y_adj if args.target == "adj" else y_raw
                    if y_val is None:
                        continue

                    xa = _hidden(text_a)
                    xb = _hidden(text_b)
                    x_delta = np.asarray(xa - xb, dtype=np.float32)

                    row = {
                        **block_key_obj,
                        "pair_idx": int(pair_idx),
                        "group_a": name_a,
                        "group_b": name_b,
                        "y_raw": y_raw,
                        "y_adj": y_adj,
                        "y_target": y_val,
                    }
                    rows.append(row)
                    X_list.append(x_delta)
                    y_list.append(float(y_val))

    if len(X_list) < 3:
        raise ValueError(f"Not enough training rows to learn direction: {len(X_list)}")

    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.asarray(y_list, dtype=np.float64)

    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean

    lam = float(args.ridge_lambda)
    d = Xc.shape[1]
    xtx = Xc.T @ Xc
    rhs = Xc.T @ yc
    w = np.linalg.solve(xtx + lam * np.eye(d, dtype=np.float64), rhs)
    w = np.asarray(w, dtype=np.float32)
    w_norm = float(np.linalg.norm(w))
    if w_norm <= 1e-12:
        raise ValueError("Learned direction has near-zero norm.")
    d_hat = w / w_norm

    pred = Xc @ w + y_mean
    train_pearson = _pearson(pred, y)
    train_spearman = _spearman(pred, y)
    train_mse = float(np.mean((pred - y) ** 2))

    np.save(run_dir / "control_dir.npy", d_hat.astype(np.float32))
    (run_dir / "train_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "filter_template": args.filter_template,
        "layer_idx": int(layer_idx),
        "pool": args.pool,
        "target": args.target,
        "holdout_frac": float(args.holdout_frac),
        "holdout_seed": int(args.holdout_seed),
        "ridge_lambda": lam,
        "n_train_rows": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "direction_norm_pre_normalize": w_norm,
        "train_target_mean": y_mean,
        "train_pearson_pred_target": train_pearson,
        "train_spearman_pred_target": train_spearman,
        "train_mse": train_mse,
        "direction_path": "control_dir.npy",
    }
    (run_dir / "control_dir_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    telemetry.log({"type": "control_dir_summary", **summary})
    telemetry.log({"type": "run_end", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.log({"type": "run_complete", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.close()
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
