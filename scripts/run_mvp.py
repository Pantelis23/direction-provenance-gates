#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.logger import TelemetryLogger
from src.utils.config import load_config
from src.utils.seed import set_seed
import numpy as np

from src.metrics.embedding import (
    EmbeddingExtractor,
    EmbeddingVariant,
    compute_seat_weat,
)
from src.metrics.pll import PLLScorer


def _std(vals: list[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
    return var ** 0.5


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = float(rank)
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    den = den_a * den_b
    return num / den if den > 0 else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


def _l2_norm_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _stable_hash8_text(text: str) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=8).hexdigest()


def _git_hash(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _system_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "time_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        info["torch"] = "not_installed"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_cfg = config.get("run", {})

    set_seed(run_cfg.get("seed", 42))

    output_root = Path(run_cfg.get("output_dir", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = run_cfg.get("name", "run") + "_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    telemetry = TelemetryLogger(run_dir / "telemetry.jsonl")

    telemetry.log({"type": "run_start", "run_id": run_id})
    telemetry.log({"type": "git", "hash": _git_hash(Path.cwd())})
    telemetry.log({"type": "system", **_system_info()})
    telemetry.log({"type": "config", "config": config})

    data_cfg = config.get("data", {})
    prompt_pairs = data_cfg.get("prompt_pairs", [])
    pll_templates = data_cfg.get("pll_templates", [])
    seat_tests = data_cfg.get("seat_tests", [])
    embedding_variants = config.get("embedding_variants", [])

    emb_enabled = bool(config.get("metrics", {}).get("embedding_bias", {}).get("enabled", False))
    pll_cfg = config.get("metrics", {}).get("pll_bias", {})
    pll_enabled = bool(pll_cfg.get("enabled", False))
    pll_debug = bool(pll_cfg.get("debug", False))
    pll_dump_tokens = bool(pll_cfg.get("dump_tokens", False))
    pll_sanity = bool(pll_cfg.get("sanity_checks", False))

    extractor = None
    pll_scorer = None
    pll_pair_bias_map = {}
    if emb_enabled:
        if not seat_tests:
            raise ValueError("Embedding bias enabled, but data.seat_tests is empty.")
        extractor = EmbeddingExtractor(config.get("model", {}))
    if pll_enabled:
        pll_scorer = PLLScorer(config.get("model", {}))
        if pll_sanity:
            sanity_pairs = [
                ("I like apples.", "I like bananas."),
                ("the the the the", "This is a coherent sentence."),
            ]
            for a, b in sanity_pairs:
                s = pll_scorer.score_sentence(a, debug=pll_debug)
                t = pll_scorer.score_sentence(b, debug=pll_debug)
                telemetry.log(
                    {
                        "type": "pll_sanity",
                        "a": a,
                        "b": b,
                        "a_avg": s.avg_logprob,
                        "b_avg": t.avg_logprob,
                        "delta_avg": s.avg_logprob - t.avg_logprob,
                    }
                )

        pll_records = []
        pll_pair_bias_map = {}
        pll_summary = {}
        pll_specs = []
        swap_err_raw_max = 0.0
        swap_err_adj_max = 0.0
        swap_err_prior_max = 0.0
    if pll_enabled and pll_scorer is not None:
        def _article(job_entry) -> tuple[str, str]:
            if isinstance(job_entry, dict):
                job = job_entry.get("job")
                article = job_entry.get("article")
            else:
                job = str(job_entry)
                article = None
            if not job:
                raise ValueError("PLL job entry missing job name.")
            if article is None:
                article = "an" if job[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            return job, article

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

        def _compute_bias(
            scores_a,
            scores_b,
            base_scores_a,
            base_scores_b,
            paired: bool,
        ):
            avg_a = sum(s.avg_logprob for s in scores_a) / max(len(scores_a), 1)
            avg_b = sum(s.avg_logprob for s in scores_b) / max(len(scores_b), 1)
            if paired:
                raw_diffs = [sa.avg_logprob - sb.avg_logprob for sa, sb in zip(scores_a, scores_b)]
                bias_raw = sum(raw_diffs) / max(len(raw_diffs), 1)
            else:
                bias_raw = avg_a - avg_b

            if base_scores_a is not None and base_scores_b is not None:
                base_avg_a = sum(s.avg_logprob for s in base_scores_a) / max(len(base_scores_a), 1)
                base_avg_b = sum(s.avg_logprob for s in base_scores_b) / max(len(base_scores_b), 1)
                if paired:
                    deltas = [
                        (sa.avg_logprob - ba.avg_logprob) - (sb.avg_logprob - bb.avg_logprob)
                        for sa, sb, ba, bb in zip(scores_a, scores_b, base_scores_a, base_scores_b)
                    ]
                    bias_adj = sum(deltas) / max(len(deltas), 1)
                    name_prior_gap = sum(
                        ba.avg_logprob - bb.avg_logprob for ba, bb in zip(base_scores_a, base_scores_b)
                    ) / max(len(base_scores_a), 1)
                else:
                    deltas_a = [s.avg_logprob - b.avg_logprob for s, b in zip(scores_a, base_scores_a)]
                    deltas_b = [s.avg_logprob - b.avg_logprob for s, b in zip(scores_b, base_scores_b)]
                    avg_delta_a = sum(deltas_a) / max(len(deltas_a), 1)
                    avg_delta_b = sum(deltas_b) / max(len(deltas_b), 1)
                    bias_adj = avg_delta_a - avg_delta_b
                    name_prior_gap = base_avg_a - base_avg_b
            else:
                base_avg_a = None
                base_avg_b = None
                bias_adj = None
                name_prior_gap = None

            return {
                "avg_a": avg_a,
                "avg_b": avg_b,
                "base_avg_a": base_avg_a,
                "base_avg_b": base_avg_b,
                "bias_raw": bias_raw,
                "bias_adj": bias_adj,
                "name_prior_gap": name_prior_gap,
            }

        def _embed_bias(
            emb_a: np.ndarray,
            emb_b: np.ndarray,
            base_a: np.ndarray | None,
            base_b: np.ndarray | None,
            paired: bool,
            normalize: bool,
            direction: np.ndarray | None = None,
        ):
            if normalize:
                emb_a = _l2_norm_rows(emb_a)
                emb_b = _l2_norm_rows(emb_b)
                if base_a is not None and base_b is not None:
                    base_a = _l2_norm_rows(base_a)
                    base_b = _l2_norm_rows(base_b)

            if paired:
                delta_raw = (emb_a - emb_b).mean(axis=0)
            else:
                delta_raw = emb_a.mean(axis=0) - emb_b.mean(axis=0)

            if base_a is not None and base_b is not None:
                if paired:
                    delta = ((emb_a - base_a) - (emb_b - base_b)).mean(axis=0)
                else:
                    delta = (emb_a - base_a).mean(axis=0) - (emb_b - base_b).mean(axis=0)
                d = base_a.mean(axis=0) - base_b.mean(axis=0)
            else:
                delta = None
                d = (emb_a.mean(axis=0) - emb_b.mean(axis=0))

            if direction is not None:
                d = direction

            d_norm = float(np.linalg.norm(d)) if d is not None else 0.0
            if d_norm < 1e-12:
                return {
                    "embed_bias_raw": None,
                    "embed_bias_adj": None,
                    "d_norm": d_norm,
                    "delta_raw": delta_raw,
                    "delta_adj": delta,
                    "d": d,
                }

            embed_bias_raw = float(delta_raw.dot(d) / d_norm)
            embed_bias_adj = float(delta.dot(d) / d_norm) if delta is not None else None

            return {
                "embed_bias_raw": embed_bias_raw,
                "embed_bias_adj": embed_bias_adj,
                "d_norm": d_norm,
                "delta_raw": delta_raw,
                "delta_adj": delta,
                "d": d,
            }

        def _resolve_groups(tpl: dict) -> tuple[list[str], list[str], list[tuple[str, str]] | None]:
            pairs = tpl.get("group_pairs", [])
            if pairs:
                ga, gb = [], []
                seen_pairs: set[tuple[str, str]] = set()
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError("group_pairs must be list of [A, B] pairs.")
                    a_name = str(pair[0])
                    b_name = str(pair[1])
                    pair_key = (a_name, b_name)
                    if pair_key in seen_pairs:
                        raise ValueError(f"Duplicate group pair detected: {pair_key}. Group pairs must be unique.")
                    seen_pairs.add(pair_key)
                    ga.append(a_name)
                    gb.append(b_name)
                return ga, gb, [(str(p[0]), str(p[1])) for p in pairs]
            group_a = tpl.get("groupA", [])
            group_b = tpl.get("groupB", [])
            if isinstance(group_a, str):
                group_a = [group_a]
            if isinstance(group_b, str):
                group_b = [group_b]
            return list(group_a), list(group_b), None

        def _pair_row_id(
            tpl_name: str,
            tpl_str: str,
            job: str | None,
            article: str | None,
            baseline_str: str | None,
            pair_id: str,
            group_a_name: str,
            group_b_name: str,
        ) -> str:
            payload = {
                "pll_template": tpl_name,
                "pll_template_str": tpl_str,
                "pll_job": job,
                "pll_article": article,
                "pll_baseline_template": baseline_str,
                "pair_id": pair_id,
                "group_a": group_a_name,
                "group_b": group_b_name,
            }
            return _stable_hash8_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

        def _pair_id(name_a: str, name_b: str, occurrence: int = 0) -> str:
            # Stable across pools/configs as long as pair names match.
            if occurrence > 0:
                return _stable_hash8_text(f"{name_a}\0{name_b}\0{occurrence}")
            return _stable_hash8_text(f"{name_a}\0{name_b}")

        pll_path = run_dir / "results_pll.jsonl"
        with pll_path.open("w", encoding="utf-8") as f_pll:
            if pll_templates:
                for tpl in pll_templates:
                    template = tpl.get("template", "")
                    templates = tpl.get("templates", [])
                    jobs = tpl.get("jobs", [])
                    tpl_name = tpl.get("name", "unnamed")

                    group_a, group_b, group_pairs = _resolve_groups(tpl)
                    baseline_template = tpl.get("baseline_template")
                    baseline_templates = tpl.get("baseline_templates", [])
                    adjust_by_baseline = bool(tpl.get("adjust_by_baseline", False))
                    if adjust_by_baseline and not baseline_template and not baseline_templates:
                        raise ValueError(f"PLL template '{tpl_name}' requires baseline_template when adjust_by_baseline=true.")

                    if templates and jobs:
                        if baseline_templates and len(baseline_templates) != len(templates):
                            raise ValueError(f"PLL template '{tpl_name}' baseline_templates must match templates length.")
                        for tpl_idx, tpl_str in enumerate(templates):
                            baseline_str = None
                            if adjust_by_baseline:
                                baseline_str = baseline_templates[tpl_idx] if baseline_templates else baseline_template
                            if adjust_by_baseline and not baseline_str:
                                raise ValueError(f"PLL template '{tpl_name}' missing baseline for template '{tpl_str}'.")
                            baseline_key = baseline_str if adjust_by_baseline else None

                            if adjust_by_baseline:
                                base_a_texts = [_fill(baseline_str, v) for v in group_a]
                                base_b_texts = [_fill(baseline_str, v) for v in group_b]
                                base_scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_a_texts]
                                base_scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_b_texts]
                                base_avg_a = sum(s.avg_logprob for s in base_scores_a) / max(len(base_scores_a), 1)
                                base_avg_b = sum(s.avg_logprob for s in base_scores_b) / max(len(base_scores_b), 1)
                            else:
                                base_scores_a = []
                                base_scores_b = []
                                base_avg_a = None
                                base_avg_b = None

                            for job_entry in jobs:
                                job, article = _article(job_entry)
                                texts_a = [_fill(tpl_str, v, job=job, article=article) for v in group_a]
                                texts_b = [_fill(tpl_str, v, job=job, article=article) for v in group_b]

                                start = time.time()
                                scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_a]
                                scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_b]
                                elapsed = time.time() - start

                                bias = _compute_bias(
                                    scores_a,
                                    scores_b,
                                    base_scores_a if adjust_by_baseline else None,
                                    base_scores_b if adjust_by_baseline else None,
                                    paired=group_pairs is not None,
                                )
                                bias_swapped = _compute_bias(
                                    scores_b,
                                    scores_a,
                                    base_scores_b if adjust_by_baseline else None,
                                    base_scores_a if adjust_by_baseline else None,
                                    paired=group_pairs is not None,
                                )
                                swap_err_raw_max = max(swap_err_raw_max, abs((bias_swapped["bias_raw"] or 0.0) + (bias["bias_raw"] or 0.0)))
                                if bias["bias_adj"] is not None and bias_swapped["bias_adj"] is not None:
                                    swap_err_adj_max = max(swap_err_adj_max, abs(bias_swapped["bias_adj"] + bias["bias_adj"]))
                                if bias["name_prior_gap"] is not None and bias_swapped["name_prior_gap"] is not None:
                                    swap_err_prior_max = max(swap_err_prior_max, abs(bias_swapped["name_prior_gap"] + bias["name_prior_gap"]))

                                record = {
                                    "variant": None,
                                    "pll_template": tpl_name,
                                    "pll_template_str": tpl_str,
                                    "pll_job": job,
                                    "pll_article": article,
                                    "pll_baseline_template": baseline_key,
                                    "pll_bias": bias["bias_adj"] if adjust_by_baseline else bias["bias_raw"],
                                    "pll_bias_raw": bias["bias_raw"],
                                    "pll_bias_adj": bias["bias_adj"],
                                    "pll_name_prior_gap": bias["name_prior_gap"],
                                    "pll_detail": {
                                        "groupA_avg": bias["avg_a"],
                                        "groupB_avg": bias["avg_b"],
                                        "groupA_base_avg": bias["base_avg_a"],
                                        "groupB_base_avg": bias["base_avg_b"],
                                        "groupA_count": len(scores_a),
                                        "groupB_count": len(scores_b),
                                        "groupA_texts": texts_a if pll_debug else None,
                                        "groupB_texts": texts_b if pll_debug else None,
                                    },
                                    "elapsed_s": elapsed,
                                }
                                f_pll.write(json.dumps(record) + "\n")
                                telemetry.log({"type": "pll_metric", **record})
                                pll_records.append(record)

                                if group_pairs is not None:
                                    pair_occurrence: dict[tuple[str, str], int] = {}
                                    for pair_idx, (name_a, name_b) in enumerate(group_pairs):
                                        pair_key_names = (name_a, name_b)
                                        occ = pair_occurrence.get(pair_key_names, 0)
                                        pair_occurrence[pair_key_names] = occ + 1
                                        pair_id = _pair_id(name_a, name_b, occ)
                                        raw_pair = scores_a[pair_idx].avg_logprob - scores_b[pair_idx].avg_logprob
                                        if adjust_by_baseline:
                                            adj_pair = (
                                                (scores_a[pair_idx].avg_logprob - base_scores_a[pair_idx].avg_logprob)
                                                - (scores_b[pair_idx].avg_logprob - base_scores_b[pair_idx].avg_logprob)
                                            )
                                        else:
                                            adj_pair = None
                                        row_id = _pair_row_id(
                                            tpl_name,
                                            tpl_str,
                                            job,
                                            article,
                                            baseline_key,
                                            pair_id,
                                            name_a,
                                            name_b,
                                        )
                                        pair_key = (
                                            tpl_name,
                                            tpl_str,
                                            job,
                                            article,
                                            baseline_key,
                                            pair_id,
                                            name_a,
                                            name_b,
                                        )
                                        pll_pair_bias_map[pair_key] = {
                                            "pair_id": pair_id,
                                            "group_a": name_a,
                                            "group_b": name_b,
                                            "row_id": row_id,
                                            "pll_bias_raw": raw_pair,
                                            "pll_bias_adj": adj_pair,
                                            "pll_bias": adj_pair if adj_pair is not None else raw_pair,
                                        }
                                pll_specs.append(
                                    {
                                        "pll_template": tpl_name,
                                        "pll_template_str": tpl_str,
                                        "pll_job": job,
                                        "pll_article": article,
                                        "pll_baseline_template": baseline_key,
                                        "group_a": list(group_a),
                                        "group_b": list(group_b),
                                        "group_pairs": list(group_pairs) if group_pairs is not None else None,
                                        "paired": group_pairs is not None,
                                        "baseline_target": tpl.get("baseline_target") or "person",
                                    }
                                )

                                if pll_debug:
                                    debug_record = {
                                        "variant": None,
                                        "template": tpl_str,
                                        "job": job,
                                        "groupA_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts_a],
                                        "groupB_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts_b],
                                        "groupA_ids": [s.token_ids[:10] if s.token_ids else None for s in scores_a],
                                        "groupB_ids": [s.token_ids[:10] if s.token_ids else None for s in scores_b],
                                        "groupA_lens": [s.token_count for s in scores_a],
                                        "groupB_lens": [s.token_count for s in scores_b],
                                    }
                                    telemetry.log({"type": "pll_debug", **debug_record})
                    else:
                        if not template:
                            raise ValueError("PLL template missing 'template' field.")

                        texts_a = [_fill(template, v) for v in group_a]
                        texts_b = [_fill(template, v) for v in group_b]

                        start = time.time()
                        scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_a]
                        scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_b]
                        if adjust_by_baseline:
                            base_a = [_fill(baseline_template, v) for v in group_a]
                            base_b = [_fill(baseline_template, v) for v in group_b]
                            base_scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_a]
                            base_scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_b]
                            base_avg_a = sum(s.avg_logprob for s in base_scores_a) / max(len(base_scores_a), 1)
                            base_avg_b = sum(s.avg_logprob for s in base_scores_b) / max(len(base_scores_b), 1)
                        else:
                            base_scores_a = []
                            base_scores_b = []
                            base_avg_a = None
                            base_avg_b = None
                        elapsed = time.time() - start

                        bias = _compute_bias(
                            scores_a,
                            scores_b,
                            base_scores_a if adjust_by_baseline else None,
                            base_scores_b if adjust_by_baseline else None,
                            paired=group_pairs is not None,
                        )
                        bias_swapped = _compute_bias(
                            scores_b,
                            scores_a,
                            base_scores_b if adjust_by_baseline else None,
                            base_scores_a if adjust_by_baseline else None,
                            paired=group_pairs is not None,
                        )
                        swap_err_raw_max = max(swap_err_raw_max, abs((bias_swapped["bias_raw"] or 0.0) + (bias["bias_raw"] or 0.0)))
                        if bias["bias_adj"] is not None and bias_swapped["bias_adj"] is not None:
                            swap_err_adj_max = max(swap_err_adj_max, abs(bias_swapped["bias_adj"] + bias["bias_adj"]))
                        if bias["name_prior_gap"] is not None and bias_swapped["name_prior_gap"] is not None:
                            swap_err_prior_max = max(swap_err_prior_max, abs(bias_swapped["name_prior_gap"] + bias["name_prior_gap"]))

                        record = {
                            "variant": None,
                            "pll_template": tpl_name,
                            "pll_template_str": template,
                            "pll_baseline_template": baseline_template if adjust_by_baseline else None,
                            "pll_bias": bias["bias_adj"] if adjust_by_baseline else bias["bias_raw"],
                            "pll_bias_raw": bias["bias_raw"],
                            "pll_bias_adj": bias["bias_adj"],
                            "pll_name_prior_gap": bias["name_prior_gap"],
                            "pll_detail": {
                                "groupA_avg": bias["avg_a"],
                                "groupB_avg": bias["avg_b"],
                                "groupA_base_avg": bias["base_avg_a"],
                                "groupB_base_avg": bias["base_avg_b"],
                                "groupA_count": len(scores_a),
                                "groupB_count": len(scores_b),
                                "groupA_texts": texts_a if pll_debug else None,
                                "groupB_texts": texts_b if pll_debug else None,
                            },
                            "elapsed_s": elapsed,
                        }
                        f_pll.write(json.dumps(record) + "\n")
                        telemetry.log({"type": "pll_metric", **record})
                        pll_records.append(record)
                        pll_specs.append(
                            {
                                "pll_template": tpl_name,
                                "pll_template_str": template,
                                "pll_job": None,
                                "pll_article": None,
                                "pll_baseline_template": baseline_template if adjust_by_baseline else None,
                                "group_a": list(group_a),
                                "group_b": list(group_b),
                                "group_pairs": list(group_pairs) if group_pairs is not None else None,
                                "paired": group_pairs is not None,
                                "baseline_target": tpl.get("baseline_target") or "person",
                            }
                        )

                        if pll_debug:
                            debug_record = {
                                "variant": None,
                                "template": template,
                                "groupA_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts_a],
                                "groupB_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts_b],
                                "groupA_ids": [s.token_ids[:10] if s.token_ids else None for s in scores_a],
                                "groupB_ids": [s.token_ids[:10] if s.token_ids else None for s in scores_b],
                                "groupA_lens": [s.token_count for s in scores_a],
                                "groupB_lens": [s.token_count for s in scores_b],
                            }
                            telemetry.log({"type": "pll_debug", **debug_record})
            else:
                for pair in prompt_pairs:
                    start = time.time()
                    pll_result = pll_scorer.score_pair(pair, debug=pll_debug or pll_dump_tokens)
                    elapsed = time.time() - start

                    stereo = pair.get("stereo", "")
                    anti = pair.get("anti", "")

                    record = {
                        "variant": None,
                        "pair": pair,
                        "pll_bias": pll_result["delta_avg"],
                        "pll_detail": {
                            "stereo_sum": pll_result["stereo"].sum_logprob,
                            "stereo_avg": pll_result["stereo"].avg_logprob,
                            "stereo_tokens": pll_result["stereo"].token_count,
                            "anti_sum": pll_result["anti"].sum_logprob,
                            "anti_avg": pll_result["anti"].avg_logprob,
                            "anti_tokens": pll_result["anti"].token_count,
                            "delta_sum": pll_result["delta_sum"],
                            "delta_avg": pll_result["delta_avg"],
                        },
                        "elapsed_s": elapsed,
                    }
                    f_pll.write(json.dumps(record) + "\n")
                    telemetry.log({"type": "pll_metric", **record})
                    pll_records.append(record)

                    if pll_debug:
                        debug_record = {
                            "variant": None,
                            "stereo_text": stereo,
                            "anti_text": anti,
                            "stereo_hash": hashlib.sha256(stereo.encode("utf-8")).hexdigest()[:16],
                            "anti_hash": hashlib.sha256(anti.encode("utf-8")).hexdigest()[:16],
                            "stereo_ids": pll_result["stereo"].token_ids,
                            "anti_ids": pll_result["anti"].token_ids,
                            "stereo_token_logprobs": pll_result["stereo"].token_logprobs,
                            "anti_token_logprobs": pll_result["anti"].token_logprobs,
                        }
                        telemetry.log({"type": "pll_debug", **debug_record})

        if pll_scorer is not None:
            telemetry.log({"type": "pll_cache_keys", "keys": list(pll_scorer._cache.keys())})

        # PLL summary stats
        if pll_records:
            from collections import defaultdict
            import random as _random

            by_job = defaultdict(list)
            by_template = defaultdict(list)
            by_tpl_str = defaultdict(dict)
            for r in pll_records:
                if r.get("pll_job") is not None:
                    by_job[r["pll_job"]].append(r["pll_bias"])
                tpl_key = r.get("pll_template_str") or r.get("pll_template")
                by_template[tpl_key].append(r["pll_bias"])
                if r.get("pll_job") is not None:
                    by_tpl_str[tpl_key][r["pll_job"]] = r["pll_bias"]

            job_stats = {k: {"mean": float(sum(v) / len(v)), "std": float((_std(v))) if len(v) > 1 else 0.0} for k, v in by_job.items()}
            tpl_stats = {k: {"mean": float(sum(v) / len(v)), "std": float((_std(v))) if len(v) > 1 else 0.0} for k, v in by_template.items()}

            # Raw/adjusted summaries
            job_stats_raw = defaultdict(list)
            job_stats_adj = defaultdict(list)
            for r in pll_records:
                if r.get("pll_job") is None:
                    continue
                job = r["pll_job"]
                job_stats_raw[job].append(r.get("pll_bias_raw", 0.0))
                if r.get("pll_bias_adj") is not None:
                    job_stats_adj[job].append(r.get("pll_bias_adj", 0.0))

            job_stats_raw = {k: {"mean": float(sum(v) / len(v)), "std": float((_std(v))) if len(v) > 1 else 0.0} for k, v in job_stats_raw.items()}
            job_stats_adj = {k: {"mean": float(sum(v) / len(v)), "std": float((_std(v))) if len(v) > 1 else 0.0} for k, v in job_stats_adj.items()}

            # Rank correlation across template families (Spearman)
            tpl_keys = list(by_tpl_str.keys())
            corr = []
            for i in range(len(tpl_keys)):
                for j in range(i + 1, len(tpl_keys)):
                    a = by_tpl_str[tpl_keys[i]]
                    b = by_tpl_str[tpl_keys[j]]
                    common = [k for k in a.keys() if k in b]
                    if len(common) < 3:
                        continue
                    ra = _ranks([a[k] for k in common])
                    rb = _ranks([b[k] for k in common])
                    corr.append(
                        {
                            "template_a": tpl_keys[i],
                            "template_b": tpl_keys[j],
                            "spearman": _pearson(ra, rb),
                            "n_jobs": len(common),
                        }
                    )

            pll_summary = {
                "job_stats": job_stats,
                "template_stats": tpl_stats,
                "template_rank_corr": corr,
                "job_stats_raw": job_stats_raw,
                "job_stats_adj": job_stats_adj,
                "swap_max_abs_err_raw": swap_err_raw_max,
                "swap_max_abs_err_adj": swap_err_adj_max,
                "swap_max_abs_err_prior": swap_err_prior_max,
            }

            # Random split control
            rnd_cfg = data_cfg.get("pll_random_split", {})
            if rnd_cfg and rnd_cfg.get("enabled", False):
                pool = list(rnd_cfg.get("pool", []))
                group_size = int(rnd_cfg.get("group_size", 0))
                n_splits = int(rnd_cfg.get("n_splits", 0))
                seed = int(rnd_cfg.get("seed", 0))
                templates = rnd_cfg.get("templates", [])
                baseline_templates = rnd_cfg.get("baseline_templates", [])
                baseline_template = rnd_cfg.get("baseline_template")
                jobs = rnd_cfg.get("jobs", [])
                adjust = bool(rnd_cfg.get("adjust_by_baseline", False))

                if not pool or group_size <= 0 or n_splits <= 0:
                    raise ValueError("pll_random_split requires pool, group_size, n_splits.")
                if len(pool) < group_size * 2:
                    raise ValueError("pll_random_split pool must be at least 2*group_size.")
                if adjust and not baseline_template and not baseline_templates:
                    raise ValueError("pll_random_split requires baseline_template(s) when adjust_by_baseline=true.")
                if baseline_templates and len(baseline_templates) != len(templates):
                    raise ValueError("pll_random_split baseline_templates must match templates length.")

                rng = _random.Random(seed)
                split_records = []
                for split_id in range(n_splits):
                    rng.shuffle(pool)
                    group_a = pool[:group_size]
                    group_b = pool[group_size:group_size * 2]

                    combo_bias_raw = []
                    combo_bias_adj = []
                    for t_idx, tpl_str in enumerate(templates):
                        baseline_str = None
                        if adjust:
                            baseline_str = baseline_templates[t_idx] if baseline_templates else baseline_template
                        if adjust and not baseline_str:
                            raise ValueError("pll_random_split missing baseline for template.")

                        if adjust:
                            base_a_texts = [_fill(baseline_str, v) for v in group_a]
                            base_b_texts = [_fill(baseline_str, v) for v in group_b]
                            base_scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_a_texts]
                            base_scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in base_b_texts]
                        else:
                            base_scores_a = None
                            base_scores_b = None

                        for job_entry in jobs:
                            job, article = _article(job_entry)
                            texts_a = [_fill(tpl_str, v, job=job, article=article) for v in group_a]
                            texts_b = [_fill(tpl_str, v, job=job, article=article) for v in group_b]

                            scores_a = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_a]
                            scores_b = [pll_scorer.score_sentence(t, debug=pll_debug) for t in texts_b]

                            bias = _compute_bias(
                                scores_a,
                                scores_b,
                                base_scores_a,
                                base_scores_b,
                                paired=False,
                            )
                            combo_bias_raw.append(bias["bias_raw"])
                            if bias["bias_adj"] is not None:
                                combo_bias_adj.append(bias["bias_adj"])

                    raw_mean = sum(combo_bias_raw) / max(len(combo_bias_raw), 1)
                    adj_mean = sum(combo_bias_adj) / max(len(combo_bias_adj), 1) if combo_bias_adj else 0.0
                    record = {
                        "split_id": split_id,
                        "bias_raw_mean": raw_mean,
                        "bias_adj_mean": adj_mean,
                        "bias_raw_std": _std(combo_bias_raw),
                        "bias_adj_std": _std(combo_bias_adj) if combo_bias_adj else 0.0,
                        "n_combos": len(combo_bias_raw),
                    }
                    split_records.append(record)

                rnd_path = run_dir / "results_pll_random_split.jsonl"
                rnd_path.write_text("\n".join(json.dumps(r) for r in split_records))

                adj_vals = [r["bias_adj_mean"] for r in split_records]
                raw_vals = [r["bias_raw_mean"] for r in split_records]
                pll_summary["random_split"] = {
                    "n_splits": n_splits,
                    "bias_raw_mean": sum(raw_vals) / len(raw_vals),
                    "bias_raw_std": _std(raw_vals),
                    "bias_raw_p05": _percentile(raw_vals, 5),
                    "bias_raw_p95": _percentile(raw_vals, 95),
                    "bias_adj_mean": sum(adj_vals) / len(adj_vals),
                    "bias_adj_std": _std(adj_vals),
                    "bias_adj_p05": _percentile(adj_vals, 5),
                    "bias_adj_p95": _percentile(adj_vals, 95),
                }

            pll_summary_path = run_dir / "results_pll_summary.json"
            pll_summary_path.write_text(json.dumps(pll_summary, indent=2))

    # Build PLL bias map for alignment
    pll_bias_map = {}
    for r in pll_records:
        key = (
            r.get("pll_template"),
            r.get("pll_template_str"),
            r.get("pll_job"),
            r.get("pll_article"),
            r.get("pll_baseline_template"),
        )
        pll_bias_map[key] = {
            "pll_bias_raw": r.get("pll_bias_raw"),
            "pll_bias_adj": r.get("pll_bias_adj"),
            "pll_bias": r.get("pll_bias_adj") if r.get("pll_bias_adj") is not None else r.get("pll_bias_raw"),
        }

    results_path = run_dir / "results_embed.jsonl"
    results_compat_path = run_dir / "results.jsonl"
    aligned_path = run_dir / "results_embed_aligned.jsonl"
    aligned_pairs_path = run_dir / "results_embed_aligned_pairs.jsonl"
    embedding_records = []
    aligned_records = []
    aligned_pair_records = []
    bias_dir_artifacts: dict[str, dict] = {}
    guardrail_logged_for: set[str] = set()
    with (
        results_path.open("w", encoding="utf-8") as f,
        results_compat_path.open("w", encoding="utf-8") as f_compat,
        aligned_path.open("w", encoding="utf-8") as f_aligned,
        aligned_pairs_path.open("w", encoding="utf-8") as f_aligned_pairs,
    ):
        for variant in embedding_variants:
            if emb_enabled and extractor is not None:
                for test in seat_tests:
                    start = time.time()
                    emb_score = compute_seat_weat(extractor, variant, test, config)
                    elapsed = time.time() - start

                    record = {
                        "variant": variant,
                        "seat_test": test.get("name", "unnamed"),
                        "embedding_bias": emb_score,
                        "elapsed_s": elapsed,
                    }
                    line = json.dumps(record) + "\n"
                    f.write(line)
                    f_compat.write(line)
                    telemetry.log({"type": "embedding_metric", **record})
                    embedding_records.append(record)

            # Aligned embedding bias per PLL combo
            if emb_enabled and extractor is not None and pll_specs:
                variant_key = json.dumps(variant, sort_keys=True, separators=(",", ":"))
                cache: dict[tuple[str, str, str], np.ndarray] = {}
                direction_samples: list[np.ndarray] = []

                def _postprocess_variant_pair(
                    emb_a: np.ndarray,
                    emb_b: np.ndarray,
                    base_emb_a: np.ndarray | None,
                    base_emb_b: np.ndarray | None,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
                    center = bool(variant.get("center", False))
                    whiten_k = int(variant.get("whiten_k", 0) or 0)
                    normalize = bool(variant.get("normalize", True))
                    parts = [emb_a, emb_b]
                    if base_emb_a is not None and base_emb_b is not None:
                        parts.extend([base_emb_a, base_emb_b])
                    all_emb = np.concatenate(parts, axis=0)

                    if center or whiten_k > 0:
                        all_emb = all_emb - all_emb.mean(axis=0, keepdims=True)
                    pcs = None
                    if whiten_k > 0:
                        _, _, vt = np.linalg.svd(all_emb, full_matrices=False)
                        k = min(whiten_k, vt.shape[0])
                        if k > 0:
                            pcs = vt[:k]
                            all_emb = all_emb - (all_emb @ pcs.T) @ pcs
                    if normalize:
                        all_emb = _l2_norm_rows(all_emb)

                    n_a = emb_a.shape[0]
                    n_b = emb_b.shape[0]
                    pa = all_emb[:n_a]
                    pb = all_emb[n_a : n_a + n_b]
                    if base_emb_a is None or base_emb_b is None:
                        return pa, pb, None, None, pcs
                    n_ba = base_emb_a.shape[0]
                    pba = all_emb[n_a + n_b : n_a + n_b + n_ba]
                    pbb = all_emb[n_a + n_b + n_ba :]
                    return pa, pb, pba, pbb, pcs

                def _encode_cached(sentences: list[str], targets: list[str] | None):
                    out = []
                    missing = []
                    missing_targets = []
                    for s, t in zip(sentences, targets or ["" for _ in sentences]):
                        key = (variant_key, s, t)
                        if key in cache:
                            out.append(cache[key])
                        else:
                            out.append(None)
                            missing.append(s)
                            missing_targets.append(t)
                    if missing:
                        emb = extractor.encode(
                            missing,
                            EmbeddingVariant(
                                layer=variant.get("layer", "late"),
                                pooling=variant.get("pooling", "mean"),
                                normalize=bool(variant.get("normalize", True)),
                            ),
                            targets=missing_targets if variant.get("pooling") == "target" else None,
                        )
                        for s, t, e in zip(missing, missing_targets, emb):
                            cache[(variant_key, s, t)] = e
                    # fill out list
                    idx = 0
                    final = []
                    for item in out:
                        if item is None:
                            s = missing[idx]
                            t = missing_targets[idx]
                            final.append(cache[(variant_key, s, t)])
                            idx += 1
                        else:
                            final.append(item)
                    return np.stack(final, axis=0)

                direction_cfg = config.get("metrics", {}).get("embedding_bias", {})
                direction_source = direction_cfg.get("direction_source", "baseline_names")
                direction_postprocess = direction_cfg.get("direction_postprocess", "match_variant")
                direction_templates = direction_cfg.get("direction_templates", [])
                direction_groups = direction_cfg.get("direction_groups", {})
                direction_group_a = list(direction_groups.get("A", []))
                direction_group_b = list(direction_groups.get("B", []))

                direction_vec = None
                if direction_source != "baseline_names":
                    if not direction_templates or not direction_group_a or not direction_group_b:
                        raise ValueError("direction_source requires direction_templates and direction_groups A/B.")

                    dir_texts_a = []
                    dir_texts_b = []
                    dir_targets_a = []
                    dir_targets_b = []
                    for tpl in direction_templates:
                        for x in direction_group_a:
                            text = tpl.format(x=x, name=x, group=x, subj=x)
                            dir_texts_a.append(text)
                            if variant.get("pooling") == "target":
                                dir_targets_a.append(x)
                        for x in direction_group_b:
                            text = tpl.format(x=x, name=x, group=x, subj=x)
                            dir_texts_b.append(text)
                            if variant.get("pooling") == "target":
                                dir_targets_b.append(x)

                    emb_dir_a = _encode_cached(dir_texts_a, dir_targets_a if variant.get("pooling") == "target" else None)
                    emb_dir_b = _encode_cached(dir_texts_b, dir_targets_b if variant.get("pooling") == "target" else None)
                    if direction_postprocess == "match_variant":
                        emb_dir_a, emb_dir_b, _, _, _ = _postprocess_variant_pair(emb_dir_a, emb_dir_b, None, None)
                    elif direction_postprocess not in {"none", "pair_transform"}:
                        raise ValueError("direction_postprocess must be one of: 'match_variant', 'none', 'pair_transform'.")
                    direction_vec = emb_dir_a.mean(axis=0) - emb_dir_b.mean(axis=0)

                for spec in pll_specs:
                    if spec.get("pll_job") is None:
                        continue
                    tpl_str = spec["pll_template_str"]
                    baseline_str = spec.get("pll_baseline_template")
                    baseline_key = baseline_str if baseline_str else None
                    job = spec["pll_job"]
                    article = spec["pll_article"]
                    group_a = spec["group_a"]
                    group_b = spec["group_b"]
                    paired = bool(spec.get("paired", False))
                    baseline_target = spec.get("baseline_target") or "person"

                    texts_a = [_fill(tpl_str, v, job=job, article=article) for v in group_a]
                    texts_b = [_fill(tpl_str, v, job=job, article=article) for v in group_b]
                    base_a = [_fill(baseline_str, v) for v in group_a] if baseline_str else []
                    base_b = [_fill(baseline_str, v) for v in group_b] if baseline_str else []

                    if variant.get("pooling") == "target":
                        targets_a = [job for _ in texts_a]
                        targets_b = [job for _ in texts_b]
                        targets_base_a = [baseline_target for _ in base_a]
                        targets_base_b = [baseline_target for _ in base_b]
                    else:
                        targets_a = targets_b = targets_base_a = targets_base_b = None

                    emb_a = _encode_cached(texts_a, targets_a)
                    emb_b = _encode_cached(texts_b, targets_b)
                    base_emb_a = _encode_cached(base_a, targets_base_a) if baseline_str else None
                    base_emb_b = _encode_cached(base_b, targets_base_b) if baseline_str else None
                    emb_a_proc, emb_b_proc, base_emb_a_proc, base_emb_b_proc, pair_pcs = _postprocess_variant_pair(
                        emb_a, emb_b, base_emb_a, base_emb_b
                    )
                    pair_direction = direction_vec
                    if (
                        pair_direction is not None
                        and direction_postprocess == "pair_transform"
                        and pair_pcs is not None
                        and pair_pcs.size > 0
                    ):
                        pair_direction = pair_direction - (pair_direction @ pair_pcs.T) @ pair_pcs

                    # Guardrail for pair-transform algebra:
                    # for projected vectors x, dot(x, d) ~= dot(x, P d)
                    if (
                        variant_key not in guardrail_logged_for
                        and direction_postprocess == "pair_transform"
                        and direction_vec is not None
                        and pair_pcs is not None
                        and getattr(pair_pcs, "size", 0) > 0
                    ):
                        xa = emb_a_proc[: min(32, emb_a_proc.shape[0])]
                        xb = emb_b_proc[: min(32, emb_b_proc.shape[0])]
                        x = np.concatenate([xa, xb], axis=0)
                        if x.shape[0] > 0:
                            d = direction_vec
                            pd = d - (d @ pair_pcs.T) @ pair_pcs
                            dots_d = x @ d
                            dots_pd = x @ pd
                            diff = dots_d - dots_pd
                            max_abs = float(np.max(np.abs(diff)))
                            mean_abs = float(np.mean(np.abs(diff)))

                            dn = float(np.linalg.norm(d))
                            pdn = float(np.linalg.norm(pd))
                            cos_d_pd = float(d.dot(pd) / (dn * pdn + 1e-12))

                            telemetry.log(
                                {
                                    "type": "direction_guardrail",
                                    "variant": variant,
                                    "variant_key": variant_key,
                                    "direction_postprocess": direction_postprocess,
                                    "sample_n": int(x.shape[0]),
                                    "pcs_k": int(pair_pcs.shape[0]),
                                    "max_abs_dot_diff": max_abs,
                                    "mean_abs_dot_diff": mean_abs,
                                    "d_norm": dn,
                                    "pd_norm": pdn,
                                    "cos_d_pd": cos_d_pd,
                                    "pll_template": spec.get("pll_template"),
                                    "pll_template_str": tpl_str,
                                    "pll_job": job,
                                }
                            )
                            if max_abs > 1e-3:
                                telemetry.log(
                                    {
                                        "type": "warning",
                                        "what": "direction_guardrail_large_diff",
                                        "variant": variant,
                                        "variant_key": variant_key,
                                        "direction_postprocess": direction_postprocess,
                                        "max_abs_dot_diff": max_abs,
                                        "mean_abs_dot_diff": mean_abs,
                                        "sample_n": int(x.shape[0]),
                                        "pcs_k": int(pair_pcs.shape[0]),
                                        "pll_template": spec.get("pll_template"),
                                        "pll_template_str": tpl_str,
                                        "pll_job": job,
                                    }
                                )
                            guardrail_logged_for.add(variant_key)

                    embed_bias = _embed_bias(
                        emb_a_proc,
                        emb_b_proc,
                        base_emb_a_proc,
                        base_emb_b_proc,
                        paired=paired,
                        normalize=False,
                        direction=pair_direction,
                    )
                    if embed_bias["d"] is not None and float(embed_bias["d_norm"] or 0.0) > 1e-12:
                        d_normed = np.asarray(embed_bias["d"], dtype=np.float32) / float(embed_bias["d_norm"])
                        direction_samples.append(d_normed)
                    swap_err = None
                    if embed_bias["d_norm"] and embed_bias["d_norm"] > 1e-12:
                        # Compute swapped delta but keep original direction d
                        if paired:
                            if base_emb_a_proc is not None and base_emb_b_proc is not None:
                                delta_swapped = ((emb_b_proc - base_emb_b_proc) - (emb_a_proc - base_emb_a_proc)).mean(axis=0)
                            else:
                                delta_swapped = (emb_b_proc - emb_a_proc).mean(axis=0)
                        else:
                            if base_emb_a_proc is not None and base_emb_b_proc is not None:
                                delta_swapped = (emb_b_proc - base_emb_b_proc).mean(axis=0) - (
                                    emb_a_proc - base_emb_a_proc
                                ).mean(axis=0)
                            else:
                                delta_swapped = emb_b_proc.mean(axis=0) - emb_a_proc.mean(axis=0)
                        d = embed_bias["d"]
                        bias_swapped = float(delta_swapped.dot(d) / embed_bias["d_norm"])
                        if embed_bias["embed_bias_adj"] is not None:
                            swap_err = abs(bias_swapped + embed_bias["embed_bias_adj"])

                    key = (
                        spec.get("pll_template"),
                        tpl_str,
                        job,
                        article,
                        baseline_key,
                    )
                    pll_vals = pll_bias_map.get(key, {})

                    aligned_record = {
                        "variant": variant,
                        "pll_template": spec.get("pll_template"),
                        "pll_template_str": tpl_str,
                        "pll_job": job,
                        "pll_article": article,
                        "pll_baseline_template": baseline_key,
                        "pll_bias_raw": pll_vals.get("pll_bias_raw"),
                        "pll_bias_adj": pll_vals.get("pll_bias_adj"),
                        "pll_bias": pll_vals.get("pll_bias"),
                        "embed_bias_raw": embed_bias["embed_bias_raw"],
                        "embed_bias_adj": embed_bias["embed_bias_adj"],
                        "embed_dir_norm": embed_bias["d_norm"],
                        "embed_swap_err_adj": swap_err,
                        "paired": paired,
                        "n_names": len(group_a),
                    }
                    f_aligned.write(json.dumps(aligned_record) + "\n")
                    aligned_records.append(aligned_record)

                    # Per-pair aligned rows for statistically valid overlap-coupled pooling.
                    pair_list = spec.get("group_pairs")
                    if pair_list is None and paired:
                        pair_list = list(zip(group_a, group_b))
                    if paired and pair_list:
                        n_pairs = min(len(pair_list), emb_a_proc.shape[0], emb_b_proc.shape[0])
                        if base_emb_a_proc is not None and base_emb_b_proc is not None:
                            n_pairs = min(n_pairs, base_emb_a_proc.shape[0], base_emb_b_proc.shape[0])
                        if n_pairs != len(pair_list):
                            telemetry.log(
                                {
                                    "type": "warning",
                                    "what": "aligned_pairs_length_mismatch",
                                    "variant": variant,
                                    "pll_template": spec.get("pll_template"),
                                    "pll_template_str": tpl_str,
                                    "pll_job": job,
                                    "expected_pairs": len(pair_list),
                                    "used_pairs": n_pairs,
                                }
                            )

                        d = embed_bias["d"]
                        d_norm = float(embed_bias["d_norm"] or 0.0)
                        pair_occurrence: dict[tuple[str, str], int] = {}
                        for pair_idx in range(n_pairs):
                            name_a, name_b = pair_list[pair_idx]
                            pair_key_names = (name_a, name_b)
                            occ = pair_occurrence.get(pair_key_names, 0)
                            pair_occurrence[pair_key_names] = occ + 1
                            pair_id = _pair_id(name_a, name_b, occ)
                            pair_key = (
                                spec.get("pll_template"),
                                tpl_str,
                                job,
                                article,
                                baseline_key,
                                pair_id,
                                name_a,
                                name_b,
                            )
                            pll_pair_vals = pll_pair_bias_map.get(pair_key, {})
                            row_id = pll_pair_vals.get(
                                "row_id",
                                _pair_row_id(
                                    spec.get("pll_template"),
                                    tpl_str,
                                    job,
                                    article,
                                    baseline_key,
                                    pair_id,
                                    name_a,
                                    name_b,
                                ),
                            )

                            if d is not None and d_norm > 1e-12:
                                delta_raw_vec = emb_a_proc[pair_idx] - emb_b_proc[pair_idx]
                                embed_pair_bias_raw = float(delta_raw_vec.dot(d) / d_norm)
                                if base_emb_a_proc is not None and base_emb_b_proc is not None:
                                    delta_adj_vec = (
                                        (emb_a_proc[pair_idx] - base_emb_a_proc[pair_idx])
                                        - (emb_b_proc[pair_idx] - base_emb_b_proc[pair_idx])
                                    )
                                    embed_pair_bias_adj = float(delta_adj_vec.dot(d) / d_norm)
                                else:
                                    embed_pair_bias_adj = None
                            else:
                                embed_pair_bias_raw = None
                                embed_pair_bias_adj = None

                            pair_record = {
                                "variant": variant,
                                "pll_template": spec.get("pll_template"),
                                "pll_template_str": tpl_str,
                                "pll_job": job,
                                "pll_article": article,
                                "pll_baseline_template": baseline_key,
                                "pair_id": pair_id,
                                "group_a": name_a,
                                "group_b": name_b,
                                "row_id": row_id,
                                "pll_bias_raw": pll_pair_vals.get("pll_bias_raw"),
                                "pll_bias_adj": pll_pair_vals.get("pll_bias_adj"),
                                "pll_bias": pll_pair_vals.get("pll_bias"),
                                "embed_bias_raw": embed_pair_bias_raw,
                                "embed_bias_adj": embed_pair_bias_adj,
                                "embed_dir_norm": d_norm if d_norm > 0 else 0.0,
                            }
                            f_aligned_pairs.write(json.dumps(pair_record) + "\n")
                            aligned_pair_records.append(pair_record)

                # Persist a variant-level direction vector for causal intervention runs.
                save_dir_vec = None
                if direction_vec is not None:
                    dn = float(np.linalg.norm(direction_vec))
                    if dn > 1e-12:
                        save_dir_vec = np.asarray(direction_vec, dtype=np.float32) / dn
                if save_dir_vec is None and direction_samples:
                    stacked = np.stack(direction_samples, axis=0)
                    mean_dir = stacked.mean(axis=0)
                    mn = float(np.linalg.norm(mean_dir))
                    if mn > 1e-12:
                        save_dir_vec = (mean_dir / mn).astype(np.float32)

                if save_dir_vec is not None:
                    variant_hash = _stable_hash8_text(variant_key)
                    out_name = f"bias_dir_{variant_hash}.npy"
                    out_path = run_dir / out_name
                    np.save(out_path, save_dir_vec)
                    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
                    info = {
                        "variant": variant,
                        "variant_key": variant_key,
                        "variant_hash": variant_hash,
                        "path": out_name,
                        "dim": int(save_dir_vec.shape[0]),
                        "sha256": sha256,
                        "direction_source": direction_source,
                        "direction_postprocess": direction_postprocess,
                    }
                    bias_dir_artifacts[variant_key] = info
                    telemetry.log({"type": "bias_dir", **info})

        # Correlation between PLL and aligned embedding bias
        if aligned_records:
            by_variant = {}
            for rec in aligned_records:
                vkey = json.dumps(rec["variant"], sort_keys=True)
                by_variant.setdefault(vkey, []).append(rec)

            corr_out = {}
            corr_cfg = config.get("metrics", {}).get("embedding_bias", {})
            corr_filter_template = corr_cfg.get("corr_filter_template", "gender_names_jobs")
            corr_bootstrap_samples = int(corr_cfg.get("corr_bootstrap_samples", 1000))
            corr_bootstrap_seed = int(corr_cfg.get("corr_bootstrap_seed", 123))
            rng = np.random.RandomState(corr_bootstrap_seed)

            def _bootstrap_spearman(pairs: list[tuple[float, float]], n_samples: int) -> list[float]:
                if len(pairs) < 3:
                    return []
                idx = np.arange(len(pairs))
                vals = []
                for _ in range(n_samples):
                    sample_idx = rng.choice(idx, size=len(pairs), replace=True)
                    xs = [pairs[i][0] for i in sample_idx]
                    ys = [pairs[i][1] for i in sample_idx]
                    vals.append(_pearson(_ranks(xs), _ranks(ys)))
                return vals

            for vkey, recs in by_variant.items():
                filtered = [
                    r
                    for r in recs
                    if r.get("pll_template") == corr_filter_template
                    and r.get("pll_bias") is not None
                    and r.get("embed_bias_adj") is not None
                ]

                # Build per-template job maps for bootstrap resampling
                by_tpl_job: dict[str, dict[str, tuple[float, float]]] = {}
                for r in filtered:
                    tpl = r["pll_template_str"]
                    job = r.get("pll_job")
                    if job is None:
                        continue
                    jid = f"{job}|{r.get('pll_article')}"
                    by_tpl_job.setdefault(tpl, {})[jid] = (r["pll_bias"], r["embed_bias_adj"])

                # Global correlation across all templates (filtered)
                global_pairs = []
                for tpl_map in by_tpl_job.values():
                    global_pairs.extend(list(tpl_map.values()))
                if global_pairs:
                    x = [p[0] for p in global_pairs]
                    y = [p[1] for p in global_pairs]
                    corr = _pearson(_ranks(x), _ranks(y))
                else:
                    corr = 0.0

                # Bootstrap CI by resampling jobs within each template
                boot_vals = []
                if corr_bootstrap_samples > 0 and by_tpl_job:
                    for _ in range(corr_bootstrap_samples):
                        sampled_pairs = []
                        for tpl_map in by_tpl_job.values():
                            jobs = list(tpl_map.keys())
                            if not jobs:
                                continue
                            sample_jobs = rng.choice(jobs, size=len(jobs), replace=True)
                            sampled_pairs.extend([tpl_map[j] for j in sample_jobs])
                        if len(sampled_pairs) >= 3:
                            xs = [p[0] for p in sampled_pairs]
                            ys = [p[1] for p in sampled_pairs]
                            val = _pearson(_ranks(xs), _ranks(ys))
                            if np.isfinite(val):
                                boot_vals.append(val)
                if boot_vals:
                    ci_global = (_percentile(boot_vals, 2.5), _percentile(boot_vals, 97.5))
                else:
                    ci_global = None

                # Per-template correlation + bootstrap
                tpl_corr = {}
                for tpl, tpl_map in by_tpl_job.items():
                    tpl_pairs = list(tpl_map.values())
                    x = [p[0] for p in tpl_pairs]
                    y = [p[1] for p in tpl_pairs]
                    tpl_corr_val = _pearson(_ranks(x), _ranks(y)) if len(tpl_pairs) >= 3 else None
                    if tpl_corr_val is not None and not np.isfinite(tpl_corr_val):
                        tpl_corr_val = None
                    tpl_boot = (
                        _bootstrap_spearman(tpl_pairs, corr_bootstrap_samples)
                        if (tpl_corr_val is not None and corr_bootstrap_samples > 0)
                        else []
                    )
                    tpl_ci = (_percentile(tpl_boot, 2.5), _percentile(tpl_boot, 97.5)) if tpl_boot else None
                    tpl_corr[tpl] = {"spearman_adj": tpl_corr_val, "n": len(tpl_pairs), "ci95": tpl_ci}
                corr_out[vkey] = {
                    "filter_template": corr_filter_template,
                    "global_spearman_adj": corr,
                    "global_n": len(global_pairs),
                    "global_ci95": ci_global,
                    "bootstrap_samples": corr_bootstrap_samples,
                    "bootstrap_seed": corr_bootstrap_seed,
                    "by_template": tpl_corr,
                }

            corr_path = run_dir / "results_pll_embed_corr.json"
            corr_path.write_text(json.dumps(corr_out, indent=2))

            # Optional: attach a compact embedding-corr summary to PLL summary
            pll_summary_path = run_dir / "results_pll_summary.json"
            if pll_summary_path.exists():
                try:
                    pll_summary_disk = json.loads(pll_summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(
                        "[warn] could not read results_pll_summary.json; "
                        f"skipping corr summary attach: {exc}"
                    )
                else:
                    pll_summary_disk["embedding_corr_summary"] = {
                        vkey: {
                            "filter_template": vals.get("filter_template"),
                            "global_spearman_adj": vals.get("global_spearman_adj"),
                            "global_ci95": vals.get("global_ci95"),
                            "global_n": vals.get("global_n"),
                            "bootstrap_samples": vals.get("bootstrap_samples"),
                            "bootstrap_seed": vals.get("bootstrap_seed"),
                        }
                        for vkey, vals in corr_out.items()
                    }
                    pll_summary_path.write_text(json.dumps(pll_summary_disk, indent=2), encoding="utf-8")

    # Holm correction across SEAT tests per variant
    if bias_dir_artifacts:
        (run_dir / "bias_directions.json").write_text(json.dumps(bias_dir_artifacts, indent=2), encoding="utf-8")

    if embedding_records:
        from collections import defaultdict

        def _holm_adjust(pvals: list[float]) -> list[float]:
            m = len(pvals)
            order = sorted(range(m), key=lambda i: pvals[i])
            adj = [0.0] * m
            prev = 0.0
            for rank, idx in enumerate(order):
                factor = m - rank
                val = min(1.0, pvals[idx] * factor)
                prev = max(prev, val)
                adj[idx] = prev
            return adj

        by_variant = defaultdict(list)
        for rec in embedding_records:
            key = json.dumps(rec["variant"], sort_keys=True)
            by_variant[key].append(rec)

        holm_summary = []
        for key, recs in by_variant.items():
            pvals = [float(r["embedding_bias"].get("p_two_sided", 1.0)) for r in recs]
            adj = _holm_adjust(pvals)
            for r, adj_p in zip(recs, adj):
                holm_summary.append(
                    {
                        "variant": r["variant"],
                        "seat_test": r.get("seat_test", "unnamed"),
                        "p_two_sided": r["embedding_bias"].get("p_two_sided"),
                        "p_holm": adj_p,
                    }
                )

        holm_path = run_dir / "embedding_holm.json"
        holm_path.write_text(json.dumps(holm_summary, indent=2))

    telemetry.log({"type": "run_end", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.log({"type": "run_complete", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.close()

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
