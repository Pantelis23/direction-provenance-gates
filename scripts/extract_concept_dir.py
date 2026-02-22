#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.pll import PLLScorer
from src.telemetry.logger import TelemetryLogger
from src.utils.config import load_config
from src.utils.seed import set_seed


def _load_texts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    texts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text") or rec.get("prompt")
            if text is None:
                raise ValueError(f"JSONL line missing text/prompt field in {path}")
            texts.append(str(text))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                texts.append(t)
    if not texts:
        raise ValueError(f"No prompts loaded from {path}")
    return texts


def _resolve_layer_index(layer_value: str | int, n_layers: int) -> int:
    if isinstance(layer_value, int):
        idx = layer_value
    else:
        lv = str(layer_value).strip().lower()
        if lv.isdigit() or (lv.startswith("-") and lv[1:].isdigit()):
            idx = int(lv)
        elif lv == "early":
            idx = 1
        elif lv == "mid":
            idx = n_layers // 2
        elif lv == "late":
            idx = n_layers - 1
        else:
            raise ValueError(f"Unsupported layer value: {layer_value}")
    if idx < 0:
        idx = n_layers + idx
    return max(0, min(int(idx), n_layers - 1))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(a.shape[0], dtype=float)
    return ranks


def _binary_auc(y01: np.ndarray, score: np.ndarray) -> float:
    pos = int(np.sum(y01 == 1))
    neg = int(np.sum(y01 == 0))
    if pos == 0 or neg == 0:
        return 0.5
    ranks = _rank(score)
    rank_sum_pos = float(np.sum(ranks[y01 == 1]))
    # Mann-Whitney U -> AUC
    auc = (rank_sum_pos - pos * (pos - 1) / 2.0) / float(pos * neg)
    return float(max(0.0, min(1.0, auc)))


def _accuracy(y01: np.ndarray, score: np.ndarray) -> float:
    pred = (score >= 0.0).astype(np.int64)
    return float(np.mean(pred == y01))


def _split_mask(texts: list[str], holdout_frac: float, holdout_seed: int) -> np.ndarray:
    out = np.zeros(len(texts), dtype=bool)
    cutoff = int(float(holdout_frac) * 10000.0)
    for i, text in enumerate(texts):
        h = int(hashlib.blake2s((text + f"|{holdout_seed}").encode("utf-8"), digest_size=8).hexdigest(), 16)
        out[i] = (h % 10000) < cutoff
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--concept-name", required=True)
    ap.add_argument("--positive-file", required=True)
    ap.add_argument("--negative-file", required=True)
    ap.add_argument("--layer", default="late")
    ap.add_argument("--pool", choices=["mean", "last", "cls"], default="mean")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--run-name", default="extract_concept_dir")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_cfg = cfg.get("run", {})
    set_seed(int(run_cfg.get("seed", 42)))

    out_root = Path(run_cfg.get("output_dir", "runs"))
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.run_name}_{args.concept_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    telemetry = TelemetryLogger(run_dir / "telemetry.jsonl")
    telemetry.log({"type": "run_start", "run_id": run_id})
    telemetry.log({"type": "config", "args": vars(args), "config": cfg})

    pos_texts = _load_texts(Path(args.positive_file))
    neg_texts = _load_texts(Path(args.negative_file))
    texts = pos_texts + neg_texts
    y01 = np.asarray([1] * len(pos_texts) + [0] * len(neg_texts), dtype=np.int64)
    ypm = np.asarray([1.0] * len(pos_texts) + [-1.0] * len(neg_texts), dtype=np.float64)

    test_mask = _split_mask(texts, holdout_frac=float(args.holdout_frac), holdout_seed=int(args.holdout_seed))
    train_mask = ~test_mask
    if int(np.sum(train_mask)) < 4 or int(np.sum(test_mask)) < 4:
        raise ValueError(
            f"Insufficient split sizes: train={int(np.sum(train_mask))}, test={int(np.sum(test_mask))}. "
            "Increase prompts or adjust --holdout-frac."
        )

    scorer = PLLScorer(cfg.get("model", {}))
    n_layers = len(scorer._resolve_layers())
    layer_idx = _resolve_layer_index(args.layer, n_layers=n_layers)

    hidden_cache: dict[str, np.ndarray] = {}

    def _hidden(text: str) -> np.ndarray:
        v = hidden_cache.get(text)
        if v is not None:
            return v
        vec = scorer.extract_hidden(text=text, layer=layer_idx, pool=args.pool)
        hidden_cache[text] = vec
        return vec

    X = np.stack([_hidden(t) for t in texts], axis=0).astype(np.float64)
    x_mean = X[train_mask].mean(axis=0, keepdims=True)
    Xc = X - x_mean
    y_train = ypm[train_mask]
    y_mean = float(np.mean(y_train))
    yc_train = y_train - y_mean

    lam = float(args.ridge_lambda)
    xtx = Xc[train_mask].T @ Xc[train_mask]
    rhs = Xc[train_mask].T @ yc_train
    w = np.linalg.solve(xtx + lam * np.eye(X.shape[1], dtype=np.float64), rhs)
    w = np.asarray(w, dtype=np.float32)
    w_norm = float(np.linalg.norm(w))
    if w_norm <= 1e-12:
        raise ValueError("Learned concept direction has near-zero norm.")
    d_hat = w / w_norm

    score = (Xc @ w).astype(np.float64)
    train_auc = _binary_auc(y01[train_mask], score[train_mask])
    test_auc = _binary_auc(y01[test_mask], score[test_mask])
    train_acc = _accuracy(y01[train_mask], score[train_mask])
    test_acc = _accuracy(y01[test_mask], score[test_mask])

    direction_path = run_dir / "concept_dir.npy"
    np.save(direction_path, d_hat.astype(np.float32))
    direction_sha256 = hashlib.sha256(np.asarray(d_hat, dtype=np.float32).tobytes()).hexdigest()

    summary = {
        "run_id": run_id,
        "concept_name": args.concept_name,
        "layer_idx": int(layer_idx),
        "pool": args.pool,
        "ridge_lambda": float(lam),
        "n_total": int(X.shape[0]),
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "positive_count": int(len(pos_texts)),
        "negative_count": int(len(neg_texts)),
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "direction_norm_pre_normalize": float(w_norm),
        "direction_dim": int(d_hat.shape[0]),
        "direction_dtype": str(np.asarray(d_hat).dtype),
        "direction_shape": list(np.asarray(d_hat).shape),
        "direction_path": str(direction_path.name),
        "direction_sha256": direction_sha256,
    }
    (run_dir / "concept_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    telemetry.log({"type": "concept_summary", **summary})
    telemetry.log({"type": "run_end", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.log({"type": "run_complete", "ok": True, "run_id": run_id, "run_dir": str(run_dir.name)})
    telemetry.close()
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

