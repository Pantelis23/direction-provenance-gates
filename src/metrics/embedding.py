from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / denom


@dataclass
class EmbeddingVariant:
    layer: Any
    pooling: str
    normalize: bool


class EmbeddingExtractor:
    def __init__(self, model_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.device = self._resolve_device(model_cfg.get("device", "auto"))
        self.backend = model_cfg.get("backend", "hf")
        self.model_name = model_cfg.get("name", "gpt2")

        if self.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = None
        else:
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModel.from_pretrained(self.model_name)
            if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.to(self.device)
            self.model.eval()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_layer_index(self, num_layers: int, layer: Any) -> int:
        if isinstance(layer, int):
            if layer < 0:
                return max(0, num_layers + layer)
            return min(num_layers, layer)
        if isinstance(layer, str):
            if layer == "early":
                return 1
            if layer == "mid":
                return max(1, num_layers // 2)
            if layer == "late":
                return num_layers
            if layer == "embedding":
                return 0
        return num_layers

    def encode(
        self,
        sentences: List[str],
        variant: EmbeddingVariant,
        targets: Optional[List[str]] = None,
    ) -> np.ndarray:
        if self.backend == "sentence_transformer":
            if variant.pooling == "target":
                raise ValueError("pooling='target' is not supported for sentence_transformer backend.")
            emb = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=variant.normalize)
            return emb

        import torch  # type: ignore

        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                output_attentions=(variant.pooling == "attn-weighted"),
            )

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1
        layer_idx = self._resolve_layer_index(num_layers, variant.layer)
        layer_hidden = hidden_states[layer_idx]

        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(layer_hidden.shape[:2], device=layer_hidden.device)

        if variant.pooling == "cls":
            pooled = layer_hidden[:, 0, :]
        elif variant.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            pooled = layer_hidden[torch.arange(layer_hidden.size(0)), lengths]
        elif variant.pooling == "target":
            if targets is None:
                raise ValueError("pooling='target' requires targets per sentence.")
            pooled_rows = []
            input_ids = encoded["input_ids"].tolist()
            for i, target in enumerate(targets):
                ids = input_ids[i]
                max_len = int(attention_mask[i].sum().item())
                ids = ids[:max_len]
                target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                start = _find_subsequence(ids, target_ids)
                if start is None and not target.startswith(" "):
                    target_ids = self.tokenizer.encode(" " + target, add_special_tokens=False)
                    start = _find_subsequence(ids, target_ids)
                if start is None:
                    raise ValueError(f"Target '{target}' not found in sentence: '{sentences[i]}'")
                end = start + len(target_ids)
                span = layer_hidden[i, start:end, :]
                pooled_rows.append(span.mean(dim=0))
            pooled = torch.stack(pooled_rows, dim=0)
        elif variant.pooling == "attn-weighted":
            if outputs.attentions is None:
                raise ValueError("Attention weights were not returned by the model.")
            # Use last layer attention: [batch, heads, seq, seq]
            attn = outputs.attentions[-1]
            # Take attention from CLS token to all tokens and average heads
            cls_attn = attn[:, :, 0, :]
            weights = cls_attn.mean(dim=1)
            weights = weights * attention_mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
            pooled = (layer_hidden * weights.unsqueeze(-1)).sum(dim=1)
        else:
            # mean pooling (default)
            masked = layer_hidden * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-12)

        emb = pooled.detach().cpu().numpy()
        return emb


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle)
    for i in range(last + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _postprocess_embeddings(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    variant: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = bool(variant.get("center", False))
    whiten_k = int(variant.get("whiten_k", 0) or 0)
    normalize = bool(variant.get("normalize", True))

    all_emb = np.concatenate([X, Y, A, B], axis=0)

    if center or whiten_k > 0:
        mean = all_emb.mean(axis=0, keepdims=True)
        all_emb = all_emb - mean

    if whiten_k > 0:
        # Remove top-k principal components to reduce anisotropy.
        # SVD on [N, D] with N small (sentence count).
        _, _, vt = np.linalg.svd(all_emb, full_matrices=False)
        k = min(whiten_k, vt.shape[0])
        if k > 0:
            pcs = vt[:k]
            all_emb = all_emb - (all_emb @ pcs.T) @ pcs

    if normalize:
        all_emb = _l2_normalize(all_emb)

    nX, nY, nA = X.shape[0], Y.shape[0], A.shape[0]
    Xp = all_emb[:nX]
    Yp = all_emb[nX : nX + nY]
    Ap = all_emb[nX + nY : nX + nY + nA]
    Bp = all_emb[nX + nY + nA :]
    return Xp, Yp, Ap, Bp


def _top_singular_values(matrix: np.ndarray, k: int = 10) -> List[float]:
    if matrix.shape[0] == 0:
        return []
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, s.shape[0])
    return [float(v) for v in s[:k]]


def _conicity(matrix: np.ndarray) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    normed = _l2_normalize(matrix)
    mean_vec = normed.mean(axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
    return float(np.mean(normed @ mean_vec))


def _extract_texts_and_targets(
    items: List[Any],
) -> Tuple[List[str], Optional[List[str]]]:
    texts: List[str] = []
    targets: List[Optional[str]] = []
    has_target = False
    for item in items:
        if isinstance(item, dict):
            text = item.get("text")
            target = item.get("target")
            if text is None or target is None:
                raise ValueError("Sentence dict must include 'text' and 'target'.")
            has_target = True
            texts.append(str(text))
            targets.append(str(target))
        else:
            texts.append(str(item))
            targets.append(None)
    if has_target:
        if any(t is None for t in targets):
            raise ValueError("All sentences must include target when using target pooling.")
        return texts, [t for t in targets if t is not None]
    return texts, None


def _expand_items(items: Any) -> List[Any]:
    if isinstance(items, dict) and "templates" in items and "targets" in items:
        templates = items.get("templates") or []
        targets = items.get("targets") or []
        expanded: List[Dict[str, str]] = []
        for t in targets:
            for template in templates:
                text = template.format(t=t, target=t)
                expanded.append({"text": text, "target": str(t)})
        return expanded
    return list(items)


def _association(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    # s(w, A, B) = mean cosine(w, a) - mean cosine(w, b)
    w = w / (np.linalg.norm(w) + 1e-12)
    A = _l2_normalize(A)
    B = _l2_normalize(B)
    return float(np.dot(A, w).mean() - np.dot(B, w).mean())


def _weat_statistic(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sum([_association(x, A, B) for x in X]) - np.sum([_association(y, A, B) for y in Y]))


def _weat_effect_size(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    X_assoc = np.array([_association(x, A, B) for x in X])
    Y_assoc = np.array([_association(y, A, B) for y in Y])
    mean_diff = X_assoc.mean() - Y_assoc.mean()
    std_dev = np.std(np.concatenate([X_assoc, Y_assoc]), ddof=1) + 1e-12
    return float(mean_diff / std_dev)


def _weat_p_value(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    num_samples: int = 10000,
    max_exact: int = 20000,
) -> float:
    # One-sided permutation test as in WEAT/SEAT.
    XY = np.concatenate([X, Y], axis=0)
    n = XY.shape[0]
    size_X = X.shape[0]

    observed = _weat_statistic(X, Y, A, B)

    total_combos = math.comb(n, size_X)
    if total_combos <= max_exact:
        # Exact test
        count = 0
        total = 0
        indices = list(range(n))
        for combo in _combinations(indices, size_X):
            X_idx = np.array(combo)
            Y_idx = np.array([i for i in indices if i not in combo])
            stat = _weat_statistic(XY[X_idx], XY[Y_idx], A, B)
            if stat >= observed:
                count += 1
            total += 1
        return count / max(total, 1)

    # Approximate test by sampling permutations
    count = 0
    for _ in range(num_samples):
        perm = np.random.permutation(n)
        X_idx = perm[:size_X]
        Y_idx = perm[size_X:]
        stat = _weat_statistic(XY[X_idx], XY[Y_idx], A, B)
        if stat >= observed:
            count += 1
    return count / max(num_samples, 1)


def _weat_permutation_stats(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    num_samples: int = 10000,
    max_exact: int = 20000,
) -> Tuple[List[float], bool]:
    XY = np.concatenate([X, Y], axis=0)
    n = XY.shape[0]
    size_X = X.shape[0]
    total_combos = math.comb(n, size_X)

    if total_combos <= max_exact:
        stats = []
        indices = list(range(n))
        for combo in _combinations(indices, size_X):
            X_idx = np.array(combo)
            Y_idx = np.array([i for i in indices if i not in combo])
            stats.append(_weat_statistic(XY[X_idx], XY[Y_idx], A, B))
        return stats, False

    stats = []
    for _ in range(num_samples):
        perm = np.random.permutation(n)
        X_idx = perm[:size_X]
        Y_idx = perm[size_X:]
        stats.append(_weat_statistic(XY[X_idx], XY[Y_idx], A, B))
    return stats, True


def _combinations(items: List[int], k: int) -> Iterable[Tuple[int, ...]]:
    # Simple combinations generator to avoid importing itertools in tight loop for clarity.
    def _helper(start: int, path: List[int]):
        if len(path) == k:
            yield tuple(path)
            return
        for i in range(start, len(items)):
            path.append(items[i])
            yield from _helper(i + 1, path)
            path.pop()

    return _helper(0, [])


def compute_seat_weat(
    extractor: EmbeddingExtractor,
    variant: Dict[str, Any],
    test: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute SEAT/WEAT effect size and p-value for one test.

    Expected test format:
    {
      "name": "optional",
      "X": [sentences...],
      "Y": [sentences...],
      "A": [sentences...],
      "B": [sentences...]
    }
    """
    variant_obj = EmbeddingVariant(
        layer=variant.get("layer", "late"),
        pooling=variant.get("pooling", "mean"),
        normalize=bool(variant.get("normalize", True)),
    )

    X = test.get("X")
    Y = test.get("Y")
    A = test.get("A")
    B = test.get("B")

    if not (X and Y and A and B):
        raise ValueError("SEAT/WEAT test requires X, Y, A, B sentence sets.")

    X_items = _expand_items(X)
    Y_items = _expand_items(Y)
    A_items = _expand_items(A)
    B_items = _expand_items(B)

    X_texts, X_targets = _extract_texts_and_targets(X_items)
    Y_texts, Y_targets = _extract_texts_and_targets(Y_items)
    A_texts, A_targets = _extract_texts_and_targets(A_items)
    B_texts, B_targets = _extract_texts_and_targets(B_items)

    X_emb_raw = extractor.encode(X_texts, variant_obj, targets=X_targets if variant_obj.pooling == "target" else None)
    Y_emb_raw = extractor.encode(Y_texts, variant_obj, targets=Y_targets if variant_obj.pooling == "target" else None)
    A_emb_raw = extractor.encode(A_texts, variant_obj, targets=A_targets if variant_obj.pooling == "target" else None)
    B_emb_raw = extractor.encode(B_texts, variant_obj, targets=B_targets if variant_obj.pooling == "target" else None)

    sv_top10_raw = _top_singular_values(np.concatenate([X_emb_raw, Y_emb_raw, A_emb_raw, B_emb_raw], axis=0), k=10)
    conicity_raw = _conicity(np.concatenate([X_emb_raw, Y_emb_raw, A_emb_raw, B_emb_raw], axis=0))

    X_emb, Y_emb, A_emb, B_emb = _postprocess_embeddings(X_emb_raw, Y_emb_raw, A_emb_raw, B_emb_raw, variant)
    sv_top10_post = _top_singular_values(np.concatenate([X_emb, Y_emb, A_emb, B_emb], axis=0), k=10)
    conicity_post = _conicity(np.concatenate([X_emb, Y_emb, A_emb, B_emb], axis=0))

    effect = _weat_effect_size(X_emb, Y_emb, A_emb, B_emb)
    observed_stat = _weat_statistic(X_emb, Y_emb, A_emb, B_emb)

    perm_samples = int(config.get("metrics", {}).get("embedding_bias", {}).get("perm_samples", 10000))
    max_exact = int(config.get("metrics", {}).get("embedding_bias", {}).get("max_exact", 20000))
    perm_seeds = config.get("metrics", {}).get("embedding_bias", {}).get("perm_seeds") or []
    perm_stats = None
    approx = True
    p_one_sided_pos = 0.0
    p_one_sided_neg = 0.0
    p_two_sided = 0.0
    p_two_sided_seed = []
    p_one_sided_pos_seed = []
    p_one_sided_neg_seed = []

    if perm_seeds:
        state = np.random.get_state()
        for seed in perm_seeds:
            np.random.seed(int(seed))
            stats, approx = _weat_permutation_stats(
                X_emb,
                Y_emb,
                A_emb,
                B_emb,
                num_samples=perm_samples,
                max_exact=max_exact,
            )
            if perm_stats is None:
                perm_stats = stats
            stats_arr = np.array(stats, dtype=np.float64)
            p_pos = float(np.mean(stats_arr >= observed_stat))
            p_neg = float(np.mean(stats_arr <= observed_stat))
            p_one_sided_pos_seed.append(p_pos)
            p_one_sided_neg_seed.append(p_neg)
            p_val = float(np.mean(np.abs(stats_arr) >= abs(observed_stat)))
            p_two_sided_seed.append(p_val)
        np.random.set_state(state)
        p_two_sided = float(np.mean(p_two_sided_seed))
        p_one_sided_pos = float(np.mean(p_one_sided_pos_seed)) if p_one_sided_pos_seed else 0.0
        p_one_sided_neg = float(np.mean(p_one_sided_neg_seed)) if p_one_sided_neg_seed else 0.0
    else:
        perm_stats, approx = _weat_permutation_stats(
            X_emb,
            Y_emb,
            A_emb,
            B_emb,
            num_samples=perm_samples,
            max_exact=max_exact,
        )
        perm_stats_arr = np.array(perm_stats, dtype=np.float64)
        p_one_sided_pos = float(np.mean(perm_stats_arr >= observed_stat))
        p_one_sided_neg = float(np.mean(perm_stats_arr <= observed_stat))
        p_two_sided = float(np.mean(np.abs(perm_stats_arr) >= abs(observed_stat)))

    # Association diagnostics
    X_assoc = np.array([_association(x, A_emb, B_emb) for x in X_emb])
    Y_assoc = np.array([_association(y, A_emb, B_emb) for y in Y_emb])
    assoc_std = float(np.std(np.concatenate([X_assoc, Y_assoc]), ddof=1))

    # Anisotropy probe
    all_emb = np.concatenate([X_emb, Y_emb, A_emb, B_emb], axis=0)
    all_norm = _l2_normalize(all_emb)
    n = all_norm.shape[0]
    sample_pairs = int(config.get("metrics", {}).get("embedding_bias", {}).get("anisotropy_samples", 200))
    sample_pairs = max(1, sample_pairs)
    cos_vals = []
    near_dup = 0
    for _ in range(sample_pairs):
        i, j = np.random.randint(0, n, size=2)
        if i == j:
            j = (j + 1) % n
        cos = float(np.dot(all_norm[i], all_norm[j]))
        cos_vals.append(cos)
        if cos > 0.999:
            near_dup += 1
    mean_cos = float(np.mean(cos_vals))
    frac_near_dup = float(near_dup / sample_pairs)

    # Bootstrap CI for effect size
    boot_samples = int(config.get("metrics", {}).get("embedding_bias", {}).get("bootstrap_samples", 0))
    ci_low = None
    ci_high = None
    if boot_samples > 0:
        rng_state = np.random.get_state()
        boot = []
        for _ in range(boot_samples):
            Xb = X_emb[np.random.randint(0, X_emb.shape[0], size=X_emb.shape[0])]
            Yb = Y_emb[np.random.randint(0, Y_emb.shape[0], size=Y_emb.shape[0])]
            Ab = A_emb[np.random.randint(0, A_emb.shape[0], size=A_emb.shape[0])]
            Bb = B_emb[np.random.randint(0, B_emb.shape[0], size=B_emb.shape[0])]
            boot.append(_weat_effect_size(Xb, Yb, Ab, Bb))
        np.random.set_state(rng_state)
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))

    perm_stats_arr = np.array(perm_stats, dtype=np.float64) if perm_stats is not None else np.array([])

    return {
        "effect_size": effect,
        "test_statistic": observed_stat,
        "p_value": p_two_sided,
        "p_one_sided_pos": p_one_sided_pos,
        "p_one_sided_neg": p_one_sided_neg,
        "p_two_sided": p_two_sided,
        "p_two_sided_seed": p_two_sided_seed,
        "p_two_sided_seed_mean": float(np.mean(p_two_sided_seed)) if p_two_sided_seed else None,
        "p_two_sided_seed_std": float(np.std(p_two_sided_seed, ddof=1)) if len(p_two_sided_seed) > 1 else None,
        "perm_mean": float(np.mean(perm_stats_arr)) if perm_stats_arr.size else None,
        "perm_min": float(np.min(perm_stats_arr)) if perm_stats_arr.size else None,
        "perm_max": float(np.max(perm_stats_arr)) if perm_stats_arr.size else None,
        "perm_approx": bool(approx),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "assoc_std": assoc_std,
        "mean_assoc_X": float(X_assoc.mean()),
        "mean_assoc_Y": float(Y_assoc.mean()),
        "anisotropy_mean_cos": mean_cos,
        "anisotropy_frac_near_dup": frac_near_dup,
        "sv_top10_raw": sv_top10_raw,
        "sv_top10_post": sv_top10_post,
        "conicity_raw": conicity_raw,
        "conicity_post": conicity_post,
    }
