from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PLLScore:
    sum_logprob: float
    avg_logprob: float
    token_count: int
    token_ids: Optional[list[int]] = None
    token_logprobs: Optional[list[float]] = None
    pooled_hidden: Optional[np.ndarray] = None


@dataclass
class PLLIntervention:
    layer: int
    direction: Any
    mode: str = "project_out"  # project_out | flip | add
    alpha: float = 1.0
    token_mask: Any | None = None  # optional [T] mask; True positions are edited


class PLLScorer:
    def __init__(self, model_cfg: Dict[str, Any]):
        self.model_name = model_cfg.get("name", "gpt2")
        self.device = _resolve_device(model_cfg.get("device", "auto"))
        self.backend = model_cfg.get("backend", "hf")

        if self.backend != "hf":
            raise ValueError(f"PLLScorer only supports backend='hf' for now, got {self.backend}")

        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()

        self._cache: Dict[str, PLLScore] = {}
        self._hidden_cache: Dict[tuple[str, int, str], np.ndarray] = {}
        self._hidden_cache_order = deque()
        self._hidden_cache_max = int(model_cfg.get("hidden_cache_max", 50000))
        self._last_hidden_layout: Dict[str, Any] | None = None

    def _resolve_layers(self):
        # GPT-2 / GPT-J style
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        # LLaMA style
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # GPT-NeoX style
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        raise ValueError("Unsupported model architecture for residual intervention.")

    @staticmethod
    def _select_hidden_layer(hidden_states, layer_idx: int, n_layers: int):
        hs_len = len(hidden_states)
        if hs_len == n_layers + 1:
            selected_index = layer_idx + 1
            layout_mode = "n_plus_1"
            return hidden_states[selected_index], hs_len, layout_mode, selected_index
        if hs_len == n_layers:
            selected_index = layer_idx
            layout_mode = "n"
            return hidden_states[selected_index], hs_len, layout_mode, selected_index
        raise ValueError(
            f"Unexpected hidden_states length {hs_len}; expected {n_layers} or {n_layers + 1}."
        )

    def _hidden_cache_get(self, key: tuple[str, int, str]) -> Optional[np.ndarray]:
        if self._hidden_cache_max <= 0:
            return None
        return self._hidden_cache.get(key)

    def _hidden_cache_put(self, key: tuple[str, int, str], value: np.ndarray) -> None:
        if self._hidden_cache_max <= 0:
            return
        if key not in self._hidden_cache:
            self._hidden_cache_order.append(key)
        self._hidden_cache[key] = value
        while len(self._hidden_cache_order) > self._hidden_cache_max:
            old = self._hidden_cache_order.popleft()
            self._hidden_cache.pop(old, None)

    def _register_intervention_hook(self, intervention: PLLIntervention, seq_len: int):
        import torch  # type: ignore

        layers = self._resolve_layers()
        n_layers = len(layers)
        layer_idx = intervention.layer
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Invalid intervention layer index {intervention.layer}; model has {n_layers} layers.")

        mode = str(intervention.mode).lower()
        if mode not in {"project_out", "flip", "add"}:
            raise ValueError("intervention.mode must be one of: project_out, flip, add")

        # Keep intervention direction in fp32 for stable projection math; cast back to hidden dtype at return.
        d = torch.as_tensor(intervention.direction, device=self.device, dtype=torch.float32)
        d_norm = torch.linalg.norm(d).item()
        if d_norm <= 0:
            raise ValueError("intervention.direction has zero norm.")
        d_hat = d / d_norm
        alpha = float(intervention.alpha)
        token_mask = None
        if intervention.token_mask is not None:
            token_mask = torch.as_tensor(intervention.token_mask, device=self.device)
            if token_mask.ndim == 2 and token_mask.shape[0] == 1:
                token_mask = token_mask.squeeze(0)
            if token_mask.ndim != 1:
                raise ValueError(f"intervention.token_mask must be 1D [T], got shape={tuple(token_mask.shape)}")
            if int(token_mask.shape[0]) != int(seq_len):
                raise ValueError(
                    f"intervention.token_mask length {token_mask.shape[0]} does not match sequence length {seq_len}."
                )
            token_mask = token_mask.bool().view(1, seq_len, 1)

        def _apply(h):
            # h: [B, T, D]
            if h.shape[-1] != d_hat.shape[0]:
                raise ValueError(
                    f"Intervention direction dim {d_hat.shape[0]} does not match hidden dim {h.shape[-1]}."
                )
            h32 = h.float()
            if mode == "add":
                edited32 = h32 + alpha * d_hat.view(1, 1, -1)
            else:
                proj = torch.matmul(h32, d_hat).unsqueeze(-1)
                coeff = alpha if mode == "project_out" else (2.0 * alpha)
                edited32 = h32 - coeff * proj * d_hat.view(1, 1, -1)
            if token_mask is not None:
                m = token_mask.to(h32.dtype)
                out32 = h32 + m * (edited32 - h32)
            else:
                out32 = edited32
            return out32.to(h.dtype)

        def _hook(_module, _inp, out):
            if isinstance(out, tuple):
                if not out:
                    return out
                first = out[0]
                if first is None:
                    return out
                edited = _apply(first)
                return (edited, *out[1:])
            return _apply(out)

        return layers[layer_idx].register_forward_hook(_hook)

    def token_mask_for_substring(self, text: str, substring: str) -> np.ndarray:
        """
        Return a boolean token mask [T] selecting tokens overlapping substring's first occurrence.
        Falls back to token-id subsequence matching if offset mappings are unavailable.
        """
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                return_offsets_mapping=True,
            )
        except Exception:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
        input_ids = encoded["input_ids"].squeeze(0)
        seq_len = int(input_ids.shape[0])
        mask = np.zeros(seq_len, dtype=np.bool_)

        start = text.find(substring)
        if start >= 0:
            end = start + len(substring)
            offsets = encoded.get("offset_mapping")
            if offsets is not None:
                off = offsets.squeeze(0).tolist()
                for i, (s, e) in enumerate(off):
                    if e <= s:
                        continue
                    if not (e <= start or s >= end):
                        mask[i] = True
                if mask.any():
                    return mask

        # Fallback: token subsequence match against the actual encoded sequence ids.
        enc_ids = encoded.get("input_ids")
        if enc_ids is None:
            return mask
        full_ids = enc_ids.squeeze(0).tolist()
        sub_ids = self.tokenizer(substring, add_special_tokens=False, truncation=True)["input_ids"]
        if sub_ids:
            for i in range(0, max(0, len(full_ids) - len(sub_ids) + 1)):
                if full_ids[i : i + len(sub_ids)] == sub_ids:
                    for j in range(i, min(i + len(sub_ids), len(full_ids))):
                        mask[j] = True
                    if mask.any():
                        return mask

        return mask

    @staticmethod
    def _pool_hidden_tensor(hs, attention_mask, pool: str):
        import torch  # type: ignore

        pool_key = str(pool).lower()
        if pool_key == "mean":
            masked = hs * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-12)
        elif pool_key == "last":
            lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            pooled = hs[torch.arange(hs.size(0), device=hs.device), lengths]
        elif pool_key == "cls":
            pooled = hs[:, 0, :]
        else:
            raise ValueError(f"Unsupported hidden pooling '{pool}'. Use mean|last|cls.")
        return pooled

    def score_sentence(
        self,
        text: str,
        debug: bool = False,
        intervention: Optional[PLLIntervention] = None,
        return_hidden: bool = False,
        hidden_layer: Optional[int] = None,
        hidden_pool: str = "mean",
    ) -> PLLScore:
        if text in self._cache and not debug and intervention is None and not return_hidden:
            return self._cache[text]
        if not debug and intervention is None and return_hidden and text in self._cache:
            layers = self._resolve_layers()
            n_layers = len(layers)
            layer_idx = int(hidden_layer if hidden_layer is not None else (n_layers - 1))
            if layer_idx < 0:
                layer_idx = n_layers + layer_idx
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Invalid hidden_layer {hidden_layer}; model has {n_layers} layers.")
            pool_key = str(hidden_pool).lower()
            hidden_cached = self._hidden_cache_get((text, int(layer_idx), pool_key))
            if hidden_cached is not None:
                cached = self._cache[text]
                return PLLScore(
                    sum_logprob=cached.sum_logprob,
                    avg_logprob=cached.avg_logprob,
                    token_count=cached.token_count,
                    token_ids=cached.token_ids if debug else None,
                    token_logprobs=cached.token_logprobs if debug else None,
                    pooled_hidden=np.asarray(hidden_cached, dtype=np.float32),
                )

        import torch  # type: ignore

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        hook_handle = None
        if intervention is not None:
            hook_handle = self._register_intervention_hook(intervention, seq_len=int(input_ids.shape[1]))

        try:
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if intervention is not None:
                model_kwargs["use_cache"] = False
            if return_hidden:
                model_kwargs["output_hidden_states"] = True
            with torch.no_grad():
                outputs = self.model(**model_kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        token_logprobs = token_logprobs * shift_mask
        sum_logprob = float(token_logprobs.sum().item())
        token_count = int(shift_mask.sum().item())
        avg_logprob = sum_logprob / max(token_count, 1)

        pooled_hidden = None
        hidden_layer_idx = None
        hidden_pool_key = str(hidden_pool).lower()
        if return_hidden:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError("Model did not return hidden_states.")
            layers = self._resolve_layers()
            n_layers = len(layers)
            layer_idx = int(hidden_layer if hidden_layer is not None else (n_layers - 1))
            if layer_idx < 0:
                layer_idx = n_layers + layer_idx
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Invalid hidden_layer {hidden_layer}; model has {n_layers} layers.")
            hs, hs_len, layout_mode, selected_index = self._select_hidden_layer(
                hidden_states, layer_idx=layer_idx, n_layers=n_layers
            )  # [B, T, D]
            pooled = self._pool_hidden_tensor(hs, attention_mask, hidden_pool_key)
            pooled_hidden = np.asarray(pooled.squeeze(0).detach().float().cpu().numpy(), dtype=np.float32)
            hidden_layer_idx = int(layer_idx)
            self._last_hidden_layout = {
                "hidden_states_len": int(hs_len),
                "n_layers": int(n_layers),
                "layout_mode": layout_mode,
                "selected_index": int(selected_index),
            }

        score = PLLScore(
            sum_logprob=sum_logprob,
            avg_logprob=avg_logprob,
            token_count=token_count,
            token_ids=input_ids.squeeze(0).tolist() if debug else None,
            token_logprobs=token_logprobs.squeeze(0).tolist() if debug else None,
            pooled_hidden=pooled_hidden,
        )

        if not debug and intervention is None:
            self._cache[text] = PLLScore(
                sum_logprob=score.sum_logprob,
                avg_logprob=score.avg_logprob,
                token_count=score.token_count,
                token_ids=None,
                token_logprobs=None,
                pooled_hidden=None,
            )
            if return_hidden and pooled_hidden is not None and hidden_layer_idx is not None:
                self._hidden_cache_put(
                    (text, hidden_layer_idx, hidden_pool_key), np.asarray(pooled_hidden, dtype=np.float32)
                )

        return score

    def score_pair(
        self,
        pair: Dict[str, str],
        debug: bool = False,
        intervention: Optional[PLLIntervention] = None,
    ) -> Dict[str, Any]:
        stereo = pair.get("stereo", "")
        anti = pair.get("anti", "")
        if not stereo or not anti:
            raise ValueError("PLL pair must include 'stereo' and 'anti' strings.")

        s = self.score_sentence(stereo, debug=debug, intervention=intervention)
        a = self.score_sentence(anti, debug=debug, intervention=intervention)

        return {
            "stereo": s,
            "anti": a,
            "delta_sum": s.sum_logprob - a.sum_logprob,
            "delta_avg": s.avg_logprob - a.avg_logprob,
        }

    def extract_hidden(
        self,
        text: str,
        layer: int,
        pool: str = "mean",
        intervention: Optional[PLLIntervention] = None,
    ) -> np.ndarray:
        """
        Extract a pooled hidden vector at a specific transformer block output.

        layer is the block index (same indexing convention used by PLLIntervention.layer).
        """
        import torch  # type: ignore

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        layers = self._resolve_layers()
        n_layers = len(layers)
        layer_idx = int(layer)
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Invalid layer index {layer}; model has {n_layers} layers.")

        hook_handle = None
        if intervention is not None:
            hook_handle = self._register_intervention_hook(intervention, seq_len=int(input_ids.shape[1]))
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states.")
        hs, hs_len, layout_mode, selected_index = self._select_hidden_layer(
            hidden_states, layer_idx=layer_idx, n_layers=n_layers
        )  # [B, T, D]
        self._last_hidden_layout = {
            "hidden_states_len": int(hs_len),
            "n_layers": int(n_layers),
            "layout_mode": layout_mode,
            "selected_index": int(selected_index),
        }

        pooled = self._pool_hidden_tensor(hs, attention_mask, pool)

        vec = pooled.squeeze(0).detach().float().cpu().numpy()
        return np.asarray(vec, dtype=np.float32)
