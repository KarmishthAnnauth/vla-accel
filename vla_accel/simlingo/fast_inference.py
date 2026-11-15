"""Fast batch-1 inference path for SimLingo's ``DrivingModel``.

LoRA merge, FlashAttention2 -> SDPA, StaticCache KV cache, CUDA-graph decode step and
the driving head as a cache continuation; ``optimize_for_inference`` patches ``model.forward``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["optimize_for_inference", "verify"]


def _merge_lora(model) -> bool:
    """Fold the LoRA deltas into the base weights; True if merged."""
    lm = model.language_model
    inner = getattr(lm, "model", None)
    if inner is None or not hasattr(inner, "merge_and_unload"):
        return False

    merged = inner.merge_and_unload().eval()
    lm.model = merged

    # Rebind references so nothing keeps the old PeftModel graph alive.
    lang = model.adaptors.language
    if hasattr(merged, "lm_head"):
        lang.lm_head = merged.lm_head
    for attr in ("embed_tokens",):
        if hasattr(merged, attr):
            setattr(merged, attr, getattr(inner, attr, getattr(merged, attr, None)))
    return True


def _force_sdpa(causal_lm) -> bool:
    """Rebind every Qwen2 attention layer to SDPA (FA2 is not graph-capturable)."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
    except Exception:
        return False

    base = getattr(causal_lm, "model", causal_lm)
    layers = getattr(base, "layers", None)
    if layers is None:
        return False

    causal_lm.config._attn_implementation = "sdpa"
    for layer in layers:
        # Qwen2 picks its attention class at __init__, so the config flag alone is not enough.
        layer.self_attn.__class__ = Qwen2SdpaAttention
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"
    return True


class _FastForward:
    """Drop-in replacement for ``DrivingModel.forward`` holding the static cache and decode graph."""

    def __init__(self, model, max_cache_len: int = 768, use_cuda_graph: bool = True,
                 max_new_tokens: int = 100):
        self.model = model
        self.max_cache_len = max_cache_len
        self.use_cuda_graph = use_cuda_graph
        self.max_new_tokens = max_new_tokens

        self.lm = model.language_model.model
        self.device = next(self.lm.parameters()).device
        self.dtype = next(self.lm.parameters()).dtype
        hidden = model.language_model.hidden_size

        from transformers import StaticCache
        self.cache = StaticCache(
            config=self.lm.config, max_batch_size=1, max_cache_len=max_cache_len,
            device=self.device, dtype=self.dtype,
        )
        self.s_emb = torch.zeros(1, 1, hidden, device=self.device, dtype=self.dtype)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.s_mask = torch.zeros(1, max_cache_len, device=self.device, dtype=torch.bool)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_out = None
        self._graph_failed = False

        tok = model.tokenizer
        self.eos_id = self._resolve_eos(model, tok)
        self.embed_w = model.adaptors.language.embed_tokens.weight
        self.logit_w = model.adaptors.language.lm_head.weight

    @staticmethod
    def _resolve_eos(model, tokenizer) -> int:
        """End-of-turn token id, chosen per variant as in driving.py."""
        variant = getattr(model.language_model, "variant", "")
        added = getattr(tokenizer, "added_tokens_encoder", {})
        if variant == "OpenGVLab/InternVL2-4B" and "<|end|>" in added:
            return added["<|end|>"]
        if variant == "OpenGVLab/InternVL2-2B" and "<|im_end|>" in added:
            return added["<|im_end|>"]
        return tokenizer.eos_token_id

    def _decode_eager(self):
        """One cached decode step on the static tensors."""
        return self.lm(
            inputs_embeds=self.s_emb, attention_mask=self.s_mask,
            past_key_values=self.cache, cache_position=self.s_pos,
            use_cache=True, output_hidden_states=True, return_dict=True,
        )

    def _capture(self) -> bool:
        """Capture one decode step into a CUDA graph; fall back to eager on failure."""
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            saved = int(self.s_pos.item())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._decode_eager()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.s_pos.fill_(saved)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._graph_out = self._decode_eager()
            torch.cuda.synchronize()
            self._graph = graph
            return True
        except Exception as exc:
            logger.warning("CUDA graph capture failed (%s); using eager decode",
                           type(exc).__name__)
            self._graph = None
            self._graph_out = None
            self._graph_failed = True
            return False

    def _decode(self):
        """One cached decode step, graph-replayed when possible."""
        if self.use_cuda_graph and self._graph is None and not self._graph_failed:
            self._capture()
        if self._graph is not None:
            self._graph.replay()
            return self._graph_out
        return self._decode_eager()

    @torch.no_grad()
    def __call__(self, example, return_language: Optional[bool] = None,
                 prompt_ids: Optional[Tensor] = None):
        model = self.model
        driving_input = getattr(example, "driving_input", example)
        if driving_input.camera_images.size(0) != 1:
            raise ValueError("fast_inference supports batch size 1 only; "
                             "call model._slow_forward for larger batches")

        adaptor_dict = model.adaptors(example, inference=True)
        adaptor_dict = model.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=model.wp_encoder,
        )
        inputs = adaptor_dict["language_inputs"][0].unsqueeze(0).to(self.dtype)
        n_prompt = inputs.size(1)
        if n_prompt + self.max_new_tokens + 64 > self.max_cache_len:
            raise ValueError(
                f"prompt of {n_prompt} tokens will not fit in a cache of "
                f"{self.max_cache_len}; raise max_cache_len")

        self.cache.reset()
        self.s_mask.zero_()
        self.s_mask[:, :n_prompt] = True
        out = self.lm(
            inputs_embeds=inputs,
            attention_mask=self.s_mask[:, :n_prompt].contiguous(),
            past_key_values=self.cache,
            cache_position=torch.arange(n_prompt, device=self.device),
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        hidden = out.hidden_states[-1][:, -1]

        tokens: List[Tensor] = []
        pos = n_prompt
        for _ in range(self.max_new_tokens):
            nxt = F.linear(hidden, self.logit_w).argmax(dim=-1)
            tokens.append(nxt)
            self.s_emb.copy_(F.embedding(nxt.unsqueeze(1), self.embed_w).to(self.dtype))
            self.s_pos.fill_(pos)
            self.s_mask[0, pos] = True
            # EOS is written into the cache before breaking, as greedy_sample does.
            step = self._decode()
            hidden = step.hidden_states[-1][:, -1]
            pos += 1
            if self.eos_id is not None and int(nxt) == int(self.eos_id):
                break

        queries = model.adaptors.driving(driving_input)["inputs"].to(self.dtype)
        n_drv = queries.size(1)
        self.s_mask[0, pos:pos + n_drv] = True
        drv = self.lm(
            inputs_embeds=queries,
            attention_mask=self.s_mask[:, :pos + n_drv].contiguous(),
            past_key_values=self.cache,
            cache_position=torch.arange(pos, pos + n_drv, device=self.device),
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        feats = drv.hidden_states[-1][:, -n_drv:]
        logits = drv[0][:, -n_drv:]
        preds = model.adaptors.driving.get_predictions(feats, logits)

        text = model.tokenizer.batch_decode(
            torch.cat(tokens).unsqueeze(0), skip_special_tokens=True)[0]

        model.speed_wps = preds.get("speed_wps")
        model.route = preds.get("route")
        model.language = [text]
        return model.speed_wps, model.route, model.language


def optimize_for_inference(model, *, max_cache_len: int = 768,
                           use_cuda_graph: bool = True,
                           max_new_tokens: int = 100) -> Dict[str, object]:
    """Patch ``model`` in place for fast batch-1 inference and report which steps applied."""
    if getattr(model, "_fast_inference", None) is not None:
        return model._fast_inference_report

    if not getattr(model, "predict_language", True):
        raise ValueError("fast_inference targets the predict_language=True path")

    report: Dict[str, object] = {}
    report["lora_merged"] = _merge_lora(model)
    report["sdpa"] = _force_sdpa(model.language_model.model)
    if use_cuda_graph and not report["sdpa"]:
        logger.warning("could not switch to SDPA; disabling CUDA graph capture "
                       "(FlashAttention2 is not capturable)")
        use_cuda_graph = False

    fast = _FastForward(model, max_cache_len=max_cache_len,
                        use_cuda_graph=use_cuda_graph,
                        max_new_tokens=max_new_tokens)
    model._slow_forward = model.forward
    model._fast_inference = fast
    model.forward = fast
    report["cuda_graph_requested"] = use_cuda_graph
    report["max_cache_len"] = max_cache_len
    model._fast_inference_report = report
    return report


def verify(model, example, *, atol_m: float = 0.25) -> Dict[str, object]:
    """Compare the fast path with the merged model's original forward on one input."""
    fast = getattr(model, "_fast_inference", None)
    if fast is None:
        raise ValueError("call optimize_for_inference(model) first")

    with torch.no_grad():
        f_speed, f_route, f_lang = model(example)
        s_speed, s_route, s_lang = model._slow_forward(example)

    def _maxdiff(a, b):
        if a is None or b is None:
            return None
        return float((a.float() - b.float()).abs().max())

    res = {
        "route_max_diff_m": _maxdiff(f_route, s_route),
        "speed_max_diff_m": _maxdiff(f_speed, s_speed),
        "language_fast": f_lang[0] if f_lang else None,
        "language_slow": s_lang[0] if s_lang else None,
        "cuda_graph_active": fast._graph is not None,
    }
    res["language_match"] = (res["language_fast"] or "").strip() == \
                            (res["language_slow"] or "").strip()
    res["ok"] = (res["language_match"]
                 and (res["route_max_diff_m"] or 0.0) <= atol_m
                 and (res["speed_max_diff_m"] or 0.0) <= atol_m)
    return res
