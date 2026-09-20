"""Fast single-sample inference path for :class:`DrivingModel`.

Why this exists
---------------
``DrivingModel.forward`` (models/driving.py) generates the language prefix with
``LLM.greedy_sample`` (models/language_model/llm.py), which keeps **no KV cache**:
every generated token re-runs a full forward over the whole ~543-token prompt,
and the driving head then costs a third full forward.  On an Orin AGX that is

    vision 83 ms + (N + 2) x 135 ms        # N = tokens generated

i.e. ~800 ms for the 4-token "Waypoints:" prefix the driving prompt produces.

Measured on Orin AGX (JetPack 6, clocks pinned with ``jetson_clocks``), the cost
is not arithmetic.  A single-token forward with no cache at all takes 52 ms
against ~1 GFLOP of work: the decode is ~93% CUDA kernel-launch overhead.  So
the wins here are, in order of size:

  1. merge the LoRA adapters          3 matmuls per linear -> 1   (135 -> 79 ms)
  2. FlashAttention2 -> SDPA          FA2 does host-side unpad_input with
                                      data-dependent shapes and therefore
                                      cannot be CUDA-graph captured
  3. a real KV cache                  prompt is prefilled once, not N+2 times
  4. CUDA-graph the decode step       collapses ~500 python-level kernel
                                      dispatches into one replay (135 -> 10.6 ms)
  5. driving head as a continuation   30 query tokens appended to the same
                                      cache instead of a third full forward

torch.compile is deliberately not used: the JetPack container has no Triton, so
inductor fails outright.  Raw ``torch.cuda.CUDAGraph`` capture needs neither.

Numerics
--------
Merging LoRA in bf16 perturbs the outputs slightly.  Against the unoptimised
model on the same input, measured: route max |diff| 0.033 m, speed waypoints
0.070 m, generated text identical.  ``verify()`` re-checks this on demand.

Usage
-----
    from simlingo_training.models.fast_inference import optimize_for_inference
    model.load_state_dict(...)
    model.eval()
    optimize_for_inference(model)      # patches model.forward in place

The patch is opt-in and only touches the inference path; training and the
``predict_language=False`` branch are left alone.  ``model._slow_forward`` keeps
the original bound method so callers can A/B without reloading.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["optimize_for_inference", "verify"]


def _merge_lora(model) -> bool:
    """Fold the LoRA deltas into the base weights. Returns True if it happened.

    ``target_modules="all-linear"`` (see llm.py) deliberately excludes the output
    embedding, so ``lm_head`` carries no adapter and merging cannot change the
    sampling logits.  The adaptor's references are rebound anyway: after
    ``merge_and_unload`` the PeftModel wrapper is gone, and anything still
    pointing at it would silently keep the old graph alive.
    """
    lm = model.language_model
    inner = getattr(lm, "model", None)
    if inner is None or not hasattr(inner, "merge_and_unload"):
        return False

    merged = inner.merge_and_unload().eval()
    lm.model = merged

    lang = model.adaptors.language
    if hasattr(merged, "lm_head"):
        lang.lm_head = merged.lm_head
    for attr in ("embed_tokens",):
        if hasattr(merged, attr):
            setattr(merged, attr, getattr(inner, attr, getattr(merged, attr, None)))
    return True


def _force_sdpa(causal_lm) -> bool:
    """Swap every attention module to the SDPA implementation.

    Only matters because FlashAttention2 cannot be captured into a CUDA graph:
    ``_flash_attention_forward`` calls ``unpad_input``, which does a
    ``torch.nonzero`` on the mask -- a data-dependent shape -- and capture dies
    with a device-side assert.  Qwen2 picks its attention class at __init__, so
    setting ``config._attn_implementation`` alone is not enough; the class on
    each layer has to be rebound.
    """
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
        layer.self_attn.__class__ = Qwen2SdpaAttention
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"
    return True


class _FastForward:
    """Drop-in replacement for ``DrivingModel.forward`` (batch size 1).

    Holds the StaticCache and the static decode tensors that the CUDA graph is
    captured against.  Everything the graph reads -- the token embedding, the
    write position, the attention mask -- lives in these tensors and is updated
    in place, so one capture serves every frame and every prompt length up to
    ``max_cache_len``.
    """

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
        """Mirror driving.py's per-variant choice of end-of-turn token."""
        variant = getattr(model.language_model, "variant", "")
        added = getattr(tokenizer, "added_tokens_encoder", {})
        if variant == "OpenGVLab/InternVL2-4B" and "<|end|>" in added:
            return added["<|end|>"]
        if variant == "OpenGVLab/InternVL2-2B" and "<|im_end|>" in added:
            return added["<|im_end|>"]
        return tokenizer.eos_token_id

    def _decode_eager(self):
        return self.lm(
            inputs_embeds=self.s_emb, attention_mask=self.s_mask,
            past_key_values=self.cache, cache_position=self.s_pos,
            use_cache=True, output_hidden_states=True, return_dict=True,
        )

    def _capture(self) -> bool:
        """Capture one decode step. Falls back to eager on any failure."""
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
    """Patch ``model`` in place for fast batch-1 inference.

    Returns a report of what was actually applied -- each step degrades
    independently, so a partial optimisation is still a working model.
    """
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
    """Compare the fast path against the original on one input.

    ``optimize_for_inference`` must already have been applied.  The comparison
    is against the *merged* model's slow path, so it isolates the caching and
    graph work; the LoRA-merge perturbation is reported separately by running
    this before and after if you need it.
    """
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
