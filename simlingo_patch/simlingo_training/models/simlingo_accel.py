"""One-call inference acceleration for SimLingo's :class:`DrivingModel`.

    from simlingo_training.models.simlingo_accel import accelerate
    model = build_and_load_checkpoint(...)   # your normal path: instantiate,
    model.eval()                             # load_state_dict, .eval()
    model = accelerate(model, warmup_example=example)   # done

``accelerate`` takes the loaded, evaluated model and returns the same object
patched for fast batch-1 inference.  It is self-contained: nothing here imports
``fast_inference.py``, so this single file is all that has to be dropped into
``<simlingo>/simlingo_training/models/``.

What it applies, in order (each step is independent and degrades gracefully):

  1. LoRA merge            fold the rank-32 adapters into the Qwen2 weights
                           (3 matmuls per linear -> 1)
  2. SDPA attention        FlashAttention-2 -> torch SDPA; FA2 unpads with a
                           data-dependent torch.nonzero and cannot be captured
                           into a CUDA graph
  3. KV cache              prefill the ~543-token prompt once into a StaticCache
                           instead of once per generated token
  4. head continuation     the 30 driving-query tokens are appended to the same
                           cache instead of a third full forward
  5. CUDA-graph decode     one captured single-token step, replayed per token
  6. torch.compile         inductor on the vision encoder (InternViT + pixel
                           shuffle + projector, static shape) and on the
                           long prefill call only.  Static shapes: dynamic
                           shapes trip an inductor autotuner assert on the
                           StaticCache write kernel (torch 2.12 / Thor).

Measured on Jetson AGX Thor (JetPack 7, torch 2.12), 543-token prompt:
stock 275 ms -> steps 1-5 128 ms -> steps 1-6 116 ms per frame.  On the Orin
(JetPack 6, no Triton) steps 1-5 gave 790 -> 294 ms; step 6 was unavailable.

Numerics: the bf16 LoRA merge perturbs outputs slightly (route max |diff|
~0.03 m against the untouched model; language identical).  ``verify`` checks
this on demand.  ``model._slow_forward`` keeps the original bound forward so
callers can A/B without reloading.

Limitations: batch size 1, ``predict_language=True`` path only, prompt +
generated tokens must fit ``max_cache_len``.  Each distinct prompt length costs
one extra prefill compile (~5-15 s, cached on disk via the inductor cache).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["accelerate", "verify"]


# --------------------------------------------------------------------------- #
# 1. LoRA merge
# --------------------------------------------------------------------------- #
def _merge_lora(model) -> bool:
    """Fold the LoRA deltas into the base weights.  Returns True if it happened.

    ``lm_head`` carries no adapter (``target_modules="all-linear"`` excludes the
    output embedding) so merging cannot change the sampling logits; the
    adaptor's references are rebound anyway because the PeftModel wrapper is
    gone after ``merge_and_unload``.
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
    if hasattr(merged, "embed_tokens"):
        merged.embed_tokens = getattr(inner, "embed_tokens", merged.embed_tokens)
    return True


# --------------------------------------------------------------------------- #
# 2. SDPA attention
# --------------------------------------------------------------------------- #
def _force_sdpa(causal_lm) -> bool:
    """Rebind every Qwen2 attention layer to the SDPA implementation.

    Qwen2 picks its attention class at ``__init__``; setting
    ``config._attn_implementation`` alone does nothing to built layers.
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


# --------------------------------------------------------------------------- #
# 6a. torch.compile of the vision encoder
# --------------------------------------------------------------------------- #
def _compile_vision(model, mode: str) -> bool:
    """Compile ``extract_feature`` (InternViT + pixel shuffle + mlp1).

    Static input [tiles, 3, 448, 448]; the tile count is fixed by the node's
    image geometry so ``dynamic=False`` is safe and fastest.
    """
    try:
        enc = model.vision_model.image_encoder.model
        if getattr(enc, "_accel_compiled", False):
            return True
        enc._accel_eager_extract_feature = enc.extract_feature
        enc.extract_feature = torch.compile(enc.extract_feature, mode=mode, dynamic=False)
        enc._accel_compiled = True
        return True
    except Exception as exc:
        logger.warning("vision compile skipped (%s: %s)", type(exc).__name__, exc)
        return False


# --------------------------------------------------------------------------- #
# 3-5 (+6b). Cached, graph-captured forward
# --------------------------------------------------------------------------- #
class _FastForward:
    """Drop-in replacement for ``DrivingModel.forward`` (batch size 1).

    Owns the StaticCache and the static decode tensors the CUDA graph is
    captured against.  ``prefill_lm`` is the (optionally compiled) module used
    for the long prompt forward; ``lm`` is the eager module used for the
    graph-captured single-token step and the 30-token driving-head step.
    """

    def __init__(self, model, max_cache_len: int, use_cuda_graph: bool,
                 max_new_tokens: int, compile_prefill: bool, compile_mode: str):
        self.model = model
        self.max_cache_len = max_cache_len
        self.use_cuda_graph = use_cuda_graph
        self.max_new_tokens = max_new_tokens

        self.lm = model.language_model.model
        self.device = next(self.lm.parameters()).device
        self.dtype = next(self.lm.parameters()).dtype
        hidden = model.language_model.hidden_size

        self.prefill_lm = self.lm
        self.prefill_compiled = False
        if compile_prefill:
            try:
                import torch._dynamo
                # one static graph per distinct prompt length; leave headroom
                torch._dynamo.config.cache_size_limit = max(
                    torch._dynamo.config.cache_size_limit, 64)
                self.prefill_lm = torch.compile(self.lm, mode=compile_mode, dynamic=False)
                self.prefill_compiled = True
            except Exception as exc:
                logger.warning("prefill compile skipped (%s: %s)", type(exc).__name__, exc)

        from transformers import StaticCache
        self.cache = StaticCache(
            config=self.lm.config, max_batch_size=1, max_cache_len=max_cache_len,
            device=self.device, dtype=self.dtype)
        self.s_emb = torch.zeros(1, 1, hidden, device=self.device, dtype=self.dtype)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.s_mask = torch.zeros(1, max_cache_len, device=self.device, dtype=torch.bool)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_out = None
        self._graph_failed = False

        self.eos_id = self._resolve_eos(model, model.tokenizer)
        self.embed_w = model.adaptors.language.embed_tokens.weight
        self.logit_w = model.adaptors.language.lm_head.weight

    @staticmethod
    def _resolve_eos(model, tokenizer) -> int:
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
            use_cache=True, output_hidden_states=True, return_dict=True)

    def _capture(self) -> bool:
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
            raise ValueError("simlingo_accel supports batch size 1 only; "
                             "call model._slow_forward for larger batches")

        # vision encode (compiled if requested) + placeholder replacement
        adaptor_dict = model.adaptors(example, inference=True)
        adaptor_dict = model.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=model.wp_encoder)
        inputs = adaptor_dict["language_inputs"][0].unsqueeze(0).to(self.dtype)
        n_prompt = inputs.size(1)
        if n_prompt + self.max_new_tokens + 64 > self.max_cache_len:
            raise ValueError(f"prompt of {n_prompt} tokens will not fit in a cache of "
                             f"{self.max_cache_len}; raise max_cache_len")

        # prefill once (compiled copy of the LM if requested)
        self.cache.reset()
        self.s_mask.zero_()
        self.s_mask[:, :n_prompt] = True
        out = self.prefill_lm(
            inputs_embeds=inputs,
            attention_mask=self.s_mask[:, :n_prompt].contiguous(),
            past_key_values=self.cache,
            cache_position=torch.arange(n_prompt, device=self.device),
            use_cache=True, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1][:, -1]

        # greedy decode, one graph replay per token
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

        # driving head as a continuation of the same cache
        queries = model.adaptors.driving(driving_input)["inputs"].to(self.dtype)
        n_drv = queries.size(1)
        self.s_mask[0, pos:pos + n_drv] = True
        drv = self.lm(
            inputs_embeds=queries,
            attention_mask=self.s_mask[:, :pos + n_drv].contiguous(),
            past_key_values=self.cache,
            cache_position=torch.arange(pos, pos + n_drv, device=self.device),
            use_cache=True, output_hidden_states=True, return_dict=True)
        feats = drv.hidden_states[-1][:, -n_drv:]
        logits = drv[0][:, -n_drv:]
        preds = model.adaptors.driving.get_predictions(feats, logits)

        text = model.tokenizer.batch_decode(
            torch.cat(tokens).unsqueeze(0), skip_special_tokens=True)[0]
        model.speed_wps = preds.get("speed_wps")
        model.route = preds.get("route")
        model.language = [text]
        return model.speed_wps, model.route, model.language


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def accelerate(model, *,
               merge_lora: bool = True,
               sdpa: bool = True,
               use_cuda_graph: bool = True,
               compile_vision: bool = True,
               compile_prefill: bool = True,
               compile_mode: str = "default",
               max_cache_len: int = 768,
               max_new_tokens: int = 100,
               warmup_example=None):
    """Patch ``model`` in place for fast batch-1 inference and return it.

    ``model`` must already be built, have its checkpoint loaded, and be in
    ``eval()`` mode on the GPU.  Pass ``warmup_example`` (any DrivingInput of
    the shape the node will feed) to pay the compile and CUDA-graph capture
    cost up front instead of on the first real frame.

    Every step is a flag so a partial configuration is still a working model;
    ``model._accel_report`` records what was actually applied.
    """
    if getattr(model, "_fast_inference", None) is not None:
        return model
    if not getattr(model, "predict_language", True):
        raise ValueError("simlingo_accel targets the predict_language=True path")

    report: Dict[str, object] = {}
    report["lora_merged"] = _merge_lora(model) if merge_lora else False
    report["sdpa"] = _force_sdpa(model.language_model.model) if sdpa else False
    if use_cuda_graph and not report["sdpa"]:
        logger.warning("SDPA unavailable; disabling CUDA graph capture "
                       "(FlashAttention-2 is not capturable)")
        use_cuda_graph = False
    report["vision_compiled"] = _compile_vision(model, compile_mode) if compile_vision else False

    fast = _FastForward(model, max_cache_len=max_cache_len, use_cuda_graph=use_cuda_graph,
                        max_new_tokens=max_new_tokens, compile_prefill=compile_prefill,
                        compile_mode=compile_mode)
    report["prefill_compiled"] = fast.prefill_compiled
    report["cuda_graph_requested"] = use_cuda_graph
    report["compile_mode"] = compile_mode
    report["max_cache_len"] = max_cache_len

    model._slow_forward = model.forward
    model._fast_inference = fast
    model.forward = fast

    if warmup_example is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(warmup_example)
            model(warmup_example)
        torch.cuda.synchronize()
    report["cuda_graph_active"] = fast._graph is not None
    model._accel_report = report
    logger.info("simlingo_accel: %s", report)
    return model


def verify(model, example, *, atol_m: float = 0.25) -> Dict[str, object]:
    """Compare the accelerated path against the merged model's slow path."""
    fast = getattr(model, "_fast_inference", None)
    if fast is None:
        raise ValueError("call accelerate(model) first")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        f_speed, f_route, f_lang = model(example)
        s_speed, s_route, s_lang = model._slow_forward(example)

    def _maxdiff(a, b):
        return None if a is None or b is None else float((a.float() - b.float()).abs().max())

    res = {
        "route_max_diff_m": _maxdiff(f_route, s_route),
        "speed_max_diff_m": _maxdiff(f_speed, s_speed),
        "language_fast": f_lang[0] if f_lang else None,
        "language_slow": s_lang[0] if s_lang else None,
        "cuda_graph_active": fast._graph is not None,
        "report": getattr(model, "_accel_report", {}),
    }
    res["language_match"] = (res["language_fast"] or "").strip() == \
                            (res["language_slow"] or "").strip()
    res["ok"] = (res["language_match"]
                 and (res["route_max_diff_m"] or 0.0) <= atol_m
                 and (res["speed_max_diff_m"] or 0.0) <= atol_m)
    return res
