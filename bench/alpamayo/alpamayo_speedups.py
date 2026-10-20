"""Inference-time speedups for Alpamayo 1.5, applied to the *loaded model*.

Nothing in here edits alpamayo1_5 or transformers: FastAlpamayo reads the
sub-modules of the model the node already has and reproduces what
``Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout`` computes for one
trajectory sample with less work.  The model is left untouched and remains the
fallback for any input the fast path does not cover (batches, several samples,
navigation text / classifier-free guidance).

What the reference path spends its time on (Orin, clocks pinned, one camera x
4 frames = 828 prompt tokens, ~16 generated tokens, 5 Euler steps, 3380 ms):

  vision encoder, 4 images        236 ms   2.4x its GEMM time
  LLM prefill, 828 tokens         739 ms   compute-bound
  LLM decode, ~16 steps          1617 ms   100 ms/token: one pass over 12.9 GiB of
                                           weights at the memory bus's ~160 GB/s.
                                           A CUDA graph does not help (85 ms floor)
  lm_head                         132 ms   incl. logits for all 828 prompt rows
  generate() bookkeeping           70 ms
  action expert, 5 steps          569 ms   110 ms/step for a 30 ms weight pass:
                                           launch overhead

and what is done about each:

  * decode: a one-token step costs the same as a 64-token one (it is bound by
    reading the weights, not by arithmetic), so tokens are not generated one
    pass at a time when that can be avoided.  The reasoning text of the
    previous frame is appended to the prompt as a *draft* and checked in the
    prefill pass itself: every draft token the model would have produced at
    that position anyway is accepted, with its keys/values already in the
    cache (KV-cache reuse); sequential decoding resumes at the first
    disagreement.  Under greedy decoding the result is the token sequence
    sequential decoding gives; under sampling, draft tokens are accepted with
    their model probability and rejected ones resampled from the remainder,
    which leaves the sampling distribution unchanged.  A scene whose reasoning
    did not change costs no decode pass at all.
  * prefill: fixed-size window under a CUDA graph; the lm_head only sees the
    rows whose logits are read.
  * decode steps that remain: one CUDA graph over a pre-allocated KV buffer.
  * expert: the whole Euler loop is one CUDA graph attending into the same KV
    buffer (no cache copies, no crop).
  * vision encoder: position tables computed once, the four equally sized
    images attend as one batch, CUDA graph.

There is no LoRA to merge: the released checkpoint has none.

``verify()`` measures the fast path against the unmodified model with greedy
decoding and the same diffusion noise.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

from alpamayo1_5.models.token_utils import extract_text_tokens, to_special_token

DRAFT_CAP = 32
STEP_ROWS = 16
LIBRARY_SIZE = 8
_NEG = float("-inf")


class _Stage:
    """A function of static tensors, run eagerly or replayed as a CUDA graph."""

    def __init__(self, fn: Callable[[], Any], capture: bool, warmup: int = 2):
        self._fn = fn
        self._graph = None
        for _ in range(warmup):
            fn()
        if capture:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.out = fn()
            self._graph = graph
        else:
            self.out = fn()

    def __call__(self) -> Any:
        if self._graph is not None:
            self._graph.replay()
        else:
            self.out = self._fn()
        return self.out


class _TrtEngine:
    """A TensorRT engine with static shapes.  Outputs live in torch tensors;
    an input is either a torch tensor of its own or bound to an existing one
    (``bind``), which is how consecutive decoder layers are chained without a
    copy."""

    def __init__(self, path: str, runtime, bind: Optional[Dict[str, torch.Tensor]] = None):
        import tensorrt as trt

        with open(path, "rb") as f:
            blob = f.read()
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        self._engine = None
        for attempt in range(3):
            self._engine = runtime.deserialize_cuda_engine(blob)
            if self._engine is not None:
                break
            torch.cuda.empty_cache()
            time.sleep(3.0)
        del blob
        if self._engine is None:
            raise RuntimeError(f"cannot deserialize {path}")
        self._context = self._engine.create_execution_context()
        self.inputs: Dict[str, torch.Tensor] = {}
        self.outputs: Dict[str, torch.Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise ValueError(f"{path}: tensor {name} is not fp32")
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input and bind and name in bind:
                tensor = bind[name]
                if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                    raise ValueError(f"{path}: {name} is {shape}, bound tensor is {tuple(tensor.shape)}")
            else:
                tensor = torch.zeros(shape, device="cuda", dtype=torch.float32)
            (self.inputs if is_input else self.outputs)[name] = tensor
            self._context.set_tensor_address(name, tensor.data_ptr())

    def run(self, stream: torch.cuda.Stream) -> None:
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            mask: torch.Tensor, scale: float) -> torch.Tensor:
    """Grouped-query attention of q [1,H,Q,D] over a whole KV buffer [1,G,T,D]
    without expanding the buffer to H heads; ``mask`` [1,1,Q or 1,T] is additive.
    The arithmetic is F.scaled_dot_product_attention's: sqrt(scale) on both
    sides, everything in bf16."""
    b, h, n_q, d = q.shape
    g = k.shape[1]
    with torch.autocast("cuda", enabled=False):
        root = math.sqrt(scale)
        q = (q.to(k.dtype) * root).reshape(b, g, (h // g) * n_q, d)
        scores = torch.matmul(q, (k * root).transpose(-1, -2)).view(b, g, h // g, n_q, -1)
        probs = torch.softmax(scores + mask.unsqueeze(2), dim=-1).view(b, g, (h // g) * n_q, -1)
        return torch.matmul(probs, v).view(b, h, n_q, d)


_F32, _BF16 = torch.float32, torch.bfloat16


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rne(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> the bf16 value eager's cast gives (round to nearest even), kept
    as fp32.  An eager bf16 op is "compute in fp32, round to bf16"; a fused
    kernel computes the whole chain in fp32 and rounds once at the store, so
    the intermediate roundings have to be put back by hand."""
    i = v.view(torch.int32)
    return ((i + 0x7FFF + ((i >> 16) & 1)) & -65536).view(_F32)


def _rope_ref(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


def _rope_fused(x, cos, sin):
    xf = x.to(_F32)
    return (_rne(xf * cos.to(_F32)) + _rne(_rotate_half(xf) * sin.to(_F32))).to(_BF16)


def _rope32_fused(x, cos, sin):
    xf = x.to(_F32)
    return xf * cos + _rotate_half(xf) * sin


def _silu_mul_ref(gate, up):
    return F.silu(gate) * up


def _silu_mul_fused(gate, up):
    g = gate.to(_F32)
    return (_rne(g / (1 + torch.exp(-g))) * up.to(_F32)).to(_BF16)


def _square_fused(x):
    return x.to(_F32).pow(2)


def _rms_tail_fused(x, inv_rms, weight):
    return (_rne(x.to(_F32) * inv_rms) * weight.to(_F32)).to(_BF16)


def _rms32_ref(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _vrope_fused(q, k, cos, sin):
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    qf, kf = q.to(_F32), k.to(_F32)
    return ((qf * cos + _rotate_half(qf) * sin).to(q.dtype),
            (kf * cos + _rotate_half(kf) * sin).to(k.dtype))


def _expert_layer_fused(h, cos, sin, kb, vb, mask, wq, wk, wv, wo, wg, wu, wd, n1, n2, qn, kn,
                        eps: float = 1e-6):
    """One expert decoder layer on the fp32 residual stream ``h`` [1,n,E], its
    keys/values going to the last n rows of the KV buffer.  Same operations
    and bf16 rounding points as FastAlpamayo._expert_layer, written on tensors
    only so that one compilation serves every layer and everything between
    the GEMMs fuses."""
    n, g, hd = h.shape[1], kb.shape[1], kb.shape[-1]
    root = hd ** -0.25
    x = (n1 * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))).to(_BF16)

    def rotated(w, wn):
        t = F.linear(x, w).view(1, n, -1, hd).to(_F32)
        t = _rne(_rne(t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)) * wn.to(_F32))
        t = t.transpose(1, 2)
        return t * cos + _rotate_half(t) * sin

    q, k = rotated(wq, qn), rotated(wk, kn)
    kb[:, :, -n:] = k.to(_BF16)
    vb[:, :, -n:] = F.linear(x, wv).view(1, n, -1, hd).transpose(1, 2)
    q = _rne(_rne(q) * root).to(_BF16).reshape(1, g, -1, hd)
    scores = torch.matmul(q, (kb * root).transpose(-1, -2)).view(1, g, -1, n, kb.shape[2])
    probs = torch.softmax(scores.to(_F32) + mask.to(_F32).unsqueeze(2), dim=-1)
    probs = probs.to(_BF16).view(1, g, -1, kb.shape[2])
    a = torch.matmul(probs, vb).view(1, -1, n, hd).transpose(1, 2).reshape(1, n, -1)
    h = h + F.linear(a, wo)
    x = (n2 * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))).to(_BF16)
    gate = F.linear(x, wg).to(_F32)
    act = (_rne(gate / (1 + torch.exp(-gate))) * F.linear(x, wu).to(_F32)).to(_BF16)
    return h + F.linear(act, wd)


class _FastLinear:
    """A bias-free nn.Linear through a pre-transposed, contiguous copy of its
    weight: F.linear's transposed bf16 GEMM runs at two thirds of the speed
    for 12288 -> 4096 at prefill row counts."""

    def __init__(self, linear: torch.nn.Linear):
        self.weight_t = linear.weight.t().contiguous()
        self.out_features = linear.out_features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x.reshape(-1, x.shape[-1]), self.weight_t).view(
            *x.shape[:-1], self.out_features)


class FastAlpamayo:
    """One-sample trajectory rollout for a loaded Alpamayo1_5, batch 1.

    model           the Alpamayo1_5 on the GPU, eval mode
    max_new_tokens  generation limit the KV buffer is sized for
    capture         False runs the same code without CUDA graphs (debugging)
    speculate       False disables the previous-frame draft (every token is a
                    sequential decode step)
    fuse            False keeps the stock linears and elementwise ops
    trt_dir         directory of TensorRT engines from tools/trt_build.py (vision
                    encoder and one engine per decoder layer for the prefill
                    window); "" = PyTorch for everything.  fp16 instead of
                    bf16, +13 GiB of GPU memory, engines are loaded only if
                    they were built for the live prompt layout.
    tie_margin      greedy decoding keeps a proposed token whose logit is within
                    this much of the best one (0 = strict argmax).  A draft row
                    is computed in a larger batch than a sequential step, which
                    moves bf16 logits by a few hundredths; without a margin a
                    near-tie between two continuations can flip with the path
                    taken.
    """

    def __init__(self, model, max_new_tokens: int = 64, capture: bool = True,
                 speculate: bool = True, fuse: bool = True,
                 trt_dir: str = "", tie_margin: float = 0.25, log: Optional[Callable[[str], None]] = None,
                 profile: bool = False):
        self.model = model
        self._log = log or (lambda msg: None)
        self.profile = profile
        self.capture = capture
        self.speculate = speculate
        self.fuse = fuse
        self.tie_margin = float(tie_margin)
        self.trt_dir = trt_dir
        self._trt_vit: Optional[_TrtEngine] = None
        self._trt_layers: Optional[List[_TrtEngine]] = None
        self.trace_logits: Optional[List[torch.Tensor]] = None
        self.max_new = int(max_new_tokens)
        self.timing: Dict[str, float] = {}
        self.stats: Dict[str, int] = {}
        self.fallbacks = 0

        vlm = model.vlm
        self._vlm = vlm.model
        self._lm = vlm.model.language_model
        self._visual = vlm.model.visual
        self._head = vlm.lm_head
        self._expert = model.expert
        self._dev = torch.device("cuda")
        self._dtype = self._lm.embed_tokens.weight.dtype
        for cfg in (self._lm.config, self._expert.config, self._visual.config):
            if cfg._attn_implementation != "sdpa":
                raise ValueError("FastAlpamayo reproduces the SDPA attention path only")
        attn = self._lm.layers[0].self_attn
        self._hd = attn.head_dim
        self._kv_heads = attn.k_proj.out_features // attn.head_dim
        e_attn = self._expert.layers[0].self_attn
        if (e_attn.head_dim, e_attn.k_proj.out_features) != (attn.head_dim, attn.k_proj.out_features):
            raise ValueError("expert and VLM key/value shapes differ")
        if len(self._expert.layers) != len(self._lm.layers):
            raise ValueError("expert and VLM depths differ")
        if not model.config.expert_non_causal_attention:
            raise ValueError("FastAlpamayo implements the non-causal expert only")
        self._n_act, self._act_dim = model.action_space.get_action_space_dims()
        self._image_token = vlm.config.image_token_id
        self._eos = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        self._pad = model.tokenizer.pad_token_id
        self._traj_lo = model.config.traj_token_start_idx
        self._traj_hi = self._traj_lo + model.config.traj_vocab_size
        self._library: List[List[int]] = []
        self._layout: Optional[Tuple] = None
        self._steps = 0

    def _build(self, ids: torch.Tensor, grid: torch.Tensor, pixels: torch.Tensor) -> None:
        t0 = time.time()
        dev, dt = self._dev, self._dtype
        n_layers = len(self._lm.layers)
        self._L = L = ids.shape[1]
        self._W = L + DRAFT_CAP
        self._C = max(self._W, L + self.max_new + STEP_ROWS)
        self._T = T = self._C + self._n_act
        self._img_mask = ids[0] == self._image_token
        self._img_pos = self._img_mask.nonzero()[:, 0]
        self._grid = grid.clone()
        sizes = (grid[:, 1] * grid[:, 2]).tolist()
        if grid[:, 0].ne(1).any() or len(set(sizes)) != 1:
            raise ValueError("FastAlpamayo needs still images of one size")
        self._n_img, self._img_seq = len(sizes), sizes[0]

        pos, delta = self._vlm.get_rope_index(ids, grid, None, torch.ones_like(ids))
        self._delta = int(delta.reshape(-1)[0])
        tail = torch.arange(L, self._C, device=dev) + self._delta
        pos = torch.cat([pos, tail.view(1, 1, -1).expand(3, 1, -1)], dim=-1)
        probe = torch.zeros(1, 1, 1, device=dev, dtype=dt)
        cos, sin = self._lm.rotary_emb(probe, pos)
        self._cos, self._sin = cos.unsqueeze(1), sin.unsqueeze(1)
        self._vit_pos = self._visual.fast_pos_embed_interpolate(grid)
        rot = self._visual.rot_pos_emb(grid)
        rot = torch.cat((rot, rot), dim=-1)
        self._vit_cos, self._vit_sin = rot.cos(), rot.sin()

        self._ids = torch.full((1, self._W), self._pad, device=dev, dtype=torch.long)
        self._pix = torch.zeros_like(pixels)
        self._kb = [torch.zeros(1, self._kv_heads, T, self._hd, device=dev, dtype=dt)
                    for _ in range(n_layers)]
        self._vb = [torch.zeros_like(k) for k in self._kb]
        self._toks = torch.full((1, STEP_ROWS), self._pad, device=dev, dtype=torch.long)
        self._rows = torch.arange(L, L + STEP_ROWS, device=dev)
        self._step_range = torch.arange(STEP_ROWS, device=dev)
        self._cols = torch.arange(T, device=dev)
        self._emask = torch.zeros(1, 1, 1, T, device=dev, dtype=dt)
        self._eoff = torch.zeros(1, device=dev, dtype=torch.long)
        self._noise = torch.zeros(1, self._n_act, self._act_dim, device=dev)
        self._act_range = torch.arange(self._n_act, device=dev)
        self._steps = int(self.model.diffusion.num_inference_steps)
        self._times = torch.linspace(0.0, 1.0, self._steps + 1, device=dev)

        self._ids[0, :L] = ids[0]
        self._pix.copy_(pixels)
        self._load_trt()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._init_kernels()
            if self._trt_vit is not None:
                self._vit = _Stage(self._vit_trt_fn, False)
            else:
                self._vit = _Stage(self._vit_fn, self.capture)
            if self._trt_layers is not None:
                self._prefill = _Stage(self._prefill_trt_fn, False)
            else:
                self._prefill = _Stage(self._prefill_fn, self.capture)
            self._body = _Stage(self._body_fn, self.capture)
            self._logits = _Stage(self._logits_fn, self.capture)
            self._plan = _Stage(self._expert_fn, self.capture)
        torch.cuda.synchronize()
        self._layout = (L, tuple(grid.flatten().tolist()))
        self._library = []
        self._log(f"FastAlpamayo ready: prompt {L} tokens, prefill window {self._W}, KV buffer "
                  f"{T} rows, {self._n_img} images x {self._img_seq} patches, {self._steps} Euler "
                  f"steps, {'CUDA graphs captured' if self.capture else 'eager'}"
                  f"{', TensorRT vision encoder' if self._trt_vit is not None else ''}"
                  f"{', TensorRT prefill' if self._trt_layers is not None else ''} in "
                  f"{time.time() - t0:.1f} s")

    def _load_trt(self) -> None:
        """Engines for the vision encoder and the prefill, if ``trt_dir`` has
        them for this layout.  Anything missing or mismatched leaves the
        PyTorch stage in place."""
        self._trt_vit = self._trt_layers = None
        if not self.trt_dir:
            return
        t0 = time.time()
        try:
            import tensorrt as trt

            with open(os.path.join(self.trt_dir, "manifest.json")) as f:
                manifest = json.load(f)
            if manifest["grid"] != self._grid.flatten().tolist():
                raise ValueError(f"engines were built for image grid {manifest['grid']}")
            self._trt_runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            self._trt_stream = torch.cuda.Stream()
            vit = _TrtEngine(self._engine_file("vit_*.engine"), self._trt_runtime)
            if tuple(vit.inputs["pixels"].shape) != tuple(self._pix.shape):
                raise ValueError("vision engine was built for another image layout")
            self._trt_vit = vit
        except Exception as exc:
            self._log(f"FastAlpamayo: TensorRT not used ({type(exc).__name__}: {str(exc)[:160]})")
            return
        try:
            W = self._W
            if manifest["rows"] != W:
                raise ValueError(f"layer engines were built for a {manifest['rows']}-row window, "
                                 f"this prompt needs {W}")
            cos = self._cos[0, :, :W].float().contiguous()
            sin = self._sin[0, :, :W].float().contiguous()
            layers: List[_TrtEngine] = []
            for i in range(len(self._lm.layers)):
                bind = {"cos": cos, "sin": sin}
                if layers:
                    bind["h"] = layers[-1].outputs["h_out"]
                layers.append(_TrtEngine(
                    self._engine_file(os.path.join("llm", f"layer_{i:02d}_*.engine")),
                    self._trt_runtime, bind))
            self._trt_tables = (cos, sin)
            self._trt_layers = layers
        except Exception as exc:
            self._log(f"FastAlpamayo: TensorRT prefill not used "
                      f"({type(exc).__name__}: {str(exc)[:160]})")
        self._log(f"FastAlpamayo: TensorRT engines ({manifest.get('precision', 'fp16')}) loaded "
                  f"in {time.time() - t0:.0f} s")

    def _engine_file(self, pattern: str) -> str:
        found = sorted(glob.glob(os.path.join(self.trt_dir, pattern)))
        if len(found) != 1:
            raise FileNotFoundError(f"{len(found)} files match {pattern} in {self.trt_dir}")
        return found[0]

    def _vit_trt_fn(self) -> Tuple[torch.Tensor, ...]:
        vit, side = self._trt_vit, self._trt_stream
        current = torch.cuda.current_stream()
        vit.inputs["pixels"].copy_(self._pix)
        side.wait_stream(current)
        vit.run(side)
        current.wait_stream(side)
        return tuple(vit.outputs[f"out{i}"] for i in range(len(vit.outputs)))

    def _prefill_trt_fn(self) -> torch.Tensor:
        """_prefill_fn with each decoder layer as a TensorRT engine; layer i+1
        reads layer i's output buffer in place."""
        lm, W, layers, side = self._lm, self._W, self._trt_layers, self._trt_stream
        image, *deep = self._vit.out
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            h = lm.embed_tokens(self._ids).float()
            layers[0].inputs["h"].copy_(h.index_copy(1, self._img_pos, image.float().unsqueeze(0)))
            for i, engine in enumerate(layers):
                engine.run(side)
                self._kb[i][:, :, :W].copy_(engine.outputs["k"])
                self._vb[i][:, :, :W].copy_(engine.outputs["v"])
                if i < len(deep):
                    engine.outputs["h_out"].index_add_(
                        1, self._img_pos, deep[i].float().unsqueeze(0))
            tail = layers[-1].outputs["h_out"][:, self._L - 1:].to(self._dtype)
            logits = self._mask_traj(self._head(self._rms(lm.norm, tail)))[0]
        current.wait_stream(side)
        return logits

    def _vit_fn(self) -> Tuple[torch.Tensor, ...]:
        vis = self._visual
        n, s = self._n_img, self._img_seq
        x = vis.patch_embed(self._pix) + self._vit_pos
        heads = vis.blocks[0].attn.num_heads
        deep = []
        for i, blk in enumerate(vis.blocks):
            qkv = blk.attn.qkv(blk.norm1(x)).reshape(n * s, 3, heads, -1).permute(1, 0, 2, 3)
            q, k = self._vrope(qkv[0], qkv[1], self._vit_cos, self._vit_sin)
            q, k, v = (t.reshape(n, s, heads, -1).transpose(1, 2) for t in (q, k, qkv[2]))
            a = F.scaled_dot_product_attention(q, k, v, scale=blk.attn.scaling)
            x = x + blk.attn.proj(a.transpose(1, 2).reshape(n * s, -1))
            x = x + blk.mlp(blk.norm2(x))
            if i in vis.deepstack_visual_indexes:
                deep.append(vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)](x))
        return (vis.merger(x), *deep)

    def _rms(self, norm, x: torch.Tensor) -> torch.Tensor:
        """Qwen3VLTextRMSNorm."""
        if x.dtype == _BF16:
            if self._rms_tail is None:
                return norm(x)
            variance = self._square(x).mean(-1, keepdim=True)
            return self._rms_tail(x, torch.rsqrt(variance + norm.variance_epsilon), norm.weight)
        return self._rms32(x, norm.weight, norm.variance_epsilon)

    def _qkv(self, attn, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, s, _ = x.shape
        q = self._rms(attn.q_norm, attn.q_proj(x).view(b, s, -1, self._hd)).transpose(1, 2)
        k = self._rms(attn.k_norm, attn.k_proj(x).view(b, s, -1, self._hd)).transpose(1, 2)
        v = attn.v_proj(x).view(b, s, -1, self._hd).transpose(1, 2)
        rope = self._rope if cos.dtype == _BF16 else self._rope32
        return rope(q, cos, sin), rope(k, cos, sin), v

    def _mlp(self, layer, h: torch.Tensor, down=None) -> torch.Tensor:
        mlp = layer.mlp
        x = self._rms(layer.post_attention_layernorm, h)
        return (down or mlp.down_proj)(self._silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))

    def _init_kernels(self) -> None:
        """torch.compile the elementwise chains into one kernel each and pick
        the faster GEMM layout, keeping only what reproduces the stock op."""
        dev, dt, W = self._dev, self._dtype, self._W
        lm, ex = self._lm, self._expert
        used, dropped = [], []
        g = torch.Generator(device=dev).manual_seed(0)
        rand = lambda *shape, dtype=dt: torch.randn(
            *shape, device=dev, dtype=dtype, generator=g)

        def close(a, b, tol):
            if tol == 0.0:
                return torch.equal(a, b)
            a, b = a.float(), b.float()
            return bool((a - b).abs().max() <= tol * b.abs().max())

        def fused(name, fn, ref, samples, tol=0.0):
            """torch.compile(fn) if it reproduces ref on every sample, else ref."""
            ok = False
            if self.fuse:
                try:
                    compiled = torch.compile(fn, dynamic=True)
                    ok = True
                    for args in samples:
                        got, want = compiled(*args), ref(*args)
                        if not isinstance(got, tuple):
                            got, want = (got,), (want,)
                        ok = ok and all(close(a, b, tol) for a, b in zip(got, want))
                except Exception as exc:
                    self._log(f"FastAlpamayo: {name} not fused "
                              f"({type(exc).__name__}: {str(exc)[:120]})")
                    ok = False
                (used if ok else dropped).append(name)
            return compiled if ok else ref

        heads = lm.layers[0].self_attn.q_proj.out_features // self._hd
        q = rand(1, heads, W, self._hd)
        k1 = rand(1, self._kv_heads, 1, self._hd)
        cos, sin = self._cos[:, :, :W], self._sin[:, :, :W]
        self._rope = fused("rope", _rope_fused, _rope_ref,
                           [(q, cos, sin), (k1, cos[:, :, :1], sin[:, :, :1])])

        mlp0 = lm.layers[0].mlp
        every = torch.arange(1 << 16, device=dev, dtype=torch.int32).to(torch.int16).view(_BF16)
        every = torch.where(torch.isfinite(every), every, torch.zeros_like(every))
        for act in (mlp0.act_fn, ex.layers[0].mlp.act_fn):
            if not torch.equal(act(every), F.silu(every)):
                raise ValueError("FastAlpamayo expects SiLU MLPs")
        inter = mlp0.gate_proj.out_features
        gate = every.repeat(W * inter // every.numel() + 1)[:W * inter].view(1, W, inter)
        self._silu_mul = fused("silu-gate", _silu_mul_fused, _silu_mul_ref,
                               [(gate, torch.ones_like(gate)), (gate, rand(1, W, inter)),
                                (every.view(1, 1, -1), rand(1, 1, every.numel()))])
        del gate

        self._square = self._rms_tail = None
        norm0, qnorm0 = lm.layers[0].input_layernorm, lm.layers[0].self_attn.q_norm
        if self.fuse:
            try:
                self._square = torch.compile(_square_fused, dynamic=True)
                self._rms_tail = torch.compile(_rms_tail_fused, dynamic=True)
                x3, x4 = rand(1, W, norm0.weight.shape[0]) * 20, rand(1, W, heads, self._hd) * 20
                same = (torch.equal(self._rms(norm0, x3), norm0(x3))
                        and torch.equal(self._rms(qnorm0, x4), qnorm0(x4))
                        and torch.equal(self._rms(norm0, x3[:, :1]), norm0(x3[:, :1])))
            except Exception as exc:
                self._log(f"FastAlpamayo: rmsnorm not fused ({type(exc).__name__}: {str(exc)[:120]})")
                same = False
            (used if same else dropped).append("rmsnorm")
            if not same:
                self._square = self._rms_tail = None

        n = self._n_act
        e_heads = ex.layers[0].self_attn.q_proj.out_features // self._hd
        e_norm = ex.layers[0].input_layernorm
        xe = rand(1, n, e_norm.weight.shape[0], dtype=_F32) * 20
        self._rms32 = fused("expert rmsnorm", _rms32_ref, _rms32_ref,
                            [(xe, e_norm.weight, e_norm.variance_epsilon)], tol=1e-5)
        e_cos, e_sin = rand(1, 1, n, self._hd, dtype=_F32), rand(1, 1, n, self._hd, dtype=_F32)
        self._rope32 = fused("expert rope", _rope32_fused, _rope_ref,
                             [(rand(1, e_heads, n, self._hd), e_cos, e_sin),
                              (rand(1, self._kv_heads, n, self._hd), e_cos, e_sin)], tol=1e-5)

        v_heads = self._visual.blocks[0].attn.num_heads
        n_patch = self._n_img * self._img_seq
        vq = rand(n_patch, v_heads, self._vit_cos.shape[-1])
        self._vrope = fused("vit rope", _vrope_fused, apply_rotary_pos_emb_vision,
                            [(vq, rand(*vq.shape), self._vit_cos, self._vit_sin)], tol=2.0 ** -7)

        self._down = [None] * len(lm.layers)
        if self.fuse and self._trt_layers is None:
            x = rand(1, W, mlp0.down_proj.in_features)
            fast = _FastLinear(mlp0.down_proj)
            same = torch.equal(fast(x), mlp0.down_proj(x))
            (used if same else dropped).append("prefill down_proj W^T")
            if same:
                self._down = [fast] + [_FastLinear(layer.mlp.down_proj) for layer in lm.layers[1:]]
        self._elayer = None
        self._eweights = [
            (a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
             l.mlp.gate_proj.weight, l.mlp.up_proj.weight, l.mlp.down_proj.weight,
             l.input_layernorm.weight, l.post_attention_layernorm.weight,
             a.q_norm.weight, a.k_norm.weight)
            for l in ex.layers for a in (l.self_attn,)]
        norms = [m for l in ex.layers for m in (l.input_layernorm, l.post_attention_layernorm,
                                                l.self_attn.q_norm, l.self_attn.k_norm)]
        plain = all(a.bias is None for l in ex.layers for a in (
            l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj, l.self_attn.o_proj,
            l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj))
        if self.fuse and plain and all(m.variance_epsilon == 1e-6 for m in norms):
            try:
                compiled = torch.compile(_expert_layer_fused)
                self._emask.zero_()
                ok = True
                for i in (0, len(ex.layers) // 2):
                    for _ in range(2):
                        hx = rand(1, n, e_norm.weight.shape[0], dtype=_F32)
                        want = self._expert_layer(i, ex.layers[i], hx, e_cos, e_sin)
                        got = compiled(hx, e_cos, e_sin, self._kb[i], self._vb[i], self._emask,
                                       *self._eweights[i])
                        ok = ok and close(got, want, 2.0 ** -6)
            except Exception as exc:
                self._log(f"FastAlpamayo: expert layer not fused "
                          f"({type(exc).__name__}: {str(exc)[:160]})")
                ok = False
            (used if ok else dropped).append("expert layer")
            if ok:
                self._elayer = compiled
        self._log("FastAlpamayo: fused kernels in use: " + (", ".join(used) or "none")
                  + (f"; dropped (do not reproduce the stock op here): {', '.join(dropped)}"
                     if dropped else ""))

    def _prefill_fn(self) -> torch.Tensor:
        """Prompt + draft through the decoder; logits of the rows that predict
        a generated token (the last prompt row and every draft row)."""
        lm, W = self._lm, self._W
        image, *deep = self._vit.out
        h = lm.embed_tokens(self._ids)
        h = h.index_copy(1, self._img_pos, image.to(h.dtype).unsqueeze(0))
        cos, sin = self._cos[:, :, :W], self._sin[:, :, :W]
        for i, layer in enumerate(lm.layers):
            attn = layer.self_attn
            q, k, v = self._qkv(attn, self._rms(layer.input_layernorm, h), cos, sin)
            self._kb[i][:, :, :W].copy_(k)
            self._vb[i][:, :, :W].copy_(v)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scaling,
                                               enable_gqa=True)
            h = h + attn.o_proj(a.transpose(1, 2).reshape(1, W, -1))
            h = h + self._mlp(layer, h, self._down[i])
            if i < len(deep):
                h = h.index_add(1, self._img_pos, deep[i].to(h.dtype).unsqueeze(0))
        return self._mask_traj(self._head(self._rms(lm.norm, h[:, self._L - 1:])))[0]

    def _body_fn(self) -> torch.Tensor:
        """STEP_ROWS tokens at consecutive KV rows ``_rows``; each attends to
        every buffer row up to its own."""
        lm = self._lm
        h = lm.embed_tokens(self._toks)
        cos = self._cos.index_select(2, self._rows)
        sin = self._sin.index_select(2, self._rows)
        mask = torch.zeros(STEP_ROWS, self._T, device=self._dev, dtype=self._dtype).masked_fill(
            self._cols[None, :] > self._rows[:, None], _NEG)[None, None]
        for i, layer in enumerate(lm.layers):
            attn = layer.self_attn
            q, k, v = self._qkv(attn, self._rms(layer.input_layernorm, h), cos, sin)
            self._kb[i].index_copy_(2, self._rows, k)
            self._vb[i].index_copy_(2, self._rows, v)
            a = _attend(q, self._kb[i], self._vb[i], mask, attn.scaling)
            h = h + attn.o_proj(a.transpose(1, 2).reshape(1, STEP_ROWS, -1))
            h = h + self._mlp(layer, h)
        return h

    def _logits_fn(self) -> torch.Tensor:
        return self._mask_traj(self._head(self._rms(self._lm.norm, self._body.out)))[0]

    def _mask_traj(self, logits: torch.Tensor) -> torch.Tensor:
        """ExpertLogitsProcessor: the discrete trajectory tokens are never generated."""
        logits = logits.float()
        logits[..., self._traj_lo:self._traj_hi] = _NEG
        return logits

    def _expert_fn(self) -> torch.Tensor:
        """The Euler loop of FlowMatching._euler with step_fn inlined.  The
        expert's own keys/values live in the last ``n_act`` rows of the buffer."""
        m, ex = self.model, self._expert
        pos = (self._act_range + self._eoff).view(1, 1, -1).expand(3, 1, -1)
        x = self._noise
        cos = sin = None
        for step in range(self._steps):
            t = self._times[step].view(1, 1, 1)
            dt = (self._times[step + 1] - self._times[step]).view(1, 1, 1)
            h = m.action_in_proj(x, t)
            if cos is None:
                cos, sin = (t.unsqueeze(1) for t in ex.rotary_emb(h, pos))
            for i, layer in enumerate(ex.layers):
                if self._elayer is not None:
                    h = self._elayer(h, cos, sin, self._kb[i], self._vb[i], self._emask,
                                     *self._eweights[i])
                else:
                    h = self._expert_layer(i, layer, h, cos, sin)
            v = m.action_out_proj(self._rms(ex.norm, h)).view(-1, self._n_act, self._act_dim)
            x = x + dt * v
        return x

    def _expert_layer(self, i: int, layer, h: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
        attn = layer.self_attn
        q, k, v = self._qkv(attn, self._rms(layer.input_layernorm, h), cos, sin)
        self._kb[i][:, :, self._C:].copy_(k)
        self._vb[i][:, :, self._C:].copy_(v)
        a = _attend(q, self._kb[i], self._vb[i], self._emask, attn.scaling)
        h = h + attn.o_proj(a.transpose(1, 2).reshape(1, self._n_act, -1))
        return h + self._mlp(layer, h)

    def _probs(self, logits: torch.Tensor, temperature: float, top_p: float,
               top_k: Optional[int]) -> torch.Tensor:
        """generate()'s warpers: temperature, top-k, top-p, in that order."""
        x = logits / temperature
        if top_k:
            kth = torch.topk(x, min(top_k, x.shape[-1]), dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, _NEG)
        if top_p < 1.0:
            srt, idx = torch.sort(x, descending=False, dim=-1)
            remove = srt.softmax(-1).cumsum(-1) <= (1.0 - top_p)
            remove[..., -1:] = False
            x = x.masked_fill(remove.scatter(-1, idx, remove), _NEG)
        return x.softmax(-1)

    def _pick(self, logits: torch.Tensor, proposal: List[int], greedy: bool, temperature: float,
              top_p: float, top_k: Optional[int]) -> List[int]:
        """One token per logits row.  Row j has proposal[j] as a candidate when
        j < len(proposal).  Greedy: the candidate is kept if it is the best
        token (within tie_margin).  Sampling: it is kept with its model
        probability and otherwise replaced by a draw from the rest of the
        distribution, which samples exactly the model's distribution."""
        n = len(proposal)
        d = torch.tensor(proposal, device=logits.device, dtype=torch.long).unsqueeze(1)
        if greedy:
            best, out = logits.max(-1, keepdim=True)
            if n and self.tie_margin > 0:
                near = logits[:n].gather(1, d) >= best[:n] - self.tie_margin
                out = torch.cat([torch.where(near, d, out[:n]), out[n:]])
            return out[:, 0].tolist()
        p = self._probs(logits, temperature, top_p, top_k)
        if n:
            keep = torch.rand(n, 1, device=p.device) < p[:n].gather(1, d)
            rest = p[:n].scatter(1, d, 0.0)
            rest = torch.where(rest.sum(-1, keepdim=True) > 0, rest, p[:n])
            out = torch.where(keep, d, torch.multinomial(rest, 1))
            if p.shape[0] > n:
                out = torch.cat([out, torch.multinomial(p[n:], 1)])
            return out[:, 0].tolist()
        return torch.multinomial(p, 1)[:, 0].tolist()

    def _lookup(self, gen: List[int]) -> List[int]:
        """Proposed continuation of ``gen``: what followed its last tokens in a
        remembered trace (longest matching suffix first, newest trace first)."""
        best: List[int] = []
        best_n = 0
        for trace in self._library:
            for n in (3, 2, 1):
                if n <= best_n or n > len(gen):
                    continue
                suffix = gen[-n:]
                for i in range(len(trace) - n - 1, -1, -1):
                    if trace[i:i + n] == suffix:
                        best, best_n = trace[i + n:], n
                        break
                if best_n == n:
                    break
        return best[:STEP_ROWS - 1]

    def _settle(self, gen: List[int], fed: int, logits: torch.Tensor, proposal: List[int],
                max_new: int, choice: Tuple) -> int:
        """Append the tokens these logits rows decide: row 0 always, row j+1
        only while proposal[j] was the token chosen at row j (its KV row is
        then valid).  Returns the new count of generated tokens with valid rows."""
        picks = self._pick(logits[:len(proposal) + 1], proposal, *choice)
        for j, g in enumerate(picks):
            gen.append(g)
            if self.trace_logits is not None:
                self.trace_logits.append(logits[j].clone())
            if j < len(proposal) and g == proposal[j] and len(gen) < max_new:
                fed += 1
                if g == self._eos:
                    break
            else:
                break
        return fed

    def _tick(self, name: str, t0: float) -> float:
        if not self.profile:
            return t0
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.timing[name] = self.timing.get(name, 0.0) + (now - t0) * 1e3
        return now

    def covers(self, data: Dict[str, Any], num_traj_samples: int = 1, num_traj_sets: int = 1) -> bool:
        tok = data["tokenized_data"]
        ids = tok["input_ids"]
        mask = tok.get("attention_mask")
        return bool(
            num_traj_samples == 1 and num_traj_sets == 1
            and ids.shape[0] == 1
            and tok.get("pixel_values") is not None
            and tok.get("pixel_values_videos") is None
            and (mask is None or bool(mask.all()))
            and data["ego_history_xyz"].shape[1] == 1
        )

    @torch.no_grad()
    def sample(self, data: Dict[str, Any], greedy: bool = True, top_p: float = 0.98,
               top_k: Optional[int] = None, temperature: float = 0.6,
               max_generation_length: Optional[int] = None,
               noise: Optional[torch.Tensor] = None, return_extra: bool = True):
        """sample_trajectories_from_data_with_vlm_rollout for one sample.
        Returns None for an input the fast path does not cover."""
        if not self.covers(data):
            self.fallbacks += 1
            return None
        m = self.model
        max_new = min(self.max_new, int(max_generation_length or self.max_new))
        self.timing = {}
        t0 = time.perf_counter()
        tok = data["tokenized_data"]
        hist_xyz, hist_rot = data["ego_history_xyz"], data["ego_history_rot"]
        ids = m.fuse_traj_tokens(
            tok["input_ids"], {"ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot})
        grid, pixels = tok["image_grid_thw"], tok["pixel_values"]
        layout = (ids.shape[1], tuple(grid.flatten().tolist()))
        if (layout != self._layout
                or self._steps != int(m.diffusion.num_inference_steps)
                or not torch.equal(ids[0] == self._image_token, self._img_mask)):
            self._build(ids, grid, pixels)
            t0 = time.perf_counter()
        L, C = self._L, self._C

        draft: List[int] = []
        if self.speculate and self._library:
            draft = self._library[0][:min(DRAFT_CAP, max_new)]
        choice = (greedy, temperature, top_p, top_k)
        self._ids[0, :L] = ids[0]
        self._ids[0, L:] = self._pad
        if draft:
            self._ids[0, L:L + len(draft)] = torch.tensor(draft, device=self._dev)
        self._pix.copy_(pixels)
        t0 = self._tick("prepare", t0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._vit()
            t0 = self._tick("vit", t0)
            logits = self._prefill()
            t0 = self._tick("prefill", t0)

            gen: List[int] = []
            fed = self._settle(gen, 0, logits, draft, max_new, choice)
            accepted = fed
            t0 = self._tick("verify", t0)

            def finished() -> bool:
                if self._eos in gen:
                    e = gen.index(self._eos)
                    return fed > e or e == max_new - 1
                return len(gen) >= max_new

            steps = 0
            while not finished():
                token = gen[fed]
                proposal = self._lookup(gen) if self.speculate and token != self._eos else []
                self._toks.fill_(self._pad)
                self._toks[0, :1 + len(proposal)] = torch.tensor([token] + proposal, device=self._dev)
                torch.add(self._step_range, L + fed, out=self._rows)
                self._body()
                fed += 1
                steps += 1
                if token != self._eos:
                    fed = self._settle(gen, fed, self._logits(), proposal, max_new, choice)
            t0 = self._tick("decode", t0)

            has_eos = self._eos in gen
            n_gen = gen.index(self._eos) + 1 if has_eos else len(gen)
            if has_eos:
                trace = gen[:n_gen]
                self._library = ([trace] + [t for t in self._library if t != trace])[:LIBRARY_SIZE]
            self._emask.fill_(_NEG)
            self._emask[..., :L + fed] = 0
            self._emask[..., C:] = 0
            self._eoff.fill_(L + n_gen + self._delta)
            if noise is None:
                noise = torch.randn(1, self._n_act, self._act_dim, device=self._dev)
            self._noise.copy_(noise)
            action = self._plan()
            t0 = self._tick("expert", t0)

            pred_xyz, pred_rot = m.action_space.action_to_traj(
                action, hist_xyz[:, -1], hist_rot[:, -1])
        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
        self._tick("to_traj", t0)
        self.stats = {"tokens": n_gen, "accepted": accepted, "decode_steps": steps,
                      "draft": len(draft)}
        self.tokens = gen[:n_gen]
        if not return_extra:
            return pred_xyz, pred_rot
        seq = torch.cat([ids, torch.tensor([gen[:n_gen]], device=ids.device)], dim=1)
        extra = extract_text_tokens(m.tokenizer, seq)
        extra = {k: np.array(v).reshape(1, 1, 1) for k, v in extra.items()}
        return pred_xyz, pred_rot, extra

    def reset(self) -> None:
        """Forget the remembered reasoning traces."""
        self._library = []

    @torch.no_grad()
    def verify(self, data: Dict[str, Any], seed: int = 0,
               max_generation_length: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Reference and fast path on one input, greedy tokens, same noise.
        The fast path runs twice: without a draft and with its own output as
        the draft."""
        if not self.covers(data):
            return None
        m = self.model
        max_new = int(max_generation_length or self.max_new)
        sample = m.diffusion.sample

        def seeded(*args, **kwargs):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            return sample(*args, **kwargs)

        m.diffusion.sample = seeded
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref_xyz, _, ref_extra = m.sample_trajectories_from_data_with_vlm_rollout(
                    data=data, top_p=1.0, temperature=1.0, top_k=1, num_traj_samples=1,
                    max_generation_length=max_new, return_extra=True)
        finally:
            del m.diffusion.sample
        report: Dict[str, Any] = {"cot": str(ref_extra["cot"].reshape(-1)[0])}
        self.reset()
        for name in ("cold", "draft"):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            noise = torch.randn(1, self._n_act, self._act_dim, device=self._dev)
            xyz, _, extra = self.sample(data, greedy=True, max_generation_length=max_new, noise=noise)
            report[f"{name}_text_equal"] = bool(
                str(extra["cot"].reshape(-1)[0]) == report["cot"])
            report[f"{name}_max_dev_m"] = float((xyz.float() - ref_xyz.float()).abs().max())
            report[f"{name}_stats"] = dict(self.stats)
        return report
