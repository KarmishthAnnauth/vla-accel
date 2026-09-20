"""Inference-time speedups for MindDrive, applied to the *built model object*.

Nothing here edits MindDrive sources: every function is a post-build transform
on the model the node already has.  Each one is either exact (same math, fewer
launches) or a precision change, and tools/bench_minddrive.py measures both the
time and the trajectory / decision delta of every step against the untouched
model, so the node only turns on what was checked.

  merge_lora_experts()      fold BOTH LoRA adapter sets into the LLM weights
                            (one merged copy per expert, switched by
                            set_adapter -- MindDrive alternates the decision
                            expert and the action expert every frame)
  set_llm_attention()       swap the Qwen2 attention class (sdpa / flash_attention_2)
  patch_vit_blocks()        fused qkv + SDPA + fused w12 in the EVA-ViT blocks
  compile_submodules()      torch.compile on the ViT, the two expert LLMs, the
                            PETR stacks of the heads
  graph_vit()               manual CUDA-graph capture of the (static-shape) ViT
  slice_map_head_one2one()  drop the 1500 one-to-many lane queries (exact)
  transpose_llm_down_proj() [K,N]-contiguous down_proj weight (fp16, exact)
  transpose_vit_linears()   same layout trick for the EVA-ViT qkv/proj/w3 (exact)
  patch_vit_window_blocks() qkv/proj on the unpadded 40x40 grid in the 16
                            window-attention blocks (exact)
  quantize_linears_int8()   W8A8 int8 linears on torch._int_mm (cuBLASLt int8
                            tensor cores): per-channel weights, per-token
                            dynamic activations; a precision change
  set_vit_input_size() / set_pipeline_input_size()   512x512 ViT input (ORION's
                            "lite vision"; a numerics change)
  install_staggered_views() rear cameras through the ViT every N frames
  StageTimer                CUDA-event timing per stage of the forward
  ParallelDecoder / parallelize_pipeline   the CPU prep on a thread pool (bit-exact)

MindDrive's ViT (EVAViT), map head and mmcv pipeline transforms are the same
code as ORION's, so those helpers are ORION's (orion_ros/orion_speedups.py)
verbatim; the LLM helpers are new because the LLM is a Qwen2 under
transformers 4.45 with two adapters, not a LLaMA under 4.31 with one.
"""
from __future__ import annotations

import copy
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

Log = Optional[Callable[[str], None]]

EXPERTS = ("action_expert", "decision_expert")


class MergedExperts:
    """Stands in for the peft-wrapped `model.lm_head` after both adapters have
    been folded into the weights.

    peft can merge only the ACTIVE adapter into a base weight, and MindDrive
    needs both experts every frame (decision expert -> meta-action logits,
    action expert -> waypoint features), so one merged model per expert is
    kept and `set_adapter` picks which one answers.  Everything the detector
    calls on `lm_head` (config, get_model, inference_action_distribution,
    inference_waypoints, generate, resize_token_embeddings ...) is delegated
    to the active expert; the two share one `config` object."""

    def __init__(self, experts: Dict[str, torch.nn.Module], active: str) -> None:
        self.experts = experts
        self._active = active

    def set_adapter(self, name: str) -> None:
        if name not in self.experts:
            raise KeyError(f"unknown adapter {name!r}; have {list(self.experts)}")
        self._active = name

    @property
    def active_adapter(self) -> str:
        return self._active

    @property
    def active(self) -> torch.nn.Module:
        return self.experts[self._active]

    def __getattr__(self, name: str):
        if name in ("experts", "_active"):
            raise AttributeError(name)
        return getattr(self.experts[self._active], name)

    def modules(self):
        for m in self.experts.values():
            yield from m.modules()

    def parameters(self):
        for m in self.experts.values():
            yield from m.parameters()


def merge_lora_experts(model, log: Log = None) -> bool:
    """Fold the two LoRA adapter sets (q/k/v/o, r=16) into two merged copies of
    the LLM and install a MergedExperts dispatcher as `model.lm_head`.

    Memory: one extra LLM (6.2 GB in fp16 for the 3B) on the GPU.  The peft
    original is dropped afterwards.  The merged copies are plain
    LlavaQwen2ForCausalLM objects, so torch.compile / attention swaps see no
    peft in the graph."""
    lm = getattr(model, "lm_head", None)
    base = getattr(lm, "base_model", None)
    if base is None or not hasattr(base, "merge_and_unload"):
        if log:
            log("merge_lora: lm_head is not a peft model, nothing to merge")
        return False
    if isinstance(lm, MergedExperts):
        return True
    cfg = lm.config
    names = [n for n in EXPERTS if n in getattr(base, "peft_config", {})] or list(base.peft_config)
    active_before = lm.active_adapter if isinstance(lm.active_adapter, str) else names[0]
    merged: Dict[str, torch.nn.Module] = {}
    for i, name in enumerate(names):
        lm.set_adapter(name)
        src = base if i == len(names) - 1 else copy.deepcopy(base)
        m = src.merge_and_unload()
        for attr in ("waypoint_token_idx", "meta_action_token_idx"):
            if hasattr(cfg, attr) and not hasattr(m.config, attr):
                setattr(m.config, attr, getattr(cfg, attr))
        m.eval()
        merged[name] = m
        if log:
            log(f"merge_lora: adapter {name!r} folded into its own LLM copy")
    if "lm_head" in model._modules:
        del model._modules["lm_head"]
    model.lm_head = MergedExperts(merged, active_before)
    torch.cuda.empty_cache()
    if log:
        log(f"merge_lora: {len(merged)} merged experts installed "
            f"(GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
    return True


def _expert_models(model) -> List[torch.nn.Module]:
    """The LlavaQwen2ForCausalLM objects the detector can run: the merged
    experts, or the single peft-wrapped base model before merging."""
    lm = getattr(model, "lm_head", None)
    if isinstance(lm, MergedExperts):
        return list(lm.experts.values())
    base = getattr(lm, "base_model", None)
    inner = getattr(base, "model", None)
    return [inner] if inner is not None else ([lm] if lm is not None else [])


def set_llm_attention(model, impl: str, log: Log = None) -> int:
    """Rebuild every Qwen2 decoder layer's self_attn as the requested class
    ('sdpa', 'flash_attention_2', 'eager') with the same weights.

    transformers picks the class at construction from config._attn_implementation
    (sdpa by default here).  Both MindDrive prefills are batch 1 with an
    all-True mask, so under sdpa the model already drops the mask and passes
    is_causal=True; flash_attention_2 goes through flash_attn_func directly
    (fp16 only).  Same math either way; which kernel wins is measured."""
    try:
        from transformers.models.qwen2 import modeling_qwen2 as Q
    except Exception as e:
        if log:
            log(f"llm attention: transformers qwen2 not importable ({e})")
        return 0
    cls = Q.QWEN2_ATTENTION_CLASSES.get(impl)
    if cls is None:
        if log:
            log(f"llm attention: unknown impl {impl!r}; known {list(Q.QWEN2_ATTENTION_CLASSES)}")
        return 0
    n = 0
    for m in _expert_models(model):
        inner = m.get_model() if hasattr(m, "get_model") else m
        layers = getattr(inner, "layers", None)
        if layers is None:
            continue
        p = next(inner.parameters())
        if impl == "flash_attention_2" and p.dtype not in (torch.float16, torch.bfloat16):
            if log:
                log("llm attention: flash_attention_2 needs fp16/bf16; keeping the current class")
            return 0
        for layer in layers:
            old = layer.self_attn
            if isinstance(old, cls):
                continue
            new = cls(old.config, old.layer_idx).to(p.device, p.dtype)
            new.load_state_dict(old.state_dict())
            new.eval()
            layer.self_attn = new
            n += 1
        m.config._attn_implementation = impl
        inner.config._attn_implementation = impl
        if hasattr(inner, "_attn_implementation"):
            inner._attn_implementation = impl
    if log:
        log(f"llm attention: {n} layers -> {impl}")
    return n


COMPILE_TARGETS = ("vit", "llm", "heads")
HEAD_SUBMODULES = (
    ("pts_bbox_head", "transformer"),
    ("pts_bbox_head", "memory_decoder_mq"),
    ("pts_bbox_head", "memory_decoder_cq"),
    ("pts_bbox_head", "motion_decoder"),
    ("map_head", "transformer"),
)


def _compile_forward(module: torch.nn.Module, mode: str) -> None:
    if not hasattr(module, "_md_eager_forward"):
        module._md_eager_forward = module.forward
    module.forward = torch.compile(module._md_eager_forward, mode=mode)


def uncompile_submodules(model) -> int:
    n = 0
    mods = list(model.modules())
    for e in _expert_models(model):
        mods += list(e.modules())
    for m in mods:
        if hasattr(m, "_md_eager_forward"):
            m.forward = m._md_eager_forward
            del m._md_eager_forward
            n += 1
    torch._dynamo.reset()
    return n


def compile_submodules(model, targets: Iterable[str], mode: str = "default",
                       log: Log = None) -> List[str]:
    """'vit'   -> model.img_backbone (static 6x3x640x640)
       'llm'   -> the Qwen2Model forward of every expert (merge LoRA first)
       'heads' -> the PETR stacks inside pts_bbox_head / map_head, always in
                  mode='default' (see cudagraph notes in graph_heads)."""
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.automatic_dynamic_shapes = False
    done: List[str] = []
    for t in targets:
        try:
            if t == "vit":
                _compile_forward(model.img_backbone, mode)
            elif t == "llm":
                n = 0
                for e in _expert_models(model):
                    _compile_forward(e.get_model(), mode)
                    n += 1
                if log:
                    log(f"compile: llm -> {n} expert Qwen2Model forwards")
            elif t == "heads":
                n = 0
                for head_name, sub_name in HEAD_SUBMODULES:
                    head = getattr(model, head_name, None)
                    sub = getattr(head, sub_name, None) if head is not None else None
                    if isinstance(sub, torch.nn.Module):
                        _compile_forward(sub, "default")
                        n += 1
                if log:
                    log(f"compile: heads -> {n} transformer stacks (mode='default')")
            else:
                if log:
                    log(f"compile: unknown target {t!r} (known: {COMPILE_TARGETS})")
                continue
            done.append(t)
            if log:
                log(f"compile: {t} wrapped with torch.compile(mode={mode!r})")
        except Exception as e:
            if log:
                log(f"compile: {t} failed, staying eager ({type(e).__name__}: {e})")
    return done


class CudaGraphed:
    """Manual CUDA-graph capture of a function with ONE static-shape tensor
    input.  Captured on the first call (after `warmup` eager runs on a side
    stream, as the CUDA-graph docs prescribe); later calls copy the input into
    the static buffer, replay, and return a clone of the static output so the
    caller may keep it past the next replay.

    Used for the ViT: 6x3x640x640 in, one tensor out, no Python-side state.
    Works on the eager forward and on an inductor-compiled one (inductor
    kernels are capture-safe; only the first call must not be inside capture)."""

    def __init__(self, fn: Callable, warmup: int = 2, log: Log = None) -> None:
        self._fn = fn
        self._warmup = warmup
        self._log = log
        self._graphs: Dict[Tuple, Tuple[torch.cuda.CUDAGraph, torch.Tensor, object]] = {}

    def _capture(self, x: torch.Tensor) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor, object]:
        static_in = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._warmup):
                self._fn(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = self._fn(static_in)
        if self._log:
            self._log(f"cuda graph: captured forward for {tuple(x.shape)} {x.dtype} ({len(self._graphs) + 1} shapes)")
        return g, static_in, static_out

    def __call__(self, x: torch.Tensor):
        key = (tuple(x.shape), x.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._graphs[key] = self._capture(x)
        graph, static_in, out = entry
        static_in.copy_(x)
        graph.replay()
        if isinstance(out, torch.Tensor):
            return out.clone()
        if isinstance(out, (list, tuple)):
            return type(out)(o.clone() if isinstance(o, torch.Tensor) else o for o in out)
        return out


def graph_vit(model, log: Log = None) -> bool:
    """Run the image backbone through a captured CUDA graph.  Applied AFTER
    any compile of the ViT, so the graph replays the compiled kernels."""
    vit = getattr(model, "img_backbone", None)
    if vit is None or isinstance(getattr(vit, "forward", None), CudaGraphed):
        return False
    if not hasattr(vit, "_md_pregraph_forward"):
        vit._md_pregraph_forward = vit.forward
    vit.forward = CudaGraphed(vit._md_pregraph_forward, log=log)
    if log:
        log("cuda graph: ViT forward will be captured on the next call")
    return True


def ungraph_vit(model) -> bool:
    vit = getattr(model, "img_backbone", None)
    if vit is not None and hasattr(vit, "_md_pregraph_forward"):
        vit.forward = vit._md_pregraph_forward
        del vit._md_pregraph_forward
        return True
    return False


def graph_heads(model, log: Log = None) -> int:
    """torch.compile(mode='reduce-overhead') on the PETR stacks, i.e. inductor
    plus CUDA-graph trees, with every output cloned on the way out.

    The same stack object is invoked several times per forward (the det
    transformer for the queries, the memory decoders, the motion decoder), and
    cudagraph trees hand back tensors that the NEXT replay overwrites; the
    clone is what makes that safe.  torch.compiler.cudagraph_mark_step_begin()
    must be called once per model forward (see StageTimer / the node)."""
    n = 0
    for head_name, sub_name in HEAD_SUBMODULES:
        head = getattr(model, head_name, None)
        sub = getattr(head, sub_name, None) if head is not None else None
        if not isinstance(sub, torch.nn.Module):
            continue
        if not hasattr(sub, "_md_eager_forward"):
            sub._md_eager_forward = sub.forward
        compiled = torch.compile(sub._md_eager_forward, mode="reduce-overhead")

        def fwd(*a, _c=compiled, **kw):
            out = _c(*a, **kw)
            return _clone_tree(out)

        sub.forward = fwd
        n += 1
    if log:
        log(f"cuda graph: {n} head stacks compiled with mode='reduce-overhead' (outputs cloned)")
    return n


def _clone_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_tree(o) for o in obj)
    if isinstance(obj, list):
        return [_clone_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _clone_tree(v) for k, v in obj.items()}
    return obj


def mark_step() -> None:
    """Once per model forward when graph_heads / reduce-overhead is in use."""
    try:
        torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        pass


def patch_vit_blocks(model, log: Log = None) -> int:
    """Rewrite the EVA-ViT Attention and SwiGLU forwards with the same weights:
      * q/k/v projections -> one [3C, C] GEMM (bias = [q_bias, 0, v_bias]);
      * flash_attn's kv-stack + permutes -> F.scaled_dot_product_attention on
        (B, heads, N, d) views (same 1/sqrt(d) scale, no mask);
      * SwiGLU w1/w2 -> one [2H, C] GEMM, then silu(x1) * x2.
    Same math, fewer launches and copies, and a graph inductor fuses better."""
    import torch.nn.functional as F

    vit = getattr(model, "img_backbone", None)
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        return 0
    n = 0
    for blk in blocks:
        attn, mlp = blk.attn, blk.mlp
        if getattr(attn, "_md_glue", False):
            continue
        C = attn.q_proj.weight.shape[1]
        dev, dt = attn.q_proj.weight.device, attn.q_proj.weight.dtype
        qkv = torch.nn.Linear(C, 3 * attn.q_proj.weight.shape[0], bias=True).to(dev, dt)
        with torch.no_grad():
            qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0))
            qkv.bias.zero_()
            if attn.q_bias is not None:
                qkv.bias[:C].copy_(attn.q_bias)
                qkv.bias[2 * C:].copy_(attn.v_bias)
        attn.qkv = qkv
        heads = attn.num_heads

        def attn_forward(self, x, _heads=heads):
            B, H, W, Cc = x.shape
            N = H * W
            x = x.view(B, N, Cc)
            q, k, v = self.qkv(x).view(B, N, 3, _heads, -1).permute(2, 0, 3, 1, 4)
            q = self.rope(q).type_as(v)
            k = self.rope(k).type_as(v)
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            return x.view(B, H, W, Cc)

        attn._md_glue_forward = attn.forward
        attn.forward = types.MethodType(attn_forward, attn)
        attn._md_glue = True

        if hasattr(mlp, "w1") and hasattr(mlp, "w2"):
            Hd, Ci = mlp.w1.weight.shape
            w12 = torch.nn.Linear(Ci, 2 * Hd, bias=mlp.w1.bias is not None).to(dev, dt)
            with torch.no_grad():
                w12.weight.copy_(torch.cat([mlp.w1.weight, mlp.w2.weight], 0))
                if w12.bias is not None:
                    w12.bias.copy_(torch.cat([mlp.w1.bias, mlp.w2.bias], 0))
            mlp.w12 = w12

            def mlp_forward(self, x):
                x1, x2 = self.w12(x).chunk(2, dim=-1)
                x = self.ffn_ln(self.act(x1) * x2)
                return self.drop(self.w3(x))

            mlp._md_glue_forward = mlp.forward
            mlp.forward = types.MethodType(mlp_forward, mlp)
        n += 1
    if log:
        log(f"vit glue: {n} blocks -> fused qkv + SDPA + fused w12")
    return n


def slice_map_head_one2one(model, log: Log = None) -> bool:
    """MinddriveHeadM carries 1800 lane queries (300 one-to-one + 1500
    one-to-many, the H-DETR hybrid-matching training trick).  Its own
    self-attention mask forbids any attention between the one-to-many block
    and the [VLM tokens + one-to-one] block in both directions, cross-attention
    is per-query, and the memory / decoding / VLM tokens come from the
    one-to-one slice only, so dropping the masked-out keys leaves every softmax
    over the same set.  Exact by construction; checked by the bench."""
    head = getattr(model, "map_head", None)
    if head is None or getattr(head, "_md_sliced", False):
        return False
    n1 = int(getattr(head, "num_lanes_one2one", 0))
    n_all = int(head.num_lane)
    if n1 <= 0 or n1 >= n_all:
        if log:
            log(f"map head slice: nothing to do (num_lane={n_all}, one2one={n1})")
        return False
    emb = head.instance_embedding_lane
    new = torch.nn.Embedding(n1, emb.embedding_dim).to(emb.weight.device, emb.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(emb.weight[:n1])
    head.instance_embedding_lane = new
    head.num_lane = n1
    head._md_sliced = True
    if log:
        log(f"map head slice: {n_all} -> {n1} lane queries (one-to-many dropped)")
    return True


def _transpose_linear(lin: torch.nn.Linear) -> bool:
    """Run an nn.Linear as x @ W^T with W^T stored [K,N] contiguous.  Same
    numbers; cuBLAS picks a different (for some shapes much faster) kernel."""
    if not isinstance(lin, torch.nn.Linear) or hasattr(lin, "_md_wt") or hasattr(lin, "_md_w8"):
        return False
    lin._md_wt = lin.weight.detach().t().contiguous()

    def fwd(self, x):
        if self.bias is None:
            return torch.matmul(x, self._md_wt)
        y = torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self._md_wt)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    lin.forward = types.MethodType(fwd, lin)
    return True


def transpose_llm_down_proj(model, log: Log = None) -> int:
    """cuBLAS runs the K=11008 down projection faster with the weight stored
    [K,N] contiguous (measured 30 % on ORION's LLaMA; same GEMM shape family
    here).  Same numbers.  Applied to every expert."""
    n = 0
    for e in _expert_models(model):
        inner = e.get_model() if hasattr(e, "get_model") else e
        for layer in getattr(inner, "layers", []):
            n += int(_transpose_linear(layer.mlp.down_proj))
    if log:
        log(f"down_proj transposed weights: {n} layers")
    return n


VIT_T_LINEARS = ("attn.qkv", "attn.proj", "mlp.w3")


def transpose_vit_linears(model, names: Iterable[str] = VIT_T_LINEARS, log: Log = None) -> int:
    """The same layout trick for the EVA-ViT GEMMs (M = 6 x 1600 tokens).
    Measured per block on the Orin, fp16 ms [N,K] -> [K,N]: qkv 3.65 -> 2.08,
    proj 0.72 -> 0.70, w3 2.93 -> 2.37; w12 (1024 -> 5460) goes the other way
    (4.92 -> 5.69) and is left alone.  Same numbers.  Needs patch_vit_blocks
    first: qkv exists only after the glue rewrite."""
    blocks = getattr(getattr(model, "img_backbone", None), "blocks", None)
    if blocks is None:
        return 0
    n = 0
    for blk in blocks:
        for name in names:
            obj = blk
            for part in name.split("."):
                obj = getattr(obj, part, None)
            n += int(obj is not None and _transpose_linear(obj))
    if log:
        log(f"vit transposed weights: {n} linears ({', '.join(names)}) over {len(blocks)} blocks")
    return n


def patch_vit_window_blocks(model, log: Log = None) -> int:
    """Window-attention blocks without the padding overhead.  16 of the 24
    EVA-ViT blocks attend inside 16x16 windows; the 40x40 token grid is padded
    to 48x48 for them, and the reference block runs the qkv and proj GEMMs
    (and the rope / reshape glue) on the padded 2304 tokens per view instead of
    1600.  Both projections are per-token, so here qkv runs on the unpadded
    grid, q/k/v are padded afterwards with exactly what the padded path
    produces for a zero token (the qkv bias: q_bias, 0, v_bias -- pad keys stay
    zero and pad values stay v_bias, so the attention sees the same keys and
    values), the attention runs per window as before, and proj runs on the
    real tokens after unpartition.  Same math, ~30 % fewer GEMM rows in those
    blocks.  Needs patch_vit_blocks (uses attn.qkv); apply before compile."""
    import torch.nn.functional as F

    blocks = getattr(getattr(model, "img_backbone", None), "blocks", None)
    if blocks is None:
        return 0
    n = 0
    for blk in blocks:
        attn = blk.attn
        if getattr(blk, "window_size", 0) <= 0 or not getattr(attn, "_md_glue", False) or getattr(blk, "_md_nopad", False):
            continue
        heads = attn.num_heads

        def _forward(self, x, _heads=heads):
            shortcut = x
            x = self.norm1(x)
            B, H, W, C = x.shape
            ws = self.window_size
            qkv = self.attn.qkv(x)
            pad_h, pad_w = (ws - H % ws) % ws, (ws - W % ws) % ws
            Hp, Wp = H + pad_h, W + pad_w
            if pad_h or pad_w:
                qkv = F.pad(qkv, (0, 0, 0, pad_w, 0, pad_h))
                keep = torch.zeros(Hp, Wp, dtype=torch.bool, device=x.device)
                keep[:H, :W] = True
                qkv = torch.where(keep[None, :, :, None], qkv, self.attn.qkv.bias)
            nh, nw = Hp // ws, Wp // ws
            qkv = qkv.view(B, nh, ws, nw, ws, 3 * C).permute(0, 1, 3, 2, 4, 5).reshape(B * nh * nw, ws * ws, 3 * C)
            q, k, v = qkv.view(B * nh * nw, ws * ws, 3, _heads, -1).permute(2, 0, 3, 1, 4)
            q = self.attn.rope(q).type_as(v)
            k = self.attn.rope(k).type_as(v)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(B, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
            if pad_h or pad_w:
                a = a[:, :H, :W, :]
            a = self.attn.proj(self.attn.inner_attn_ln(a))
            x = shortcut + self.drop_path(a)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if getattr(self, "use_residual_block", False):
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            return x

        blk._md_pre_nopad_forward = blk._forward
        blk._forward = types.MethodType(_forward, blk)
        blk._md_nopad = True
        n += 1
    if log:
        log(f"vit window blocks without padding: {n} blocks")
    return n


def _quantize_linear_int8(lin: torch.nn.Linear, act_max: Optional[torch.Tensor] = None,
                          alpha: float = 0.5, mode: str = "w8a8") -> bool:
    """Replace an nn.Linear forward with per-channel int8 weights x per-token
    dynamic int8 activations on torch._int_mm (cuBLASLt int8 tensor cores,
    2x the fp16 rate on the Orin's SM 8.7).  The int32 product is rescaled
    by (row scale x column scale) in fp32 and cast back to the input dtype.

    torch._int_mm wants K and N multiples of 8 and M > 16, so weights are
    zero-padded to [Np,Kp] once and rows are padded per call only when M is
    tiny.  The weight is passed as `w8.t()` (a view of the [N,K] storage):
    measured 4x faster than a [K,N]-contiguous int8 weight.  The fp16 weight
    is freed (replaced by an empty tensor) so a 3B expert drops from 6 to 3 GB.
    Compile the owner afterwards: eager, the quant/rescale passes are
    memory-bound elementwise kernels that eat most of the GEMM gain; inductor
    fuses them into the neighbouring ops.

    `act_max` ([K] per-input-channel |x| maxima from calibration forwards)
    turns on SmoothQuant: channel j of the input is divided by
    s_j = act_max_j^alpha / wmax_j^(1-alpha) at run time and column j of the
    weight is multiplied by s_j before quantisation -- same product, but the
    activation outlier channels (Qwen2.5 has them) no longer set the
    per-token scale for the whole row."""
    if not isinstance(lin, torch.nn.Linear) or hasattr(lin, "_md_w8"):
        return False
    if mode not in ("w8a8", "w8", "a8"):
        raise ValueError(f"int8 mode {mode!r}: w8a8 | w8 (weights only, fp16 GEMM, parity check) | a8 (activations only)")
    w = lin.weight.detach()
    N, K = w.shape
    Np, Kp = -(-N // 8) * 8, -(-K // 8) * 8
    wf = w.float()
    sq = None
    if act_max is not None:
        wmax = wf.abs().amax(dim=0).clamp_min(1e-5)
        sm = (act_max.to(wf.device).float().clamp_min(1e-5) ** alpha) / (wmax ** (1.0 - alpha))
        sm = sm.clamp_min(1e-5)
        wf = wf * sm[None, :]
        sq = (1.0 / sm)
        if Kp != K:
            sq = torch.nn.functional.pad(sq, (0, Kp - K))
        lin._md_sq = sq.contiguous()
    ws = wf.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    w8 = torch.round(wf / ws[:, None]).clamp_(-127, 127).to(torch.int8)
    if (Np, Kp) != (N, K):
        w8 = torch.nn.functional.pad(w8, (0, Kp - K, 0, Np - N))
        ws = torch.nn.functional.pad(ws, (0, Np - N))
    if mode != "w8a8":
        wd = (w8[:N, :K].float() * ws[:N, None]) if mode == "w8" else wf
        lin._md_wt = wd.t().contiguous().to(w.dtype)
        lin._md_nk = (N, K, Np, Kp)
        lin._md_mode = mode

        def fwd_dbg(self, x):
            N, K, _, _ = self._md_nk
            x2 = x.reshape(-1, K).float()
            sq = getattr(self, "_md_sq", None)
            if sq is not None:
                x2 = x2 * sq[:K]
            if self._md_mode == "a8":
                s = x2.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * (1.0 / 127.0)
                x2 = torch.round(x2 / s).clamp_(-127, 127) * s
            y = torch.matmul(x2.to(x.dtype), self._md_wt)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], N)

        lin.forward = types.MethodType(fwd_dbg, lin)
        lin._md_w8 = True
        return True
    del wf
    lin._md_w8 = w8.contiguous()
    lin._md_ws = ws.contiguous()
    lin._md_nk = (N, K, Np, Kp)
    if hasattr(lin, "_md_wt"):
        del lin._md_wt
    lin.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)

    def fwd(self, x):
        N, K, Np, Kp = self._md_nk
        shp = x.shape
        x2 = x.reshape(-1, K).float()
        sq = getattr(self, "_md_sq", None)
        if sq is not None:
            x2 = x2 * sq[:K]
        M = x2.shape[0]
        s = x2.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * (1.0 / 127.0)
        xq = torch.round(x2 / s).clamp_(-127, 127).to(torch.int8)
        Mp = M if M > 16 else 32
        if Kp != K or Mp != M:
            xq = torch.nn.functional.pad(xq, (0, Kp - K, 0, Mp - M))
        y = torch._int_mm(xq, self._md_w8.t())
        if Mp != M or Np != N:
            y = y[:M, :N]
        y = (y.to(torch.float32) * s * self._md_ws[:N]).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], N)

    lin.forward = types.MethodType(fwd, lin)
    return True


INT8_TARGETS = ("llm", "vit")
INT8_LLM_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
INT8_VIT_LINEARS = ("attn.qkv", "attn.proj")


def _get_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
    return obj


def _int8_linears(model, targets: Iterable[str], skip: Iterable[str] = ()) -> List[Tuple[str, torch.nn.Linear]]:
    """(key, linear) for every quantisation target; keys are stable across
    processes ("llm:<expert>:<layer>:<name>", "vit:<block>:<name>") so
    calibration statistics can be saved and reloaded."""
    skip = [x for x in skip if x]
    out: List[Tuple[str, torch.nn.Linear]] = []
    for t in targets:
        if t == "llm":
            lm = getattr(model, "lm_head", None)
            experts = list(lm.experts.items()) if isinstance(lm, MergedExperts) else [("base", e) for e in _expert_models(model)]
            for ename, e in experts:
                inner = e.get_model() if hasattr(e, "get_model") else e
                for i, layer in enumerate(getattr(inner, "layers", [])):
                    for name in INT8_LLM_LINEARS:
                        out.append((f"llm:{ename}:{i}:{name}", _get_path(layer, name)))
        elif t == "vit":
            for i, blk in enumerate(getattr(getattr(model, "img_backbone", None), "blocks", None) or []):
                for name in INT8_VIT_LINEARS:
                    out.append((f"vit:{i}:{name}", _get_path(blk, name)))
    return [(k, l) for k, l in out if isinstance(l, torch.nn.Linear) and not any(x in k for x in skip)]


def collect_int8_calibration(model, targets: Iterable[str], run_forwards: Callable[[], None],
                             skip: Iterable[str] = (), log: Log = None) -> Dict[str, torch.Tensor]:
    """Per-input-channel |x| maxima of every int8 target over the forwards
    `run_forwards` performs (real frames).  Call BEFORE quantize_linears_int8;
    the result (CPU fp32 tensors, ~1 MB for the 3B) can be torch.save'd and
    given to the node as int8_calib_path."""
    stats: Dict[str, torch.Tensor] = {}
    handles = []
    for key, lin in _int8_linears(model, targets, skip):
        def hook(mod, inp, _key=key):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().abs().amax(dim=0)
            stats[_key] = torch.maximum(stats[_key], x) if _key in stats else x
        handles.append(lin.register_forward_pre_hook(hook))
    try:
        run_forwards()
    finally:
        for h in handles:
            h.remove()
    stats = {k: v.cpu() for k, v in stats.items()}
    if log:
        log(f"int8 calibration: {len(stats)} linears, {sum(v.numel() for v in stats.values()) * 4 / 2**20:.1f} MB")
    return stats


def quantize_linears_int8(model, targets: Iterable[str], skip: Iterable[str] = (),
                          log: Log = None, calib: Optional[Dict[str, torch.Tensor]] = None,
                          alpha: float = 0.5, mode: str = "w8a8") -> Dict[str, int]:
    """'llm' -> the 7 projections of every decoder layer of every expert
               (embeddings, norms and the vocab head stay fp16);
       'vit' -> the fused qkv and the output proj of every EVA-ViT block
               (needs patch_vit_blocks).
    `skip`: substrings of the linear keys to leave in fp16: 'down_proj',
            'self_attn', ':0:' (layer 0), 'decision_expert' ...
    `calib`: collect_int8_calibration() output -> SmoothQuant with `alpha`.
    Apply before compile_submodules / graph_vit."""
    done: Dict[str, int] = {}
    for t in targets:
        if t not in INT8_TARGETS:
            if log:
                log(f"int8: unknown target {t!r} (known: {INT8_TARGETS})")
            continue
        n = miss = 0
        for key, lin in _int8_linears(model, [t], skip):
            am = calib.get(key) if calib else None
            if calib and am is None:
                miss += 1
            n += int(_quantize_linear_int8(lin, am, alpha, mode))
        done[t] = n
        if log:
            log(f"int8: {t} -> {n} linears quantised ({mode}" + (", torch._int_mm" if mode == "w8a8" else " PARITY-ONLY")
                + (f", smoothquant alpha={alpha}" if calib else "") + ")"
                + (f", skipped {[x for x in skip if x]}" if any(skip) else "")
                + (f", {miss} WITHOUT calibration stats" if miss else ""))
    torch.cuda.empty_cache()
    return done


def set_pipeline_input_size(pipeline, size: int) -> bool:
    """Point the pipeline's ResizeMultiview3D at size x size (the config has
    640x640).  Accepts the config list of dicts (before Compose) or a built
    Compose.  The transform rescales cam_intrinsic / lidar2img itself, so the
    PETR 3D position embedding follows."""
    changed = False
    for t in (pipeline if isinstance(pipeline, list) else getattr(pipeline, "transforms", [])):
        if isinstance(t, dict):
            if t.get("type") == "ResizeMultiview3D":
                t["img_scale"] = (int(size), int(size))
                changed = True
        elif type(t).__name__ == "ResizeMultiview3D":
            t.img_scale = [(int(size), int(size))]
            changed = True
    return changed


def set_vit_input_size(model, size: int, log: Log = None) -> bool:
    """Make the EVA-ViT accept size x size (multiple of 16; size/16 a multiple
    of the 16-token window so window blocks need no padding).  The absolute
    position embedding is interpolated by the backbone itself; the
    global-attention rotary table is built for the training grid (40x40) and
    is rebuilt here for the new grid.  Call BEFORE torch.compile.  This is
    perception at a resolution the model was not trained at: a numerics
    change, gated by the bench and a closed-loop score."""
    vit = getattr(model, "img_backbone", None)
    if vit is None or not hasattr(vit, "rope_glb"):
        return False
    patch = vit.patch_embed.proj.kernel_size[0]
    if size % patch or (size // patch) % 16:
        raise ValueError(f"vit input size {size} must be a multiple of {16 * patch}")
    hw = size // patch
    if vit.rope_glb.freqs_cos.shape[0] == hw * hw:
        return False
    from mmcv.models.backbones.eva_vit import VisionRotaryEmbeddingFast
    old = vit.rope_glb
    dim = old.freqs_cos.shape[-1] // 2
    new = VisionRotaryEmbeddingFast(dim=dim, pt_seq_len=16, ft_seq_len=hw)
    assert new.freqs_cos.shape[-1] == old.freqs_cos.shape[-1], (new.freqs_cos.shape, old.freqs_cos.shape)
    new = new.to(old.freqs_cos.device, old.freqs_cos.dtype)
    vit.rope_glb = new
    n = 0
    for blk in vit.blocks:
        if getattr(blk.attn, "rope", None) is old:
            blk.attn.rope = new
            n += 1
    if log:
        log(f"vit input size {size}: global rope rebuilt for {hw}x{hw} tokens ({n} global blocks)")
    return True


class StaggeredViews:
    """Wrap the backbone so `stale_views` (the rear cameras) go through the ViT
    only every `refresh_every` frames; in between their previous features are
    reused, so the heads always receive all six views.  Install AFTER compile
    and CUDA graphs (CudaGraphed keeps one graph per batch shape).  The
    staleness is one inference period."""

    def __init__(self, forward: Callable, refresh_every: int = 2,
                 always_views=(0, 1, 2), stale_views=(3, 4, 5)) -> None:
        self._forward = forward
        self.refresh_every = max(1, int(refresh_every))
        self.always = list(always_views)
        self.stale = list(stale_views)
        self._cache: Optional[torch.Tensor] = None
        self._count = 0

    def reset(self) -> None:
        self._cache, self._count = None, 0

    def __call__(self, x: torch.Tensor):
        n = x.shape[0]
        full = (self.refresh_every == 1 or self._cache is None
                or self._count % self.refresh_every == 0 or n != len(self.always) + len(self.stale))
        self._count += 1
        if full:
            out = self._forward(x)
            feat = out[0] if isinstance(out, (list, tuple)) else out
            if n == len(self.always) + len(self.stale):
                self._cache = feat[self.stale].clone()
            return [feat]
        out = self._forward(x[self.always])
        part = out[0] if isinstance(out, (list, tuple)) else out
        slots = [None] * n
        for i, v in enumerate(self.always):
            slots[v] = part[i]
        for i, v in enumerate(self.stale):
            slots[v] = self._cache[i]
        return [torch.stack(slots, dim=0)]


def install_staggered_views(model, refresh_every: int, log: Log = None) -> Optional[StaggeredViews]:
    if refresh_every <= 1:
        return None
    bb = model.img_backbone
    if isinstance(getattr(bb, "_md_stagger", None), StaggeredViews):
        return bb._md_stagger
    sv = StaggeredViews(bb.forward, refresh_every)
    bb.forward = sv
    bb._md_stagger = sv
    if log:
        log(f"staggered views: rear cameras refreshed every {refresh_every} frames")
    return sv


class StageTimer:
    """Wrap methods with CUDA events and report per-stage ms for one forward
    (stream time, so GPU idle while the CPU launches counts -- that is the
    launch-bound overhead we want to see)."""

    def __init__(self) -> None:
        self._pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.enabled = True

    def wrap(self, obj, attr: str, label: str) -> None:
        fn = getattr(obj, attr)
        if isinstance(fn, torch.nn.Module):
            obj, attr, fn = fn, "forward", fn.forward
        if getattr(fn, "_md_timed", False):
            return

        def wrapped(*a, **kw):
            if not self.enabled:
                return fn(*a, **kw)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            self._pending.append((label, s, e))
            return out

        wrapped._md_timed = True
        setattr(obj, attr, wrapped)

    def wrap_minddrive(self, model) -> None:
        """The stages of Minddrive.simple_test_pts."""
        self.wrap(model, "extract_feat", "vit")
        self.wrap(model, "position_embeding", "pos_embed")
        self.wrap(model.pts_bbox_head, "forward", "det_head")
        self.wrap(model.pts_bbox_head, "get_motion_bboxes", "det_decode")
        self.wrap(model.pts_bbox_head, "get_bboxes", "det_decode")
        if getattr(model, "map_head", None) is not None:
            self.wrap(model.map_head, "forward", "map_head")
            self.wrap(model.map_head, "get_bboxes", "map_decode")
        self.wrap_llm(model)
        self.wrap(model, "distribution_forward", "planner")
        self.wrap(model, "pw_distribution_forward", "planner")
        self.wrap(model, "future_states_predict", "planner")
        self.wrap(model, "pw_future_states_predict", "planner")

    def wrap_llm(self, model) -> None:
        """(Re)wrap the two LLM entry points; call again after merge_lora,
        which replaces lm_head."""
        lm = getattr(model, "lm_head", None)
        if lm is None:
            return
        if isinstance(lm, MergedExperts):
            for e in lm.experts.values():
                self.wrap(e, "inference_action_distribution", "llm_decision")
                self.wrap(e, "inference_waypoints", "llm_action")
        else:
            self.wrap(lm, "inference_action_distribution", "llm_decision")
            self.wrap(lm, "inference_waypoints", "llm_action")

    def report(self) -> Dict[str, float]:
        torch.cuda.synchronize()
        out: Dict[str, float] = {}
        for label, s, e in self._pending:
            out[label] = out.get(label, 0.0) + s.elapsed_time(e)
        self._pending.clear()
        return out

    @staticmethod
    def format(stages: Dict[str, float], total_ms: Optional[float] = None) -> str:
        parts = [f"{k}={v:.0f}" for k, v in stages.items()]
        if total_ms is not None:
            acc = sum(stages.values())
            parts.append(f"other={max(total_ms - acc, 0.0):.0f}")
        return " ".join(parts)


class ParallelDecoder:
    """Run the per-camera decode (BGR + JPEG re-encode) on a thread pool.
    OpenCV releases the GIL inside imencode/imdecode.  Identical record
    objects are decoded once."""

    def __init__(self, decode_fn: Callable, workers: int = 6) -> None:
        self._decode = decode_fn
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None

    def __call__(self, records: List, jpeg_quality: int) -> List:
        uniq: Dict[int, object] = {}
        for r in records:
            uniq.setdefault(id(r), r)
        if self._pool is None or len(uniq) == 1:
            decoded = {rid: self._decode(r, jpeg_quality) for rid, r in uniq.items()}
        else:
            futs = {rid: self._pool.submit(self._decode, r, jpeg_quality) for rid, r in uniq.items()}
            decoded = {rid: f.result() for rid, f in futs.items()}
        return [decoded[id(r)] for r in records]

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def parallelize_pipeline(pipeline, workers: int = 6, log: Log = None) -> List[str]:
    """Run the per-image loops of the pipeline's own transforms on a thread
    pool.  The per-image code (`_img_transform`, `imnormalize`) is untouched;
    only the sequential `for img in imgs` becomes a map."""
    import numpy as np

    if workers <= 1:
        return []
    pool = ThreadPoolExecutor(max_workers=workers)
    done: List[str] = []
    for t in getattr(pipeline, "transforms", []):
        name = type(t).__name__
        if name == "ResizeCropFlipRotImage":
            from PIL import Image

            def _resize_call(self, results):
                imgs = results["img"]
                assert self.data_aug_conf["rot_lim"] == (0.0, 0.0), "Rotation is not currently supported"
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

                def one(img):
                    img = Image.fromarray(np.uint8(img))
                    img, ida_mat = self._img_transform(
                        img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
                    return np.array(img).astype(np.float32), ida_mat

                outs = list(pool.map(one, imgs))
                results["img"] = [o[0] for o in outs]
                for i, (_, ida_mat) in enumerate(outs):
                    results["cam_intrinsic"][i][:3, :3] = ida_mat @ results["cam_intrinsic"][i][:3, :3]
                results["lidar2img"] = [results["cam_intrinsic"][i] @ results["lidar2cam"][i]
                                        for i in range(len(results["lidar2cam"]))]
                return results

            t.__class__ = type("ParallelResizeCropFlipRotImage", (type(t),), {"__call__": _resize_call})
            done.append(name)
        elif name == "NormalizeMultiviewImage":
            from mmcv.image import imnormalize

            def _norm_call(self, results):
                results["img"] = list(pool.map(
                    lambda img: imnormalize(img, self.mean, self.std, self.to_rgb), results["img"]))
                results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
                return results

            t.__class__ = type("ParallelNormalizeMultiviewImage", (type(t),), {"__call__": _norm_call})
            done.append(name)
    if log:
        log(f"pipeline: parallelised {done or 'nothing'} on {workers} threads")
    return done


def slice_decision_logits(model, log: Log = None) -> int:
    """LlavaQwen2ForCausalLM.inference_action_distribution projects every
    position of the prefill through the 151936-way vocab head and then keeps
    `output_logits[:, -2, :]`.  Projecting that one row is the same numbers
    (a Linear is row-wise) minus a 570 x 151936 GEMM and its 170 MB write.
    The method body below is upstream's, with only that change."""
    import torch.nn.functional as F

    def inference_action_distribution(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, new_input_ids
             ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        outputs = self.model(
            input_ids=inputs, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=None, inputs_embeds=inputs_embeds, use_cache=True,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.use_return_dict)
        hidden_states = outputs[0]
        last_token_logits = self.lm_head(hidden_states[:, -2, :])
        ma = torch.stack([torch.tensor(m) for m in self.config.meta_action_token_idx[:7]])
        ma_logits = last_token_logits.gather(dim=-1, index=ma.to(last_token_logits.device).unsqueeze(0))
        action_log_probs_normalized = F.log_softmax(ma_logits, dim=-1)
        return action_log_probs_normalized, inputs_embeds, new_input_ids

    n = 0
    for e in _expert_models(model):
        if getattr(e, "_md_logits_sliced", False):
            continue
        e.inference_action_distribution = types.MethodType(torch.no_grad()(inference_action_distribution), e)
        e._md_logits_sliced = True
        n += 1
    if log:
        log(f"decision logits: vocab projection on 1 position instead of the whole prefill ({n} experts)")
    return n


def reorder_rounds_for_overlap(batch, model) -> bool:
    """Put the waypoint (action expert) round before the meta-action
    (decision expert) round in the batch's input_ids.

    Minddrive.simple_test_pts loops over the prompt rounds; the decision
    branch appends nothing to the conversation history (its `if False:`
    block), so the action expert's input is the same whichever round comes
    first, and the two prefills are independent functions of the same
    inputs.  Upstream's order runs the decision expert first and syncs on its
    argmax at once, which leaves nothing for it to overlap with; reversed,
    the action expert's prefill and the planner are already queued when the
    decision prefill starts, and overlap_experts() puts that one on a side
    stream.  Returns True if the rounds were swapped."""
    try:
        rounds = batch["input_ids"][0][0]
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(rounds, list) or len(rounds) != 2:
        return False
    lm = getattr(model, "lm_head", None)
    ma = getattr(getattr(lm, "config", None), "meta_action_token_idx", None)
    if not ma:
        return False
    ma = set(int(t) for t in ma)
    first_is_decision = any(int(t) in ma for t in rounds[0].flatten().tolist())
    if first_is_decision:
        rounds.reverse()
    return first_is_decision


def overlap_experts(model, log: Log = None) -> bool:
    """Run the decision expert's prefill on a side CUDA stream, waiting only
    for the inputs both prefills share (recorded when the action expert's
    prefill starts), so the two ~200 ms prefills execute concurrently.  Needs
    the rounds reordered (reorder_rounds_for_overlap) or it degenerates to
    the sequential order.  Same kernels, same numbers."""
    lm = getattr(model, "lm_head", None)
    if lm is None or getattr(model, "_md_overlap", False):
        return False
    experts = _expert_models(model)
    side = torch.cuda.Stream()
    state: Dict[str, object] = {}

    def wrap_action(e):
        orig = e.inference_waypoints

        def inference_waypoints(*a, **kw):
            state["inputs_ready"] = torch.cuda.current_stream().record_event()
            return orig(*a, **kw)
        e.inference_waypoints = inference_waypoints

    def wrap_decision(e):
        orig = e.inference_action_distribution

        def inference_action_distribution(*a, **kw):
            cur = torch.cuda.current_stream()
            ev = state.pop("inputs_ready", None)
            if ev is None:
                side.wait_stream(cur)
            else:
                side.wait_event(ev)
            with torch.cuda.stream(side):
                out = orig(*a, **kw)
            cur.wait_stream(side)
            for t in _tensors(out):
                t.record_stream(cur)
            return out
        e.inference_action_distribution = inference_action_distribution

    if isinstance(lm, MergedExperts):
        for name, e in lm.experts.items():
            (wrap_decision if name == "decision_expert" else wrap_action)(e)
    else:
        wrap_action(experts[0])
        wrap_decision(experts[0])
    model._md_overlap = True
    if log:
        log("overlap experts: decision-expert prefill on a side stream (rounds reordered per batch)")
    return True


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def apply_speedups(model, opts: Dict[str, object], log: Log = None) -> List[str]:
    """Apply the selected speedups in the order the bench validated them.
    opts keys (all optional): merge_lora, llm_attn, vit_glue, down_proj_t,
    vit_weight_t, vit_window_nopad, int8_targets (list), int8_skip (list), int8_calib_path,
    int8_alpha, vit_input_size (pipeline resize done by the caller with
    set_pipeline_input_size), map_head_slice,
    compile_targets (list), compile_mode, cuda_graph_vit, cuda_graph_heads,
    logits_slice, overlap_experts, rear_view_refresh_every.  With overlap_experts the
    caller must run reorder_rounds_for_overlap(batch, model) on every batch."""
    done: List[str] = []
    if opts.get("merge_lora"):
        if merge_lora_experts(model, log):
            done.append("merge_lora")
    impl = str(opts.get("llm_attn") or "")
    if impl and impl != "keep":
        if set_llm_attention(model, impl, log):
            done.append(f"llm_attn={impl}")
    if opts.get("vit_glue"):
        if patch_vit_blocks(model, log):
            done.append("vit_glue")
    if opts.get("down_proj_t"):
        if transpose_llm_down_proj(model, log):
            done.append("down_proj_t")
    if opts.get("vit_weight_t"):
        if transpose_vit_linears(model, log=log):
            done.append("vit_weight_t")
    if opts.get("vit_window_nopad"):
        if patch_vit_window_blocks(model, log):
            done.append("vit_window_nopad")
    size = int(opts.get("vit_input_size") or 0)
    if size and size != 640:
        if set_vit_input_size(model, size, log):
            done.append(f"vit_input_size={size}")
    int8_t = [t for t in (opts.get("int8_targets") or []) if t]
    if int8_t:
        calib = None
        path = str(opts.get("int8_calib_path") or "")
        if path:
            calib = torch.load(path, map_location="cpu")
            if log:
                log(f"int8: calibration stats loaded from {path} ({len(calib)} linears)")
        q = quantize_linears_int8(model, int8_t, opts.get("int8_skip") or (), log,
                                  calib, float(opts.get("int8_alpha") or 0.5))
        done += [f"int8:{t}" for t, n in q.items() if n]
    if opts.get("map_head_slice"):
        if slice_map_head_one2one(model, log):
            done.append("map_head_slice")
    targets = [t for t in (opts.get("compile_targets") or []) if t]
    if opts.get("cuda_graph_heads"):
        targets = [t for t in targets if t != "heads"]
    if targets:
        done += [f"compile:{t}" for t in compile_submodules(model, targets, str(opts.get("compile_mode") or "default"), log)]
    if opts.get("cuda_graph_heads"):
        if graph_heads(model, log):
            done.append("cuda_graph_heads")
    if opts.get("cuda_graph_vit"):
        if graph_vit(model, log):
            done.append("cuda_graph_vit")
    if opts.get("logits_slice"):
        if slice_decision_logits(model, log):
            done.append("logits_slice")
    if opts.get("overlap_experts"):
        if overlap_experts(model, log):
            done.append("overlap_experts")
    every = int(opts.get("rear_view_refresh_every") or 1)
    if every > 1:
        if install_staggered_views(model, every, log) is not None:
            done.append(f"rear_view_refresh_every={every}")
    return done
