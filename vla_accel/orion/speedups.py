"""Inference speedups for ORION, applied as post-build transforms on the model object.

No ORION source is edited, so the container's pre-built /root/Orion runs unchanged.
"""
from __future__ import annotations

import types
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


def merge_lora(model, log: Optional[Callable[[str], None]] = None) -> bool:
    """Fold the peft LoRA adapters (q/k/v/o, r=16) into the LLM base weights."""
    lm = getattr(model, "lm_head", None)
    base = getattr(lm, "base_model", None)
    if base is None or not hasattr(base, "merge_and_unload"):
        if log:
            log("merge_lora: lm_head is not a peft model, nothing to merge")
        return False
    cfg = lm.config
    merged = base.merge_and_unload()
    if hasattr(cfg, "waypoint_token_idx") and not hasattr(merged.config, "waypoint_token_idx"):
        merged.config.waypoint_token_idx = cfg.waypoint_token_idx
    merged.eval()
    model.lm_head = merged
    if log:
        log("merge_lora: LoRA folded into LLM weights")
    return True


def patch_llm_flash_attention(model, log: Optional[Callable[[str], None]] = None) -> bool:
    """Run the unpadded batch-1 LLaMA prefill through flash_attn_func(causal=True)."""
    try:
        import transformers.models.llama.modeling_llama as L
        from flash_attn import flash_attn_func
    except Exception as e:
        if log:
            log(f"flash-attn patch skipped: {type(e).__name__}: {e}")
        return False

    lm = getattr(model, "lm_head", None)
    llama = lm.get_model() if hasattr(lm, "get_model") else None
    if llama is None:
        return False
    if next(llama.parameters()).dtype not in (torch.float16, torch.bfloat16):
        if log:
            log("flash-attn patch skipped: LLM is not fp16/bf16")
        return False
    if getattr(L.LlamaAttention.forward, "_orion_flash", False):
        return True

    orig_prep = llama._prepare_decoder_attention_mask

    def _prep(attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Batch 1 with an all-True mask is pure causal: let the kernel apply it.
        if (past_key_values_length == 0 and input_shape[0] == 1
                and (attention_mask is None or bool(attention_mask.all()))):
            return None
        return orig_prep(attention_mask, input_shape, inputs_embeds, past_key_values_length)

    llama._prepare_decoder_attention_mask = _prep

    orig_forward = L.LlamaAttention.forward

    def _forward(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, output_attentions=False, use_cache=False):
        if (attention_mask is not None or past_key_value is not None
                or output_attentions or getattr(self, "pretraining_tp", 1) > 1
                or hidden_states.dtype not in (torch.float16, torch.bfloat16)):
            return orig_forward(self, hidden_states, attention_mask, position_ids,
                                past_key_value, output_attentions, use_cache)
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = L.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        present = (k, v) if use_cache else None
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
        out = out.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(out), None, present

    _forward._orion_flash = True
    L.LlamaAttention.forward = _forward
    if log:
        log("flash-attn patch: LLaMA prefill attention -> flash_attn_func(causal)")
    return True


COMPILE_TARGETS = ("vit", "llm", "heads", "posembed")
HEAD_SUBMODULES = (
    ("pts_bbox_head", "transformer"),
    ("pts_bbox_head", "memory_decoder_mq"),
    ("pts_bbox_head", "memory_decoder_cq"),
    ("map_head", "transformer"),
)


def _compile_forward(module: torch.nn.Module, mode: str) -> None:
    """Replace a module's forward by torch.compile of its eager forward."""
    if not hasattr(module, "_orion_eager_forward"):
        module._orion_eager_forward = module.forward
    module.forward = torch.compile(module._orion_eager_forward, mode=mode)


def uncompile_submodules(model) -> int:
    """Restore eager forwards on everything compile_submodules touched."""
    n = 0
    for m in model.modules():
        if hasattr(m, "_orion_eager_forward"):
            m.forward = m._orion_eager_forward
            del m._orion_eager_forward
            n += 1
    torch._dynamo.reset()
    return n


def compile_submodules(model, targets: Iterable[str], mode: str = "default",
                       log: Optional[Callable[[str], None]] = None) -> List[str]:
    """torch.compile the ViT, LlamaModel, PETR head stacks or position embedding in place."""
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.automatic_dynamic_shapes = False
    done: List[str] = []
    for t in targets:
        try:
            if t == "vit":
                _compile_forward(model.img_backbone, mode)
            elif t == "llm":
                _compile_forward(model.lm_head.get_model(), mode)
            elif t == "posembed":
                model.position_embeding = torch.compile(model.position_embeding, mode=mode)
            elif t == "heads":
                n = 0
                for head_name, sub_name in HEAD_SUBMODULES:
                    head = getattr(model, head_name, None)
                    sub = getattr(head, sub_name, None) if head is not None else None
                    if isinstance(sub, torch.nn.Module):
                        # No CUDA graphs: each stack runs several times per forward.
                        _compile_forward(sub, "default")
                        n += 1
                if log:
                    log(f"compile: heads -> {n} transformer stacks wrapped (mode='default')")
            else:
                if log:
                    log(f"compile: unknown target '{t}' (known: {COMPILE_TARGETS})")
                continue
            done.append(t)
            if log:
                log(f"compile: {t} wrapped with torch.compile(mode={mode!r})")
        except Exception as e:
            if log:
                log(f"compile: {t} failed, staying eager ({type(e).__name__}: {e})")
    return done


class StageTimer:
    """Per-stage CUDA-event timing of one ORION forward."""

    def __init__(self) -> None:
        self._pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.enabled = True

    def wrap(self, obj, attr: str, label: str) -> None:
        """Time every call of obj.attr (or of a sub-module's forward) under label."""
        fn = getattr(obj, attr)
        if isinstance(fn, torch.nn.Module):
            obj, attr, fn = fn, "forward", fn.forward

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

        setattr(obj, attr, wrapped)

    def wrap_orion(self, model) -> None:
        """Wrap the stages of Orion.simple_test / simple_test_pts."""
        self.wrap(model, "extract_feat", "vit")
        self.wrap(model, "position_embeding", "pos_embed")
        self.wrap(model.pts_bbox_head, "forward", "det_head")
        self.wrap(model.pts_bbox_head, "get_motion_bboxes", "det_decode")
        self.wrap(model.pts_bbox_head, "get_bboxes", "det_decode")
        if getattr(model, "map_head", None) is not None:
            self.wrap(model.map_head, "forward", "map_head")
            self.wrap(model.map_head, "get_bboxes", "map_decode")
        if getattr(model, "lm_head", None) is not None:
            self.wrap(model.lm_head, "inference_ego", "llm")
        self.wrap(model, "distribution_forward", "planner_vae")
        self.wrap(model, "future_states_predict", "planner_gru")
        self.wrap(model, "ego_fut_decoder", "planner_dec")

    def report(self) -> Dict[str, float]:
        """Summed ms per stage since the last report."""
        torch.cuda.synchronize()
        out: Dict[str, float] = {}
        for label, s, e in self._pending:
            out[label] = out.get(label, 0.0) + s.elapsed_time(e)
        self._pending.clear()
        return out

    @staticmethod
    def format(stages: Dict[str, float], total_ms: Optional[float] = None) -> str:
        """One-line stage summary, with the untimed remainder as 'other'."""
        parts = [f"{k}={v:.0f}" for k, v in stages.items()]
        if total_ms is not None:
            acc = sum(stages.values())
            parts.append(f"other={max(total_ms - acc, 0.0):.0f}")
        return " ".join(parts)


class ParallelDecoder:
    """Run the per-camera load and JPEG re-encode on a thread pool."""

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
        """Release the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def parallelize_pipeline(pipeline, workers: int = 6,
                         log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Map the per-image loops of ORION's resize and normalise transforms over a thread pool."""
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


class TrtBackbone:
    """Drop-in img_backbone forward backed by a static-shape TensorRT engine."""

    def __init__(self, engine_path: str, log: Optional[Callable[[str], None]] = None) -> None:
        import tensorrt as trt

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self._logger) as rt:
            self._engine = rt.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self._ctx = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        self._in = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self._out = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        dt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self._in_dtype = dt[self._engine.get_tensor_dtype(self._in)]
        self._out_dtype = dt[self._engine.get_tensor_dtype(self._out)]
        self._in_shape = tuple(self._engine.get_tensor_shape(self._in))
        self._out_shape = tuple(self._engine.get_tensor_shape(self._out))
        self._out_buf = torch.empty(self._out_shape, dtype=self._out_dtype, device="cuda")
        self._ctx.set_tensor_address(self._out, self._out_buf.data_ptr())
        if log:
            log(f"TrtBackbone: {engine_path} {self._in}{self._in_shape}/{self._in_dtype} -> "
                f"{self._out}{self._out_shape}/{self._out_dtype}")

    def __call__(self, x: torch.Tensor):
        x = x.to(self._in_dtype).contiguous()
        assert tuple(x.shape) == self._in_shape, f"engine expects {self._in_shape}, got {tuple(x.shape)}"
        self._ctx.set_tensor_address(self._in, x.data_ptr())
        ok = self._ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return [self._out_buf.clone()]


def install_trt_backbone(model, engine_path: str,
                         log: Optional[Callable[[str], None]] = None) -> bool:
    """Route model.img_backbone through a TensorRT engine (fallback: eager)."""
    try:
        runner = TrtBackbone(engine_path, log)
    except Exception as e:
        if log:
            log(f"TrtBackbone unavailable ({type(e).__name__}: {e}); ViT stays PyTorch")
        return False
    bb = model.img_backbone
    if not hasattr(bb, "_orion_eager_forward"):
        bb._orion_eager_forward = bb.forward
    bb.forward = runner
    bb._orion_trt = runner
    return True


def patch_vit_blocks(model, log: Optional[Callable[[str], None]] = None) -> int:
    """Rewrite EVA-ViT blocks with fused QKV, SDPA and fused SwiGLU W1/W2, same weights."""
    import torch.nn.functional as F

    vit = getattr(model, "img_backbone", None)
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        return 0
    n = 0
    for blk in blocks:
        attn, mlp = blk.attn, blk.mlp
        if getattr(attn, "_orion_glue", False):
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

        attn._orion_glue_forward = attn.forward
        attn.forward = types.MethodType(attn_forward, attn)
        attn._orion_glue = True

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

            mlp._orion_glue_forward = mlp.forward
            mlp.forward = types.MethodType(mlp_forward, mlp)
        n += 1
    if log:
        log(f"vit glue: {n} blocks -> fused qkv + SDPA + fused w12")
    return n


def slice_map_head_one2one(model, log: Optional[Callable[[str], None]] = None) -> bool:
    """Keep only the map head's one-to-one lane queries; exact, as its mask isolates the rest."""
    head = getattr(model, "map_head", None)
    if head is None or getattr(head, "_orion_sliced", False):
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
    head._orion_sliced = True
    if log:
        log(f"map head slice: {n_all} -> {n1} lane queries (one-to-many dropped)")
    return True


def overlap_heads(model, log: Optional[Callable[[str], None]] = None) -> bool:
    """Run the map head on a side CUDA stream while the det head runs."""
    det, mh = getattr(model, "pts_bbox_head", None), getattr(model, "map_head", None)
    if det is None or mh is None or getattr(model, "_orion_overlap", False):
        return False
    side = torch.cuda.Stream()
    det_fwd, map_fwd = det.forward, mh.forward
    state: Dict[str, object] = {}

    def det_forward(img_metas, pos_embed, **data):
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            state["out"] = map_fwd(img_metas, pos_embed, **data)
            state["event"] = side.record_event()
        return det_fwd(img_metas, pos_embed, **data)

    def map_forward(img_metas, pos_embed, **data):
        if "out" not in state:
            return map_fwd(img_metas, pos_embed, **data)
        torch.cuda.current_stream().wait_event(state.pop("event"))
        out = state.pop("out")
        for t in _tensors(out):
            t.record_stream(torch.cuda.current_stream())
        return out

    det.forward = det_forward
    mh.forward = map_forward
    model._orion_overlap = True
    if log:
        log("overlap heads: map head runs on a side stream during the det head")
    return True


def _tensors(obj):
    """Yield every tensor in a nested dict/list/tuple."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors(v)


class Int8Linear(torch.nn.Module):
    """W8A8 nn.Linear: per-channel int8 weight, per-token dynamic int8 activation, torch._int_mm."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.detach()
        scale = w.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-8) / 127.0
        self.register_buffer("weight_int8", (w.float() / scale).round().clamp_(-127, 127).to(torch.int8))
        self.register_buffer("w_scale", scale.squeeze(1))
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.shape[0] <= 16:  # _int_mm needs M > 16
            w = (self.weight_int8.to(x.dtype) * self.w_scale.to(x.dtype)[:, None])
            y = torch.nn.functional.linear(x2, w, self.bias)
            return y.reshape(*shape[:-1], self.out_features)
        xf = x2.float()
        x_scale = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        xq = (xf / x_scale).round().clamp_(-127, 127).to(torch.int8)
        acc = torch._int_mm(xq, self.weight_int8.t())  # [N,K].t() layout is the fast cuBLASLt path
        y = acc.float() * (x_scale * self.w_scale[None, :])
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*shape[:-1], self.out_features)


LLM_INT8_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize_llm_int8(model, targets=LLM_INT8_TARGETS,
                      log: Optional[Callable[[str], None]] = None,
                      skip_layers: Iterable[int] = ()) -> int:
    """Swap LLaMA decoder projections for Int8Linear; apply after merge_lora, before compile."""
    lm = getattr(model, "lm_head", None)
    llama = lm.get_model() if hasattr(lm, "get_model") else None
    layers = getattr(llama, "layers", None)
    if layers is None:
        return 0
    n = 0
    skip = set(int(i) for i in skip_layers)
    for i, layer in enumerate(layers):
        if i in skip:
            continue
        for parent in (layer.self_attn, layer.mlp):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, torch.nn.Linear):
                    setattr(parent, name, Int8Linear(lin))
                    n += 1
    torch.cuda.empty_cache()
    if log:
        log(f"int8 llm: {n} projections -> W8A8 dynamic (torch._int_mm); "
            f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return n


def transpose_llm_down_proj(model, log: Optional[Callable[[str], None]] = None) -> int:
    """Store the K=11008 down_proj weight [K,N] contiguous for a faster fp16 GEMM."""
    lm = getattr(model, "lm_head", None)
    llama = lm.get_model() if hasattr(lm, "get_model") else None
    n = 0
    for layer in getattr(llama, "layers", []):
        lin = layer.mlp.down_proj
        if isinstance(lin, torch.nn.Linear):
            lin._orion_wt = lin.weight.detach().t().contiguous()

            def fwd(self, x):
                y = torch.matmul(x, self._orion_wt)
                return y if self.bias is None else y + self.bias

            lin.forward = types.MethodType(fwd, lin)
            n += 1
    if log:
        log(f"down_proj transposed weights: {n} layers")
    return n


def collect_llm_act_stats(model, run_calibration: Callable[[], None],
                          targets=LLM_INT8_TARGETS) -> Dict[Tuple[int, str], torch.Tensor]:
    """Per-input-channel max|x| of each target projection over run_calibration()."""
    llama = model.lm_head.get_model()
    stats: Dict[Tuple[int, str], torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(llama.layers):
        for parent in (layer.self_attn, layer.mlp):
            for name in targets:
                lin = getattr(parent, name, None)
                if not isinstance(lin, torch.nn.Linear):
                    continue

                def hook(mod, inp, key=(i, name)):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs().amax(dim=0).float()
                    stats[key] = torch.maximum(stats[key], x) if key in stats else x

                handles.append(lin.register_forward_pre_hook(hook))
    try:
        run_calibration()
    finally:
        for h in handles:
            h.remove()
    return stats


@torch.no_grad()
def smooth_llm(model, stats: Dict[Tuple[int, str], torch.Tensor], alpha: float = 0.5,
               log: Optional[Callable[[str], None]] = None) -> int:
    """SmoothQuant: move activation outliers into the consumer weights, exact in fp16."""
    llama = model.lm_head.get_model()
    n = 0
    for i, layer in enumerate(llama.layers):
        at, mlp = layer.self_attn, layer.mlp
        groups = [
            ([at.q_proj, at.k_proj, at.v_proj], ("norm", layer.input_layernorm), (i, "q_proj")),
            ([mlp.gate_proj, mlp.up_proj], ("norm", layer.post_attention_layernorm), (i, "gate_proj")),
            ([at.o_proj], ("rows", at.v_proj), (i, "o_proj")),
            ([mlp.down_proj], ("rows", mlp.up_proj), (i, "down_proj")),
        ]
        for consumers, (kind, producer), key in groups:
            if key not in stats or not all(isinstance(c, torch.nn.Linear) for c in consumers):
                continue
            x_max = stats[key].to(consumers[0].weight.device).clamp_min(1e-5)
            w_max = torch.stack([c.weight.detach().abs().amax(dim=0).float() for c in consumers]).amax(dim=0).clamp_min(1e-5)
            s = (x_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-2, max=1e4)
            for c in consumers:
                c.weight.mul_(s.to(c.weight.dtype)[None, :])
                if hasattr(c, "_orion_wt"):
                    c._orion_wt.copy_(c.weight.detach().t())
            if kind == "norm":
                producer.weight.div_(s.to(producer.weight.dtype))
            else:
                producer.weight.div_(s.to(producer.weight.dtype)[:, None])
                if producer.bias is not None:
                    producer.bias.div_(s.to(producer.bias.dtype))
            n += 1
    if log:
        log(f"smoothquant: {n} producer/consumer groups rescaled (alpha={alpha})")
    return n


def save_llm_act_stats(stats: Dict[Tuple[int, str], torch.Tensor], path: str) -> None:
    """Save calibration stats as {'layer:proj': tensor}."""
    torch.save({f"{i}:{n}": v.cpu() for (i, n), v in stats.items()}, path)


def load_llm_act_stats(path: str) -> Dict[Tuple[int, str], torch.Tensor]:
    """Load stats written by save_llm_act_stats."""
    raw = torch.load(path, map_location="cpu")
    out = {}
    for k, v in raw.items():
        i, n = k.split(":")
        out[(int(i), n)] = v
    return out


def apply_llm_int8(model, stats_path: str, alpha: float, skip_layers: Iterable[int],
                   log: Optional[Callable[[str], None]] = None,
                   targets: Iterable[str] = LLM_INT8_TARGETS) -> bool:
    """SmoothQuant with saved stats (int8_sweep.py --save-stats), then W8A8 on the rest."""
    if stats_path:
        try:
            stats = load_llm_act_stats(stats_path)
        except Exception as e:
            if log:
                log(f"int8 llm: cannot load stats {stats_path} ({e}); quantising WITHOUT smoothing")
            stats = None
        if stats and alpha > 0:
            smooth_llm(model, stats, alpha, log)
    elif log:
        log("int8 llm: no calibration stats given; quantising WITHOUT smoothing")
    return quantize_llm_int8(model, tuple(targets), log=log, skip_layers=skip_layers) > 0


def set_pipeline_input_size(pipeline_cfg: list, size: int) -> bool:
    """Point the pipeline's ResizeMultiview3D at size x size (camera matrices follow)."""
    changed = False
    for t in pipeline_cfg:
        if isinstance(t, dict) and t.get("type") == "ResizeMultiview3D":
            t["img_scale"] = (int(size), int(size))
            changed = True
    return changed


def set_vit_input_size(model, size: int, log: Optional[Callable[[str], None]] = None) -> bool:
    """Rebuild the EVA-ViT global rotary table for a size x size input; call before compile."""
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
    """Backbone wrapper that refreshes the rear views only every N frames; install after compile."""

    def __init__(self, forward: Callable, refresh_every: int = 2,
                 always_views=(0, 1, 2), stale_views=(3, 4, 5)) -> None:
        self._forward = forward
        self.refresh_every = max(1, int(refresh_every))
        self.always = list(always_views)
        self.stale = list(stale_views)
        self._cache: Optional[torch.Tensor] = None
        self._count = 0

    def reset(self) -> None:
        """Drop the cached rear features (sequence start)."""
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


def install_staggered_views(model, refresh_every: int,
                            log: Optional[Callable[[str], None]] = None) -> Optional[StaggeredViews]:
    """Wrap model.img_backbone in StaggeredViews when refresh_every > 1."""
    if refresh_every <= 1:
        return None
    bb = model.img_backbone
    sv = StaggeredViews(bb.forward, refresh_every)
    bb.forward = sv
    bb._orion_stagger = sv
    if log:
        log(f"staggered views: rear cameras refreshed every {refresh_every} frames")
    return sv
