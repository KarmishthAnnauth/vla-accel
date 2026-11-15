"""TensorRT helpers shared by trt_build.py and trt_int8.py: engine build and runner, export modules."""

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

OUT = os.environ.get("ALPAMAYO_TRT_DIR", "/data/alpamayo-1.5-trt")


class TrtRunner:
    """A static-shape engine with its inputs and outputs bound to torch tensors."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                  trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32,
                  trt.DataType.BOOL: torch.bool}
        self.inputs, self.outputs = {}, {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            t = torch.zeros(tuple(self._engine.get_tensor_shape(name)), device="cuda",
                            dtype=dtypes[self._engine.get_tensor_dtype(name)])
            is_in = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_in else self.outputs)[name] = t
            self._context.set_tensor_address(name, t.data_ptr())
        self._stream = torch.cuda.Stream()

    def __call__(self):
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        current.wait_stream(self._stream)
        return self.outputs


def bench(fn, n=20, warm=5):
    """Mean ms per call of fn after warm-up."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of a against b."""
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
    """Expand grouped K/V heads to all query heads."""
    b, g, s, d = x.shape
    return x[:, :, None].expand(b, g, rep, s, d).reshape(b, g * rep, s, d)


class Vit(torch.nn.Module):
    """FastAlpamayo._vit_fn in fp32: pixels [N, 1536] -> merged + deepstack features."""

    def __init__(self, visual, grid):
        super().__init__()
        self.vis = copy.deepcopy(visual).float()
        sizes = (grid[:, 1] * grid[:, 2]).tolist()
        self.n, self.s = len(sizes), sizes[0]
        self.register_buffer("pos", self.vis.fast_pos_embed_interpolate(grid).float())
        rot = self.vis.rot_pos_emb(grid)
        rot = torch.cat((rot, rot), dim=-1)
        self.register_buffer("cos", rot.cos().float())
        self.register_buffer("sin", rot.sin().float())

    def forward(self, pixels):
        vis, n, s = self.vis, self.n, self.s
        x = vis.patch_embed(pixels) + self.pos
        heads = vis.blocks[0].attn.num_heads
        deep = []
        for i, blk in enumerate(vis.blocks):
            qkv = blk.attn.qkv(blk.norm1(x)).reshape(n * s, 3, heads, -1).permute(1, 0, 2, 3)
            q, k = apply_rotary_pos_emb_vision(qkv[0], qkv[1], self.cos, self.sin)
            q, k, v = (t.reshape(n, s, heads, -1).transpose(1, 2) for t in (q, k, qkv[2]))
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + blk.attn.proj(a.transpose(1, 2).reshape(n * s, -1))
            x = x + blk.mlp(blk.norm2(x))
            if i in vis.deepstack_visual_indexes:
                deep.append(vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)](x))
        return (vis.merger(x), *deep)


class DecoderLayer(torch.nn.Module):
    """One Qwen3VLTextDecoderLayer over a causal window: (h, cos, sin) -> (h, keys, values)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = copy.deepcopy(layer).float()

    def forward(self, h, cos, sin):
        layer, at = self.layer, self.layer.self_attn
        b, s, _ = h.shape
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(b, s, -1, at.head_dim)).transpose(1, 2)
        v = at.v_proj(x).view(b, s, -1, at.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = q.shape[1] // k.shape[1]
        a = F.scaled_dot_product_attention(q, repeat_kv(k, rep), repeat_kv(v, rep), is_causal=True)
        h = h + at.o_proj(a.transpose(1, 2).reshape(b, s, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v


def build_engine(onnx_path: str, engine_path: str, level: int, fp32_names=(),
                 int8: bool = False) -> float:
    """Build an fp16 (optionally INT8) engine from ONNX, with fp32 layers by name; returns build s."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = level
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    pinned = 0
    if fp32_names:  # RMSNorm x^2 overflows fp16
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (any(s in layer.name for s in fp32_names) and layer.num_outputs
                    and layer.get_output(0).dtype == trt.float32
                    and layer.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT)):
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                pinned += 1
    cache_path = os.path.join(OUT, "timing.cache")
    cache = config.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)
    with open(cache_path, "wb") as f:
        f.write(config.get_timing_cache().serialize())
    return time.time() - t0


@torch.no_grad()
def real_activations(model, data, rows):
    """Per-layer inputs/outputs and rotary tables of the unmodified model over the padded prefill window."""
    lm = model.vlm.model.language_model
    tok = data["tokenized_data"]
    ids = model.fuse_traj_tokens(tok["input_ids"], {
        "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]})
    pad = torch.full((1, rows - ids.shape[1]), model.tokenizer.pad_token_id, device=ids.device)
    ids = torch.cat([ids, pad], dim=1)
    rec = {"in": [], "out": []}
    hooks = []
    for layer in lm.layers:
        def pre(_m, args, kwargs, rec=rec):
            rec["in"].append(args[0].clone())
            rec["pos"] = kwargs["position_embeddings"]

        def post(_m, _a, _k, out, rec=rec):
            rec["out"].append((out[0] if isinstance(out, tuple) else out).clone())

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.register_forward_hook(post, with_kwargs=True))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        pixel_values=tok["pixel_values"], image_grid_thw=tok["image_grid_thw"],
                        use_cache=False)
    for h in hooks:
        h.remove()
    return rec
