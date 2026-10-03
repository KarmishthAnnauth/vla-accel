#!/usr/bin/env python3
"""Per-op cost inside one LLM layer and one ViT layer (CUDA-graph replay of
each op alone, so dispatch is excluded): where does compute go beyond GEMMs?"""
import time, torch, common as C
import torch.nn.functional as F
from flash_attn import flash_attn_func
agent = C.build_agent()
vlm = agent.backbone.model; lm = vlm.language_model
L = lm.model.layers[5]; A = L.self_attn

def gbench(fn, n=30):
    with torch.no_grad():
        fn(); fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = fn()
        g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): g.replay()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

N = 2536; bf = torch.bfloat16
x = torch.randn(1, N, 1536, device="cuda", dtype=bf)
q = torch.randn(1, N, 12, 128, device="cuda", dtype=bf); k = torch.randn(1, N + 291, 2, 128, device="cuda", dtype=bf)
cos = torch.randn(1, N, 1, 128, device="cuda", dtype=bf)
g1 = torch.randn(1, N, 8960, device="cuda", dtype=bf)
def rot(t): return torch.cat((-t[..., 64:], t[..., :64]), dim=-1)
rows = [
    ("input_layernorm (RMSNorm via fp32)", lambda: L.input_layernorm(x)),
    ("q_proj", lambda: A.q_proj(x)), ("k_proj", lambda: A.k_proj(x)), ("v_proj", lambda: A.v_proj(x)),
    ("RoPE on q", lambda: q * cos + rot(q) * cos),
    ("flash_attn causal 2536x2827 GQA", lambda: flash_attn_func(q, k, k, causal=True)),
    ("o_proj", lambda: A.o_proj(x)),
    ("residual add", lambda: x + x),
    ("gate_proj", lambda: L.mlp.gate_proj(x)), ("up_proj", lambda: L.mlp.up_proj(x)),
    ("silu(g)*u", lambda: F.silu(g1) * g1),
    ("down_proj", lambda: L.mlp.down_proj(g1)),
    ("whole mlp", lambda: L.mlp(x)),
]
tot = 0
print("--- LLM layer, 2536 tokens")
for name, fn in rows:
    ms = gbench(fn); print(f"  {name:36s} {ms:6.2f} ms")
print("  => x28 layers measured end-to-end: 392 ms = 14.0 ms/layer")

V = vlm.vision_model.encoder.layers[5]
xv = torch.randn(9, 1025, 1024, device="cuda", dtype=bf)
hv = torch.randn(9, 1025, 4096, device="cuda", dtype=bf)
print("--- ViT layer, 9 tiles x 1025 tokens  (measured end-to-end: 297 ms = 12.4 ms/layer)")
for name, fn in [
    ("norm1 (LayerNorm bf16)", lambda: V.norm1(xv)),
    ("attn.qkv linear", lambda: V.attn.qkv(xv)),
    ("attn whole (qkv+flash+proj)", lambda: V.attn(xv)),
    ("attn.proj", lambda: V.attn.proj(xv)),
    ("* ls1 + residual", lambda: xv + xv * V.ls1),
    ("mlp.fc1", lambda: V.mlp.fc1(xv)), ("mlp.act (GELU)", lambda: V.mlp.act(hv)), ("mlp.fc2", lambda: V.mlp.fc2(hv)),
    ("whole layer", lambda: V(xv)),
]:
    ms = gbench(fn); print(f"  {name:36s} {ms:6.2f} ms")
print("qk_normalization:", V.attn.qk_normalization, "| act:", type(V.mlp.act).__name__, "| norm:", type(V.norm1).__name__)
