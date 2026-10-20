"""Where the time of one decoder layer / one ViT block goes, op by op (eager, synchronised).

Shapes are the fast path's: 860-row prefill window, 64-row expert, 2880 ViT patches.

    ./run.sh opbench.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

import common


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def table(title, rows, layers):
    total = sum(ms for _, ms in rows)
    print(f"\n{title}: {total:.2f} ms/layer -> {total * layers:.0f} ms for {layers} layers")
    for name, ms in rows:
        print(f"  {name:<34}{ms:7.3f} ms  {100 * ms / total:5.1f} %")


@torch.no_grad()
def main() -> None:
    model, _ = common.load_model()
    bf = torch.bfloat16
    with torch.autocast("cuda", dtype=bf):
        lm = model.vlm.model.language_model
        layer = lm.layers[5]
        at, mlp = layer.self_attn, layer.mlp
        W = 860
        h = torch.randn(1, W, 4096, device="cuda", dtype=bf)
        pos = torch.arange(W, device="cuda").view(1, 1, -1).expand(3, 1, -1)
        cos, sin = lm.rotary_emb(h, pos)
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(1, W, -1, 128)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(1, W, -1, 128)).transpose(1, 2)
        v = at.v_proj(x).view(1, W, -1, 128).transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        a = F.scaled_dot_product_attention(qr, kr, v, is_causal=True, enable_gqa=True)
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        act = mlp.act_fn(g) * u
        down_t = mlp.down_proj.weight.t().contiguous()
        rows = [
            ("input RMSNorm", bench(lambda: layer.input_layernorm(h))),
            ("q_proj", bench(lambda: at.q_proj(x))),
            ("k_proj + v_proj", bench(lambda: (at.k_proj(x), at.v_proj(x)))),
            ("q_norm + k_norm", bench(lambda: (at.q_norm(at.q_proj(x).view(1, W, -1, 128)),))
             - bench(lambda: at.q_proj(x))
             + bench(lambda: at.k_norm(k))),
            ("rope", bench(lambda: apply_rotary_pos_emb(q, k, cos, sin))),
            ("sdpa causal gqa", bench(lambda: F.scaled_dot_product_attention(
                qr, kr, v, is_causal=True, enable_gqa=True))),
            ("o_proj (+reshape)", bench(lambda: at.o_proj(a.transpose(1, 2).reshape(1, W, -1)))),
            ("residual add x2", 2 * bench(lambda: h + x)),
            ("post RMSNorm", bench(lambda: layer.post_attention_layernorm(h))),
            ("gate_proj", bench(lambda: mlp.gate_proj(x))),
            ("up_proj", bench(lambda: mlp.up_proj(x))),
            ("silu(gate) * up", bench(lambda: mlp.act_fn(g) * u)),
            ("down_proj F.linear", bench(lambda: mlp.down_proj(act))),
        ]
        table("VLM layer, 860 rows", rows, len(lm.layers))
        print(f"  (down_proj via W^T mm: {bench(lambda: torch.mm(act[0], down_t)):.3f} ms; "
              f"sdpa with repeated kv: "
              f"{bench(lambda: F.scaled_dot_product_attention(qr, kr.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True)):.3f} ms; "
              f"whole layer eager: {bench(lambda: layer(h, position_embeddings=(cos, sin))):.2f} ms)")
        del down_t

        ex = model.expert
        layer = ex.layers[5]
        at, mlp = layer.self_attn, layer.mlp
        n = 64
        h = torch.randn(1, n, 2048, device="cuda")
        x = layer.input_layernorm(h)
        g, u = mlp.gate_proj(x), mlp.up_proj(x)
        act = mlp.act_fn(g) * u
        rows = [
            ("input RMSNorm", bench(lambda: layer.input_layernorm(h))),
            ("q/k/v proj", bench(lambda: (at.q_proj(x), at.k_proj(x), at.v_proj(x)))),
            ("q_norm + k_norm", bench(lambda: (at.q_norm(at.q_proj(x).view(1, n, -1, 128)),
                                               at.k_norm(at.k_proj(x).view(1, n, -1, 128))))
             - bench(lambda: (at.q_proj(x), at.k_proj(x)))),
            ("o_proj", bench(lambda: at.o_proj(x))),
            ("post RMSNorm", bench(lambda: layer.post_attention_layernorm(h))),
            ("gate + up", bench(lambda: (mlp.gate_proj(x), mlp.up_proj(x)))),
            ("silu(gate) * up", bench(lambda: mlp.act_fn(g) * u)),
            ("down_proj", bench(lambda: mlp.down_proj(act))),
        ]
        table("expert layer, 64 rows (without attention)", rows, len(ex.layers))

        vis = model.vlm.model.visual
        blk = vis.blocks[5]
        N, n_img, s = 2880, 4, 720
        x = torch.randn(N, 1152, device="cuda", dtype=bf)
        cos = torch.randn(N, 72, device="cuda")
        sin = torch.randn(N, 72, device="cuda")
        xn = blk.norm1(x)
        qkv = blk.attn.qkv(xn).reshape(N, 3, 16, -1).permute(1, 0, 2, 3)
        q, k = apply_rotary_pos_emb_vision(qkv[0], qkv[1], cos, sin)
        q, k, v = (t.reshape(n_img, s, 16, -1).transpose(1, 2) for t in (q, k, qkv[2]))
        a = F.scaled_dot_product_attention(q, k, v)
        f1 = blk.mlp.linear_fc1(xn)
        fc1_t = blk.mlp.linear_fc1.weight.t().contiguous()
        fc2_t = blk.mlp.linear_fc2.weight.t().contiguous()
        qkv_t = blk.attn.qkv.weight.t().contiguous()
        rows = [
            ("norm1 + norm2 (LayerNorm)", 2 * bench(lambda: blk.norm1(x))),
            ("qkv", bench(lambda: blk.attn.qkv(xn))),
            ("rope (fp32 round trip)", bench(lambda: apply_rotary_pos_emb_vision(qkv[0], qkv[1], cos, sin))),
            ("sdpa 4x16x720x720", bench(lambda: F.scaled_dot_product_attention(q, k, v))),
            ("proj (+reshape)", bench(lambda: blk.attn.proj(a.transpose(1, 2).reshape(N, -1)))),
            ("fc1", bench(lambda: blk.mlp.linear_fc1(xn))),
            ("gelu", bench(lambda: blk.mlp.act_fn(f1))),
            ("fc2", bench(lambda: blk.mlp.linear_fc2(f1))),
            ("residual add x2", 2 * bench(lambda: x + xn)),
        ]
        table("ViT block, 2880 patches", rows, len(vis.blocks))
        print(f"  (W^T addmm: qkv {bench(lambda: torch.addmm(blk.attn.qkv.bias, xn, qkv_t)):.3f}, "
              f"fc1 {bench(lambda: torch.addmm(blk.mlp.linear_fc1.bias, xn, fc1_t)):.3f}, "
              f"fc2 {bench(lambda: torch.addmm(blk.mlp.linear_fc2.bias, f1, fc2_t)):.3f} ms; "
              f"fp16 fc1 {bench(lambda: F.linear(xn.half(), blk.mlp.linear_fc1.weight.half())):.3f} incl. casts)")


if __name__ == "__main__":
    main()
