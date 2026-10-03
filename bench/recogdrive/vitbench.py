import time, torch, common as C
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_qkvpacked_func
agent = C.build_agent()
vlm = agent.backbone.model; V = vlm.vision_model.encoder.layers[5]; bf = torch.bfloat16
def gbench(fn, n=30):
    with torch.no_grad():
        for _ in range(3): fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = fn()
        g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): g.replay()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3, out
qkv = torch.randn(9, 1025, 3, 16, 64, device="cuda", dtype=bf)
a_ms, a = gbench(lambda: V.attn.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=False)[0])
b_ms, b = gbench(lambda: flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
print(f"vit attention: stock varlen-packed {a_ms:.2f} ms | dense flash_attn_func {b_ms:.2f} ms  {'bit-identical' if torch.equal(a, b) else 'differs'}")
qc, kc, vc = (qkv[:, :, i].contiguous() for i in range(3))
c_ms, c = gbench(lambda: flash_attn_func(qc, kc, vc)); print(f"   dense on contiguous q,k,v {c_ms:.2f} ms  {'bit-identical' if torch.equal(a, c) else 'differs'}")
h = torch.randn(9, 1025, 4096, device="cuda", dtype=bf) * 2
every = torch.arange(1 << 16, device="cuda", dtype=torch.int32).to(torch.int16).view(bf)
lut = V.mlp.act(every)
def gelu_lut(t): return lut[t.view(torch.int16).to(torch.int32) & 0xFFFF]
e_ms, e = gbench(lambda: V.mlp.act(h)); l_ms, l = gbench(lambda: gelu_lut(h))
print(f"gelu eager {e_ms:.2f} ms | LUT {l_ms:.2f} ms  {'bit-identical' if torch.equal(e, l) else 'differs'}")
cl = torch.compile(gelu_lut); l2_ms, l2 = gbench(lambda: cl(h))
print(f"   LUT compiled {l2_ms:.2f} ms  {'bit-identical' if torch.equal(e, l2) else 'differs'}")
ln_ms, ln = gbench(lambda: V.norm1(h[..., :1024].contiguous())); print(f"layernorm eager {ln_ms:.2f} ms")
