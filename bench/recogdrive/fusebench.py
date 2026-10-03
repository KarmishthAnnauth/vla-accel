#!/usr/bin/env python3
"""Can the non-GEMM work be fused, and can the slow down_proj GEMM be done better?"""
import time, torch, common as C
import torch.nn.functional as F
agent = C.build_agent()
vlm = agent.backbone.model; lm = vlm.language_model
L = lm.model.layers[5]; A = L.self_attn
bf = torch.bfloat16

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

N = 2536
x = torch.randn(1, N, 1536, device="cuda", dtype=bf)
h = torch.randn(1, N, 8960, device="cuda", dtype=bf)
W = L.mlp.down_proj.weight
print("--- down_proj 8960 -> 1536 on", N, "rows")
ref_ms, ref = gbench(lambda: F.linear(h, W)); print(f"  F.linear (reference)            {ref_ms:5.2f} ms")
Wt = W.t().contiguous()
h2 = h[0]
for name, fn in [
    ("mm(h, Wt contiguous)", lambda: torch.mm(h2, Wt)),
    ("matmul 3d (h @ Wt)", lambda: h @ Wt),
    ("(W @ h^T)^T", lambda: torch.mm(W, h2.t()).t()),
    ("K split in 2", lambda: torch.mm(h2[:, :4480], Wt[:4480]) + torch.mm(h2[:, 4480:], Wt[4480:])),
    ("K split in 4", lambda: sum(torch.mm(h2[:, i:i + 2240], Wt[i:i + 2240]) for i in range(0, 8960, 2240))),
    ("rows split in 2", lambda: torch.cat((F.linear(h2[:1268], W), F.linear(h2[1268:], W)))),
    ("rows 2560 (padded)", None),
]:
    if fn is None:
        hp = torch.randn(2560, 8960, device="cuda", dtype=bf); ms, _ = gbench(lambda: F.linear(hp, W)); print(f"  {name:31s} {ms:5.2f} ms"); continue
    ms, out = gbench(fn)
    same = torch.equal(out.reshape(ref.shape), ref)
    print(f"  {name:31s} {ms:5.2f} ms   {'bit-identical' if same else 'differs: rel %.4f' % ((out.reshape(ref.shape).float() - ref.float()).norm() / ref.float().norm()).item()}")
for rows in (2048, 2304, 2536, 2560, 2816, 3072, 4096):
    hp = torch.randn(rows, 8960, device="cuda", dtype=bf)
    ms, _ = gbench(lambda: F.linear(hp, W)); print(f"  F.linear rows={rows}: {ms:5.2f} ms  {2*rows*8960*1536/ms/1e9:5.1f} TFLOP/s")

print("--- gate+up as one GEMM")
Wgu = torch.cat((L.mlp.gate_proj.weight, L.mlp.up_proj.weight), dim=0)
ms0, r0 = gbench(lambda: (L.mlp.gate_proj(x), L.mlp.up_proj(x)))
ms1, r1 = gbench(lambda: F.linear(x, Wgu).chunk(2, dim=-1))
print(f"  separate {ms0:5.2f} ms | one GEMM {ms1:5.2f} ms  {'bit-identical' if torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]) else 'differs'}")

print("--- fused elementwise (torch.compile) vs eager")
def rms_eager(t, w, eps=1e-6):
    d = t.dtype; t = t.to(torch.float32)
    v = t.pow(2).mean(-1, keepdim=True)
    return w * (t * torch.rsqrt(v + eps)).to(d)
def actmul(g, u): return F.silu(g) * u
def rot(t): return torch.cat((-t[..., 64:], t[..., :64]), dim=-1)
def rope(q, cos, sin): return (q * cos) + (rot(q) * sin)
q = torch.randn(1, N, 12, 128, device="cuda", dtype=bf)
cos = torch.randn(1, N, 1, 128, device="cuda", dtype=bf); sin = torch.randn(1, N, 1, 128, device="cuda", dtype=bf)
g = torch.randn(1, N, 8960, device="cuda", dtype=bf); u = torch.randn(1, N, 8960, device="cuda", dtype=bf)
w = L.input_layernorm.weight
for name, fn, a in [("rmsnorm", rms_eager, (x, w)), ("silu(g)*u", actmul, (g, u)), ("rope(q)", rope, (q, cos, sin))]:
    cf = torch.compile(fn)
    e_ms, e = gbench(lambda: fn(*a))
    try:
        c_ms, c = gbench(lambda: cf(*a))
        same = torch.equal(e, c)
        print(f"  {name:10s} eager {e_ms:5.2f} ms | compiled {c_ms:5.2f} ms   {'bit-identical' if same else 'differs: %d of %d elements, rel %.5f' % ((e != c).sum().item(), e.numel(), ((e.float() - c.float()).norm() / e.float().norm()).item())}")
    except Exception as ex:
        print(f"  {name:10s} eager {e_ms:5.2f} ms | compile failed: {type(ex).__name__}: {str(ex)[:200]}")
