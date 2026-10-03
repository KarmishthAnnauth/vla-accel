#!/usr/bin/env python3
import time, torch, common as C
import torch.nn.functional as F
agent = C.build_agent()
vlm = agent.backbone.model; lm = vlm.language_model
L = lm.model.layers[5]; A = L.self_attn; V = vlm.vision_model.encoder.layers[5]
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

print("--- F.linear vs mm with pre-transposed contiguous weight")
N = 2536
tot_a = tot_b = 0
for name, lin, rows, mult in [("llm q_proj", A.q_proj, N, 28), ("llm k_proj", A.k_proj, N, 28), ("llm v_proj", A.v_proj, N, 28),
                        ("llm o_proj", A.o_proj, N, 28), ("llm gate_proj", L.mlp.gate_proj, N, 28), ("llm up_proj", L.mlp.up_proj, N, 28),
                        ("llm down_proj", L.mlp.down_proj, N, 28),
                        ("vit qkv", V.attn.qkv, 9225, 24), ("vit proj", V.attn.proj, 9225, 24), ("vit fc1", V.mlp.fc1, 9225, 24), ("vit fc2", V.mlp.fc2, 9225, 24),
                        ("mlp1[1]", vlm.mlp1[1], 2304, 1), ("mlp1[3]", vlm.mlp1[3], 2304, 1)]:
    x = torch.randn(rows, lin.in_features, device="cuda", dtype=bf)
    Wt = lin.weight.t().contiguous()
    a_ms, a = gbench(lambda: lin(x))
    if lin.bias is None:
        b_ms, b = gbench(lambda: torch.mm(x, Wt))
    else:
        b_ms, b = gbench(lambda: torch.addmm(lin.bias, x, Wt))
    tot_a += a_ms * mult; tot_b += b_ms * mult
    print(f"  {name:14s} {lin.in_features:5d}->{lin.out_features:5d} x{rows}: linear {a_ms:5.2f} ms | mm(Wt) {b_ms:5.2f} ms  {'bit-identical' if torch.equal(a, b) else 'DIFFERS rel %.4f' % ((a.float()-b.float()).norm()/a.float().norm()).item()}")
print(f"  whole model: linear {tot_a:.0f} ms -> mm(Wt) {tot_b:.0f} ms")

print("--- fused kernels that reproduce eager's rounding points")
x = torch.randn(1, N, 1536, device="cuda", dtype=bf)
w = L.input_layernorm.weight
def rms_eager(t, w, eps=1e-6):
    d = t.dtype; t = t.to(torch.float32)
    v = t.pow(2).mean(-1, keepdim=True)
    return w * (t * torch.rsqrt(v + eps)).to(d)
def rms_tail(t, r, w):
    y = (t.to(torch.float32) * r).to(torch.bfloat16)
    return (w.to(torch.float32) * y.to(torch.float32)).to(torch.bfloat16)
rms_tail_c = torch.compile(rms_tail)
def rms_split(t, w, eps=1e-6):
    v = t.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return rms_tail_c(t, torch.rsqrt(v + eps), w)
def rms_full(t, w, eps: float = 1e-6):
    tf = t.to(torch.float32)
    r = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)
    y = (tf * r).to(torch.bfloat16)
    return (w.to(torch.float32) * y.to(torch.float32)).to(torch.bfloat16)
rms_full_c = torch.compile(rms_full)
e_ms, e = gbench(lambda: rms_eager(x, w))
for name, fn in [("split (eager reduction + fused tail)", lambda: rms_split(x, w)), ("fully fused, explicit rounding", lambda: rms_full_c(x, w))]:
    ms, o = gbench(fn); print(f"  rmsnorm {name:38s} {ms:5.2f} ms (eager {e_ms:.2f})  {'bit-identical' if torch.equal(e, o) else 'differs in %d elements' % (e != o).sum().item()}")

def rot(t): return torch.cat((-t[..., 64:], t[..., :64]), dim=-1)
def rope_eager(q, cos, sin): return (q * cos) + (rot(q) * sin)
def rope_exact(q, cos, sin):
    f, b = torch.float32, torch.bfloat16
    qf = q.to(f)
    a = (qf * cos.to(f)).to(b).to(f)
    c = (rot(qf) * sin.to(f)).to(b).to(f)
    return (a + c).to(b)
rope_c = torch.compile(rope_exact)
q = torch.randn(1, N, 12, 128, device="cuda", dtype=bf)
cos = torch.randn(1, N, 1, 128, device="cuda", dtype=bf); sin = torch.randn(1, N, 1, 128, device="cuda", dtype=bf)
e_ms, e = gbench(lambda: rope_eager(q, cos, sin)); ms, o = gbench(lambda: rope_c(q, cos, sin))
print(f"  rope    explicit rounding                      {ms:5.2f} ms (eager {e_ms:.2f})  {'bit-identical' if torch.equal(e, o) else 'differs in %d elements' % (e != o).sum().item()}")

def actmul_exact(g, u):
    f, b = torch.float32, torch.bfloat16
    s = F.silu(g.to(f)).to(b).to(f)
    return (s * u.to(f)).to(b)
act_c = torch.compile(actmul_exact)
g = torch.randn(1, N, 8960, device="cuda", dtype=bf) * 3; u = torch.randn(1, N, 8960, device="cuda", dtype=bf)
e_ms, e = gbench(lambda: F.silu(g) * u); ms, o = gbench(lambda: act_c(g, u))
print(f"  silu*u  explicit rounding                      {ms:5.2f} ms (eager {e_ms:.2f})  {'bit-identical' if torch.equal(e, o) else 'differs in %d of %d elements' % ((e != o).sum().item(), e.numel())}")

xv = torch.randn(9, 1025, 1024, device="cuda", dtype=bf); av = torch.randn(9, 1025, 1024, device="cuda", dtype=bf)
hv = torch.randn(9, 1025, 4096, device="cuda", dtype=bf)
def ls_eager(x, a, ls): return x + a * ls
def ls_exact(x, a, ls):
    f, b = torch.float32, torch.bfloat16
    return (x.to(f) + (a.to(f) * ls.to(f)).to(b).to(f)).to(b)
ls_c = torch.compile(ls_exact)
e_ms, e = gbench(lambda: ls_eager(xv, av, V.ls1)); ms, o = gbench(lambda: ls_c(xv, av, V.ls1))
print(f"  vit x + a*ls                                   {ms:5.2f} ms (eager {e_ms:.2f})  {'bit-identical' if torch.equal(e, o) else 'differs in %d elements' % (e != o).sum().item()}")
gelu_c = torch.compile(lambda t: F.gelu(t.to(torch.float32)).to(torch.bfloat16))
e_ms, e = gbench(lambda: V.mlp.act(hv)); ms, o = gbench(lambda: gelu_c(hv))
print(f"  vit gelu                                       {ms:5.2f} ms (eager {e_ms:.2f})  {'bit-identical' if torch.equal(e, o) else 'differs in %d of %d elements' % ((e != o).sum().item(), e.numel())}")
ln_ms, _ = gbench(lambda: V.norm1(xv)); print(f"  vit layernorm eager {ln_ms:.2f} ms")
