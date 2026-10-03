import torch
bf, f = torch.bfloat16, torch.float32
torch.manual_seed(0)
x = torch.randn(1 << 20, device="cuda"); a = torch.randn(1 << 20, device="cuda", dtype=bf); b = torch.randn(1 << 20, device="cuda", dtype=bf)
def rne(v):
    i = v.view(torch.int32)
    i = (i + 0x7FFF + ((i >> 16) & 1)) & -65536
    return i.view(f)
print("eager f32->bf16 is round-to-nearest-even:", torch.equal(x.to(bf).to(f), rne(x)))
print("eager bf16*bf16 == rne(f32 product):     ", torch.equal((a * b).to(f), rne(a.to(f) * b.to(f))))
print("eager bf16+bf16 == rne(f32 sum):         ", torch.equal((a + b).to(f), rne(a.to(f) + b.to(f))))
c1 = torch.compile(lambda v: v.to(bf))
print("compiled f32->bf16 == eager:             ", torch.equal(c1(x), x.to(bf)))
c2 = torch.compile(lambda p, q: (p.to(f) * q.to(f)).to(bf))
print("compiled (f32 product)->bf16 == eager a*b:", torch.equal(c2(a, b), a * b), (c2(a, b) != a * b).sum().item())
c3 = torch.compile(lambda p, q: p * q)
print("compiled a*b == eager a*b:               ", torch.equal(c3(a, b), a * b), (c3(a, b) != a * b).sum().item())
def two(p, q, r): return (p.to(f) * q.to(f)).to(bf).to(f) + r.to(f)
c4 = torch.compile(lambda p, q, r: two(p, q, r).to(bf))
e = a * b + a
print("compiled roundtrip-in-chain == eager a*b+a:", torch.equal(c4(a, b, a), e), (c4(a, b, a) != e).sum().item())
def two_rne(p, q, r): return rne(rne(p.to(f) * q.to(f)) + r.to(f)).to(bf)
c5 = torch.compile(two_rne)
print("compiled explicit-RNE chain == eager a*b+a:", torch.equal(c5(a, b, a), e), (c5(a, b, a) != e).sum().item())
