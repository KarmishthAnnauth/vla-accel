import time, torch, common as C
import torch.nn.functional as F
agent = C.build_agent()
from recogdrive_ros.recogdrive_speedups import FastReCogDrive
fast = FastReCogDrive(agent, log=print)
ai = C.make_input("frames/carla_0.bmp", speed=8.0)
for _ in range(3): fast.compute_trajectory(ai)
ah = fast._ah; dit = ah.model
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
ms, vl = gbench(lambda: ah.feature_encoder(fast._ctx)); print(f"feature_encoder 2827x1536->384 fp32   {ms:6.2f} ms")
ms, _ = gbench(lambda: ((vl * fast._ctx_maskf).sum(1) * fast._ctx_inv_len)); print(f"masked mean                           {ms:6.2f} ms")
blk = dit.transformer_blocks[1]; attn = blk.attn
ms, k = gbench(lambda: attn.to_k(vl)); print(f"one to_k 2827x384->384                {ms:6.2f} ms   (x16 per frame)")
ms, kn = gbench(lambda: attn.k_norm(k.view(1, -1, 8, 48).transpose(1, 2))); print(f"one k_norm                            {ms:6.2f} ms   (x8)")
def prep():
    out = []
    for idx, block in enumerate(dit.transformer_blocks):
        if idx % 2 == 0: continue
        a = block.attn
        kk = a.to_k(vl).view(1, -1, a.num_heads, a.head_dim).transpose(1, 2)
        vv = a.to_v(vl).view(1, -1, a.num_heads, a.head_dim).transpose(1, 2)
        out.append((a.k_norm(kk), vv))
    return out
ms, kv = gbench(prep); print(f"all context K/V                       {ms:6.2f} ms")
q = torch.randn(1, 8, 8, 48, device="cuda")
ms, _ = gbench(lambda: F.scaled_dot_product_attention(q, kv[0][0], kv[0][1], attn_mask=fast._ctx_mask)); print(f"one masked cross-attn SDPA             {ms:6.2f} ms   (x40)")
ms, _ = gbench(lambda: F.scaled_dot_product_attention(q, kv[0][0], kv[0][1])); print(f"one unmasked cross-attn SDPA           {ms:6.2f} ms")
x = torch.randn(1, 8, 3, device="cuda"); hist = torch.randn(1, 8, 384, device="cuda"); ego = torch.randn(1, 384, device="cuda")
args = (x, fast._t[0], fast._index[0], fast._time_emb[0], x, hist, hist, ego, kv)
ms, _ = gbench(lambda: fast._ddim_step(*args)); print(f"one step, eager ops in graph          {ms:6.2f} ms   (x5)")
ms, _ = gbench(lambda: fast._step(*args)); print(f"one step, compiled in graph           {ms:6.2f} ms   (x5)")
m = torch.randn(1, 8, 384, device="cuda")
ms, _ = gbench(lambda: blk.attn.to_q(m)); print(f"one 8x384 linear                      {ms:6.3f} ms")
ms, _ = gbench(lambda: blk.norm1(m)); print(f"one RMSNorm on 8x384                  {ms:6.3f} ms")
print("allow_tf32:", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
