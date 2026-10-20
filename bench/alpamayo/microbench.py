"""Platform questions behind the Alpamayo fast path, answered on the loaded model.

  1. Which attention path does the expert actually take (it is handed a 4-D mask)?
  2. bf16 GEMM: F.linear vs a pre-transposed weight, at prefill / expert / decode row counts.
  3. How fast can the GPU stream the decoder's weights (the floor of a one-token step)?

    ./run.sh microbench.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import common


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def main() -> None:
    model, processor = common.load_model()
    lm = model.vlm.model.language_model
    print("vlm attn:", lm.config._attn_implementation, "| expert attn:",
          model.expert.config._attn_implementation,
          "| vit attn:", model.vlm.model.visual.config._attn_implementation)
    print("expert layers", len(model.expert.layers), "hidden", model.expert.config.hidden_size,
          "heads", model.expert.config.num_attention_heads,
          "kv heads", model.expert.config.num_key_value_heads)

    seen = {}
    attn = model.expert.layers[0].self_attn
    orig = attn.forward

    def spy(*a, **k):
        m = k.get("attention_mask")
        seen["mask"] = None if m is None else (tuple(m.shape), m.dtype, float(m.min()), float(m.max()))
        seen["kwargs"] = {x: v for x, v in k.items() if x in ("is_causal",)}
        seen["hidden"] = tuple(k["hidden_states"].shape)
        return orig(*a, **k)

    attn.forward = spy
    inputs = common.make_inputs(processor, common.make_payload(1))
    model.diffusion.num_inference_steps = 5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, top_p=1.0, temperature=1.0, num_traj_samples=1,
            max_generation_length=64)
    attn.forward = orig
    print("expert attention sees:", seen)

    print("\nbf16 GEMM, ms per call (F.linear | mm with W^T contiguous | TFLOP/s of the faster)")
    with torch.no_grad():
        for name, lin in (("vlm gate_proj 4096->12288", lm.layers[5].mlp.gate_proj),
                          ("vlm down_proj 12288->4096", lm.layers[5].mlp.down_proj),
                          ("vlm q_proj 4096->4096", lm.layers[5].self_attn.q_proj),
                          ("expert gate 2048->8256", model.expert.layers[5].mlp.gate_proj),
                          ("expert down 8256->2048", model.expert.layers[5].mlp.down_proj)):
            wt = lin.weight.t().contiguous()
            for rows in (1, 64, 828):
                x = torch.randn(rows, lin.in_features, device="cuda", dtype=torch.bfloat16)
                a = bench(lambda: F.linear(x, lin.weight))
                b = bench(lambda: torch.mm(x, wt))
                flop = 2 * rows * lin.in_features * lin.out_features / (min(a, b) * 1e-3) / 1e12
                print(f"  {name:<28} rows={rows:<4} {a:7.3f} | {b:7.3f} | {flop:5.1f}")
            del wt

        ws = [m.weight for layer in lm.layers for m in (
            layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
            layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)]
        xs = {w.shape[1]: torch.randn(1, w.shape[1], device="cuda", dtype=torch.bfloat16) for w in ws}
        nbytes = sum(w.numel() for w in ws) * 2

        def stream():
            for w in ws:
                F.linear(xs[w.shape[1]], w)

        ms = bench(stream, n=10)
        print(f"\none-row pass over all {len(ws)} decoder weights ({nbytes / 2**30:.1f} GiB): "
              f"{ms:.1f} ms -> {nbytes / 1e9 / (ms * 1e-3):.0f} GB/s")
        g = torch.cuda.CUDAGraph()
        stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            stream()
        ms = bench(g.replay, n=10)
        print(f"same under a CUDA graph: {ms:.1f} ms -> {nbytes / 1e9 / (ms * 1e-3):.0f} GB/s")


if __name__ == "__main__":
    main()
