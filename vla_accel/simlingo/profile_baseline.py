"""Stage profile of the untouched SimLingo model on one Bench2Drive frame.

Splits each forward into vision encode, LLM forwards (decode + driving head) and other.
"""

import argparse
import time

import torch

from vla_accel.common import b2d, jetson
from vla_accel.simlingo import inputs


def _nparams(m):
    """Parameter count."""
    return sum(p.numel() for p in m.parameters())


def main():
    ap = inputs.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()
    jetson.warn_if_unpinned()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("loading model...", flush=True)
    t0 = time.time()
    model, ctx = inputs.build_model(args)
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"  vision_model  : {_nparams(model.vision_model) / 1e6:7.1f} M")
    print(f"  language_model: {_nparams(model.language_model) / 1e6:7.1f} M")
    print(f"  total         : {_nparams(model) / 1e6:7.1f} M")
    print(f"  predict_language = {model.predict_language}")

    clip = inputs.resolve_clip(args)
    idx = b2d.frame_indices(clip, 1, args.stride, args.start)[0]
    rgb, speed, tps = inputs.load_frame(clip, idx, args.jpeg_quality)
    example, n_tok = inputs.build_example(ctx, rgb, speed, tps)
    print(f"  frame {idx:05d} of {clip}: image {rgb.shape[1]}x{rgb.shape[0]}, "
          f"speed {speed:.2f} m/s, prompt tokens {n_tok}")

    import simlingo_training.models.language_model.llm as llmmod
    stats = {"calls": 0, "t": 0.0, "seqlens": []}
    vstats = {"calls": 0, "t": 0.0}
    orig_fwd = llmmod.LLM.forward

    def timed_fwd(self, embeddings, *a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_fwd(self, embeddings, *a, **kw)
        torch.cuda.synchronize()
        stats["calls"] += 1
        stats["t"] += time.perf_counter() - t
        stats["seqlens"].append(embeddings.shape[1])
        return out

    llmmod.LLM.forward = timed_fwd
    ie = model.vision_model.image_encoder
    orig_rp = ie.replace_placeholder_tokens

    def timed_rp(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_rp(*a, **kw)
        torch.cuda.synchronize()
        vstats["calls"] += 1
        vstats["t"] += time.perf_counter() - t
        return out

    ie.replace_placeholder_tokens = timed_rp

    def run(tag, n):
        for i in range(n):
            for d in (stats, vstats):
                d["calls"], d["t"] = 0, 0.0
            stats["seqlens"] = []
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, lang = model(example)
            torch.cuda.synchronize()
            total = time.perf_counter() - t
            sl = stats["seqlens"]
            print(f"[{tag}] iter {i}: total {total * 1000:8.1f} ms | "
                  f"vision {vstats['t'] * 1000:7.1f} ms ({vstats['calls']}x) | "
                  f"llm {stats['t'] * 1000:8.1f} ms in {stats['calls']} fwd "
                  f"(seqlen {min(sl) if sl else 0}->{max(sl) if sl else 0}) | "
                  f"other {(total - stats['t'] - vstats['t']) * 1000:6.1f} ms "
                  f"| gpu {inputs.gpu_mhz()} MHz", flush=True)
            if i == n - 1 and lang:
                print(f"        language: {lang[0][:160]!r}")

    from PIL import Image as PILImage
    from simlingo_training.utils.internvl2_utils import dynamic_preprocess

    use_thumb = ctx.cfg.model.vision_model.use_global_img
    t = time.perf_counter()
    for _ in range(5):
        tiles = dynamic_preprocess(PILImage.fromarray(rgb), image_size=inputs.TILE,
                                   use_thumbnail=use_thumb, max_num=2)
        torch.stack([ctx.transform(x) for x in tiles])
    print(f"\n[cpu] image preprocess (tile + normalise): {(time.perf_counter() - t) / 5 * 1000:.1f} ms")

    print()
    run("as-is", args.iters)

    print("\n--- predict_language = False (driving head only, no language) ---")
    model.predict_language = False
    try:
        run("no-lang", args.iters)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
