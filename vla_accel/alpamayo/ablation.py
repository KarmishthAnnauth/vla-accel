"""Cumulative step ablation of FastAlpamayo on held-out Bench2Drive scenes, one row per process.

Run ``--row 0`` first (it writes the reference), then ``--row 1`` .. ``--row 8``, then ``--summarize``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from vla_accel.alpamayo import common
from vla_accel.alpamayo.speedups import FastAlpamayo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "step_ablation")
_TRT = os.environ.get("ALPAMAYO_TRT_DIR", "/data/alpamayo-1.5-trt")
TRT = {5: _TRT,
       6: os.environ.get("ALPAMAYO_TRT_INT8_DIR", _TRT + "-int8"),
       7: os.environ.get("ALPAMAYO_TRT_INT8_ALL_DIR", _TRT + "-int8-all"),
       8: os.environ.get("ALPAMAYO_TRT_VIT_INT8_DIR", _TRT + "-vit-int8")}
FLAGS = {
    1: (False, False, False),
    2: (True, False, False),
    3: (True, False, True),
    4: (True, True, True),
    5: (True, True, True),
    6: (True, True, True),
    7: (True, True, True),
    8: (True, True, True),
}
NAMES = {0: "reference", 1: "+ own inference loop", 2: "+ CUDA graphs", 3: "+ fused kernels (torch.compile)",
         4: "+ draft decoding", 5: "+ TensorRT fp16", 6: "opt-in: TensorRT INT8 prefill",
         7: "rejected: INT8 prefill incl. down_proj", 8: "rejected: INT8 vision encoder"}
STAGES = ("prepare", "vit", "prefill", "verify", "decode", "expert", "to_traj")
GREEDY = {"top_p": 1.0, "temperature": 1.0, "top_k": 1}
SAMPLED = {"top_p": 0.98, "temperature": 0.6}


def timed(fn):
    """(result, synchronised wall ms) of fn()."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def stats(ms):
    """Mean, sd and count of a list of times."""
    return {"mean": float(np.mean(ms)), "sd": float(np.std(ms)), "n": len(ms)}


def reference_row(model, scenes, args):
    """Row 0: reference trajectories and texts (seeded noise), then reference timing."""
    m = model
    sample = m.diffusion.sample
    refs = []
    for v, data in enumerate(scenes):
        def seeded(*a, _v=v, **k):
            torch.manual_seed(_v)
            torch.cuda.manual_seed_all(_v)
            return sample(*a, **k)
        m.diffusion.sample = seeded
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _, extra = m.sample_trajectories_from_data_with_vlm_rollout(
                    data=data, num_traj_samples=1, max_generation_length=args.max_gen,
                    return_extra=True, **GREEDY)
        finally:
            del m.diffusion.sample
        refs.append({"xyz": xyz.float().cpu(), "cot": str(extra["cot"].reshape(-1)[0])})
        print(f"ref scene {v}: {refs[-1]['cot']!r}", flush=True)
    torch.save(refs, os.path.join(args.out_dir, "reference.pt"))

    result = {"row": 0, "name": NAMES[0], "gpu_mhz": common.gpu_mhz(), "timing": {}}
    for label, kw, passes in (("greedy", GREEDY, args.passes), ("sampled", SAMPLED, 1)):
        ms, toks = [], []
        order = [i % len(scenes) for i in range(1 + passes * len(scenes))]
        for j, s in enumerate(order):
            torch.manual_seed(j)
            torch.cuda.manual_seed_all(j)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                (_, _, extra), t = timed(lambda: m.sample_trajectories_from_data_with_vlm_rollout(
                    data=scenes[s], num_traj_samples=1, max_generation_length=args.max_gen,
                    return_extra=True, **kw))
            if j >= 1:
                ms.append(t)
                toks.append(len(m.tokenizer(str(extra["cot"].reshape(-1)[0]))["input_ids"]))
        result["timing"][label] = {**stats(ms), "cot_tokens": float(np.mean(toks))}
        print(f"reference {label}: {np.mean(ms):.1f} +- {np.std(ms):.1f} ms (~{np.mean(toks):.0f} tokens)", flush=True)
    result["traj"] = {"median_m": 0.0, "worst_same_text_m": 0.0, "same_text": len(scenes), "scenes": len(scenes)}
    return result


def fast_row(row, model, scenes, args):
    """Rows 1-8: deviation from the reference, then changing and steady timing."""
    refs = torch.load(os.path.join(args.out_dir, "reference.pt"))
    capture, speculate, fuse = FLAGS[row]
    fast = FastAlpamayo(model, max_new_tokens=args.max_gen, capture=capture, speculate=speculate,
                        fuse=fuse, trt_dir=TRT.get(row, ""), log=print, profile=True)
    result = {"row": row, "name": NAMES[row], "flags": {"capture": capture, "speculate": speculate,
              "fuse": fuse, "trt_dir": TRT.get(row, "")}, "timing": {}, "scenes": []}

    for v, data in enumerate(scenes):
        fast.reset()
        for p in range(2 if speculate else 1):
            torch.manual_seed(v)
            torch.cuda.manual_seed_all(v)
            noise = torch.randn(1, fast._n_act, fast._act_dim, device="cuda")
            xyz, _, extra = fast.sample(data, greedy=True, max_generation_length=args.max_gen, noise=noise)
        cot = str(extra["cot"].reshape(-1)[0])
        dev = float((xyz.float().cpu() - refs[v]["xyz"]).abs().max())
        same = cot == refs[v]["cot"]
        result["scenes"].append({"dev_m": dev, "same_text": same, "stats": dict(fast.stats)})
        print(f"scene {v}: dev {dev:.4f} m, same text {same}, {fast.stats}", flush=True)
    assert fast.fallbacks == 0, "fast path fell back"
    devs = [s["dev_m"] for s in result["scenes"]]
    same = [s["dev_m"] for s in result["scenes"] if s["same_text"]]
    result["traj"] = {"median_m": float(np.median(devs)), "worst_same_text_m": float(max(same) if same else float("nan")),
                      "worst_m": float(max(devs)), "mean_m": float(np.mean(devs)),
                      "same_text": len(same), "scenes": len(scenes)}

    def run(name, frames, reset_each=None):
        """Time a sequence of (scene, timed?) frames."""
        rows = []
        for j, (s, timed_frame) in enumerate(frames):
            if reset_each is not None and reset_each(j):
                fast.reset()
            torch.manual_seed(j)
            torch.cuda.manual_seed_all(j)
            _, t = timed(lambda: fast.sample(scenes[s], greedy=True, max_generation_length=args.max_gen))
            if timed_frame:
                rows.append({**fast.timing, "total": t, **fast.stats, "scene": s})
        tot = [r["total"] for r in rows]
        stage = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in STAGES + ("tokens", "accepted", "decode_steps")}
        result["timing"][name] = {**stats(tot), "stages": stage, "frames": rows}
        print(f"{name:<9} {np.mean(tot):7.1f} +- {np.std(tot):5.1f} ms | "
              + " ".join(f"{k}={stage[k]:.0f}" for k in STAGES)
              + f" | tokens {stage['tokens']:.1f} accepted {stage['accepted']:.1f} steps {stage['decode_steps']:.1f}",
              flush=True)

    n = len(scenes)
    fast.reset()
    run("changing", [(i % n, i >= 2) for i in range(2 + args.passes * n)])
    if speculate:
        frames = []
        for _ in range(args.passes):
            for s in range(n):
                frames += [(s, False), (s, True), (s, True)]
        run("steady", frames, reset_each=lambda j: j % 3 == 0)
    else:
        result["timing"]["steady"] = result["timing"]["changing"]
    result["gpu_mhz"] = common.gpu_mhz()
    return result


def summarize(out_dir):
    """Print the ablation table from the saved rows."""
    rows = {}
    for r in range(9):
        p = os.path.join(out_dir, f"row_{r}.json")
        if os.path.exists(p):
            rows[r] = json.load(open(p))
    prev = None
    print(f"{'row':<34} {'steady ms':>10} {'sd':>6} {'decr.':>7} {'changing':>9} {'median m':>9} {'worst m':>8} {'text':>6}")
    for r, d in rows.items():
        t = d["timing"]["greedy"] if r == 0 else d["timing"]["steady"]
        ch = d["timing"]["greedy"]["mean"] if r == 0 else d["timing"]["changing"]["mean"]
        base = rows[5] if r >= 6 and 5 in rows else prev
        dec = (base["timing"]["steady" if base["row"] else "greedy"]["mean"] - t["mean"]) if base else float("nan")
        tr = d["traj"]
        print(f"{r} {d['name']:<32} {t['mean']:10.1f} {t['sd']:6.1f} {dec:7.1f} {ch:9.1f} "
              f"{tr['median_m']:9.3f} {tr['worst_same_text_m']:8.3f} {tr['same_text']:>3}/{tr['scenes']}")
        if r < 6:
            prev = d
    if 0 in rows:
        print(f"reference sampled: {rows[0]['timing']['sampled']['mean']:.1f} ms "
              f"(~{rows[0]['timing']['sampled']['cot_tokens']:.0f} tokens), greedy ~{rows[0]['timing']['greedy']['cot_tokens']:.0f} tokens")


def main():
    """Run one row or summarise."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int)
    ap.add_argument("--summarize", action="store_true")
    common.add_b2d_args(ap)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.summarize:
        summarize(args.out_dir)
        return
    print(f"row {args.row}: {NAMES[args.row]}, gpu {common.gpu_mhz()} MHz", flush=True)
    model, processor = common.load_model()
    model.diffusion.num_inference_steps = args.steps
    scenes = common.scenes(processor, args)
    result = reference_row(model, scenes, args) if args.row == 0 else fast_row(args.row, model, scenes, args)
    result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    with open(os.path.join(args.out_dir, f"row_{args.row}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"done, gpu {common.gpu_mhz()} MHz, peak {result['peak_gib']:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
