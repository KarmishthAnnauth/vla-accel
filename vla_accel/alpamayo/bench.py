"""FastAlpamayo against the unmodified model on Bench2Drive frames: verify(), then timing.

Timing regimes: steady (scene repeats, draft accepted), changing (scenes alternate), no draft.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vla_accel.alpamayo import common
from vla_accel.alpamayo.speedups import FastAlpamayo

STAGES = ("prepare", "vit", "prefill", "verify", "decode", "expert", "to_traj")


def timed(fn):
    """(result, synchronised wall ms) of fn()."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def main() -> None:
    """Run the check and the timing regimes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--max-gen", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--tie-margin", type=float, default=0.25)
    ap.add_argument("--trt-dir", default="")
    common.add_b2d_args(ap)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, processor = common.load_model()
    model.diffusion.num_inference_steps = args.steps
    scenes = common.scenes(processor, args)
    n_scenes = len(scenes)
    fast = FastAlpamayo(model, max_new_tokens=args.max_gen, capture=not args.no_capture, tie_margin=args.tie_margin,
                        trt_dir=args.trt_dir, log=print, profile=True)
    report = {"verify": [], "timing": {}}

    print("\n== correctness: greedy tokens, same noise ==")
    for v, data in enumerate(scenes):
        r = fast.verify(data, seed=v, max_generation_length=args.max_gen)
        report["verify"].append(r)
        print(f"scene {v}: text cold/draft equal = {r['cold_text_equal']}/{r['draft_text_equal']}  "
              f"max traj dev cold {r['cold_max_dev_m']:.4f} m, draft {r['draft_max_dev_m']:.4f} m  "
              f"draft pass {r['draft_stats']}\n         cot={r['cot']!r}")

    devs = [max(r["cold_max_dev_m"], r["draft_max_dev_m"]) for r in report["verify"]]
    texts = sum(r["cold_text_equal"] + r["draft_text_equal"] for r in report["verify"])
    print(f"summary over {n_scenes} scenes: reasoning text equal to the reference in {texts}/{2 * n_scenes} "
          f"passes; trajectory max |diff| mean {np.mean(devs):.3f} m, median {np.median(devs):.3f} m, "
          f"worst {max(devs):.3f} m")
    print("\n== logit noise between paths (strict argmax) ==")
    margin = fast.tie_margin
    fast.tie_margin = 0.0
    for v, data in enumerate(scenes):
        traces = []
        fast.reset()
        fast.speculate = False
        for _ in range(2):
            fast.trace_logits = []
            fast.sample(data, greedy=True, max_generation_length=args.max_gen)
            traces.append((fast.trace_logits, list(fast.tokens)))
            fast._library = [list(fast.tokens)]
            fast.speculate = True
        fast.trace_logits = None
        (seq, seq_tok), (win, win_tok) = traces
        n = min(len(seq), len(win))
        same = 0
        while same < n and seq_tok[same] == win_tok[same]:
            same += 1
        rows = range(min(n, same + 1))
        diff = max(float((seq[j] - win[j]).nan_to_num(0, 0, 0).abs().max()) for j in rows)
        gaps = [float(seq[j].topk(2).values.diff().abs()) for j in rows]
        report["verify"][v]["logit_noise"] = diff
        report["verify"][v]["min_top2_gap"] = min(gaps)
        print(f"scene {v}: max |dlogit| {diff:.3f} over {len(rows)} positions, smallest top-2 gap "
              f"{min(gaps):.3f}, tokens agree for {same}/{len(seq_tok)}")
    fast.tie_margin = margin

    def regime(name, pick_scene, speculate, greedy=True):
        """Mean per-stage timing over args.runs frames after two warm-ups."""
        fast.speculate = speculate
        fast.reset()
        rows = []
        for i in range(args.runs + 2):
            data = scenes[pick_scene(i)]
            torch.manual_seed(i)
            torch.cuda.manual_seed_all(i)
            _, ms = timed(lambda: fast.sample(data, greedy=greedy, max_generation_length=args.max_gen))
            if i >= 2:
                rows.append({**fast.timing, "total": ms, **fast.stats})
        mean = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in rows[0]}
        report["timing"][name] = mean
        stages = " ".join(f"{s}={mean.get(s, 0):.0f}" for s in STAGES)
        print(f"{name:<22} total {mean['total']:7.1f} ms | {stages} | tokens {mean['tokens']:.1f} "
              f"accepted {mean['accepted']:.1f} decode steps {mean['decode_steps']:.1f}")

    print("\n== fast path timing (ms, mean) ==")
    regime("steady, greedy", lambda i: 0, True)
    regime("changing, greedy", lambda i: i % n_scenes, True)
    regime("no draft, greedy", lambda i: i % n_scenes, False)
    regime("steady, sampled", lambda i: 0, True, greedy=False)
    regime("no draft, sampled", lambda i: i % n_scenes, False, greedy=False)
    fast.speculate = True

    if not args.skip_reference:
        print("\n== reference (unmodified model), same scenes ==")
        for label, kw in (("greedy (top_k=1)", {"top_p": 1.0, "temperature": 1.0, "top_k": 1}),
                          ("sampled 0.98/0.6", {"top_p": 0.98, "temperature": 0.6})):
            rows, toks = [], []
            for i in range(args.runs + 1):
                data = scenes[i % n_scenes]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    (_, _, extra), ms = timed(
                        lambda: model.sample_trajectories_from_data_with_vlm_rollout(
                            data=data, num_traj_samples=1, max_generation_length=args.max_gen,
                            return_extra=True, **kw))
                if i >= 1:
                    rows.append(ms)
                    toks.append(len(model.tokenizer(str(extra["cot"].reshape(-1)[0]))["input_ids"]))
            report["timing"][f"reference {label}"] = {"total": float(np.mean(rows))}
            print(f"reference {label:<18} total {np.mean(rows):7.1f} ms  (~{np.mean(toks):.0f} cot tokens)")

    print(f"\ngpu {common.gpu_mhz()} MHz, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
