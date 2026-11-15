"""W8A8 static INT8 (explicit Q/DQ) TensorRT engines for the VLM prefill and the ViT, calibrated on Bench2Drive.

Subcommands: llm-sim (PyTorch simulation of the quantised prefill), llm-build, vit-int8.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from vla_accel.alpamayo import common
from vla_accel.alpamayo.speedups import DRAFT_CAP
from vla_accel.alpamayo.trt_common import (OUT, DecoderLayer, TrtRunner, Vit, bench, build_engine,
                                           real_activations, rel)

QUANTILES = (0.999, 0.9999, 0.99999, 1.0)
LLM_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def wq(weight: torch.Tensor):
    """Per-output-channel symmetric int8 weight, dequantised."""
    s = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return (weight / s).round().clamp_(-127, 127) * s


def fq(x: torch.Tensor, s: float) -> torch.Tensor:
    """Fake-quantise x to int8 with scale s."""
    return (x / s).round().clamp_(-127, 127) * s


def decoder_linears(layer):
    """(name, linear) of one decoder layer, in execution order."""
    at, mlp = layer.self_attn, layer.mlp
    return [("q_proj", at.q_proj), ("k_proj", at.k_proj), ("v_proj", at.v_proj), ("o_proj", at.o_proj),
            ("gate_proj", mlp.gate_proj), ("up_proj", mlp.up_proj), ("down_proj", mlp.down_proj)]


def vit_linears(vit):
    """(name, linear) of the vision encoder, in execution order."""
    out = []
    vis = vit.vis
    for i, blk in enumerate(vis.blocks):
        out += [(f"B{i}.qkv", blk.attn.qkv), (f"B{i}.proj", blk.attn.proj),
                (f"B{i}.fc1", blk.mlp.linear_fc1), (f"B{i}.fc2", blk.mlp.linear_fc2)]
        if i in vis.deepstack_visual_indexes:
            m = vis.deepstack_merger_list[vis.deepstack_visual_indexes.index(i)]
            out += [(f"D{i}.fc1", m.linear_fc1), (f"D{i}.fc2", m.linear_fc2)]
    out += [("M.fc1", vis.merger.linear_fc1), ("M.fc2", vis.merger.linear_fc2)]
    return out


@torch.no_grad()
def calibrate(mods, run):
    """Per-linear activation scale minimising output error; ``run()`` feeds the calibration inputs."""
    cand, err, n_calls = {}, {}, {}

    def h1(name):
        def hook(m, inp):
            a = inp[0].abs().flatten().float()
            sub = a[torch.randint(0, a.numel(), (min(a.numel(), 2_000_000),), device=a.device)]
            q = torch.quantile(sub, torch.tensor(QUANTILES[:-1], device=a.device))
            c = torch.cat((q, a.max()[None]))
            cand[name] = cand.get(name, 0) + c
            n_calls[name] = n_calls.get(name, 0) + 1
        return hook

    hs = [m.register_forward_pre_hook(h1(n)) for n, m in mods]
    run()
    for h in hs:
        h.remove()
    for n in cand:
        cand[n] = cand[n] / n_calls[n]

    def h2(name):
        def hook(m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            x = x[torch.randint(0, x.shape[0], (min(x.shape[0], 512),), device=x.device)]
            w = m.weight.float()
            y = x @ w.t()
            w_q = wq(w)
            e = torch.stack([((fq(x, c / 127) @ w_q.t() - y) ** 2).sum() / (y ** 2).sum()
                             for c in cand[name]])
            err[name] = err.get(name, 0) + e
        return hook

    hs = [m.register_forward_pre_hook(h2(n)) for n, m in mods]
    run()
    for h in hs:
        h.remove()
    scales = {}
    for n, _ in mods:
        i = int(err[n].argmin())
        scales[n] = {"act_scale": float(cand[n][i] / 127), "clip": float(cand[n][i]),
                     "quantile": QUANTILES[i], "max": float(cand[n][-1]),
                     "rel_err": float((err[n][i] / n_calls[n]).sqrt())}
    return scales


def insert_qdq(src: str, dst: str, names, scales, skip=()) -> int:
    """Insert Q/DQ on input and weight of every constant-weight MatMul/Gemm of ``src`` (order = ``names``)."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    new_nodes, k, done = [], 0, 0
    for node in g.node:
        is_lin = (node.op_type in ("MatMul", "Gemm") and len(node.input) > 1
                  and node.input[1] in init and len(init[node.input[1]].dims) == 2)
        if is_lin:
            name = names[k]
            k += 1
            if not any(s in name for s in skip):
                w = numpy_helper.to_array(init[node.input[1]])
                axis = 1 if node.op_type == "MatMul" else 0
                s_w = (np.abs(w).max(axis=1 - axis).clip(1e-8) / 127).astype(np.float32)
                tag = f"qdq{k}"
                for nm, arr in ((f"{tag}_as", np.array(scales[name]["act_scale"], np.float32)),
                                (f"{tag}_az", np.array(0, np.int8)),
                                (f"{tag}_ws", s_w), (f"{tag}_wz", np.zeros(s_w.shape[0], np.int8))):
                    g.initializer.append(numpy_helper.from_array(arr, nm))
                a_in, w_in = node.input[0], node.input[1]
                new_nodes += [
                    helper.make_node("QuantizeLinear", [a_in, f"{tag}_as", f"{tag}_az"], [f"{tag}_aq"]),
                    helper.make_node("DequantizeLinear", [f"{tag}_aq", f"{tag}_as", f"{tag}_az"], [f"{tag}_adq"]),
                    helper.make_node("QuantizeLinear", [w_in, f"{tag}_ws", f"{tag}_wz"], [f"{tag}_wq"], axis=axis),
                    helper.make_node("DequantizeLinear", [f"{tag}_wq", f"{tag}_ws", f"{tag}_wz"], [f"{tag}_wdq"], axis=axis),
                ]
                node.input[0], node.input[1] = f"{tag}_adq", f"{tag}_wdq"
                done += 1
        new_nodes.append(node)
    assert k == len(names), f"{k} linears in the graph, {len(names)} names"
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data")
    return done


def engine_summary(path: str) -> str:
    """Most common (layer type, precision) pairs of an engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(open(path, "rb").read())
    info = json.loads(engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    kinds = {}
    for layer in info["Layers"]:
        if isinstance(layer, str):
            return "(build with detailed profiling for a per-layer summary)"
        key = (layer.get("LayerType"), (layer.get("Outputs") or [{}])[0].get("Format/Datatype", "?"))
        kinds[key] = kinds.get(key, 0) + 1
    return ", ".join(f"{k[0]}:{k[1]} x{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:8])


def time_engine(path, inputs, ref):
    """(ms, rel-L2 vs ref, output) of an engine on these inputs."""
    runner = TrtRunner(path)
    for k, v in inputs.items():
        runner.inputs[k].copy_(v)
    ms = bench(runner, n=20, warm=5)
    torch.cuda.synchronize()
    out = runner.outputs["h_out" if "h_out" in runner.outputs else "out0"].clone()
    return ms, rel(out, ref), out


def prompts(processor, args, per_clip_calib: int, per_clip_test: int):
    """(calibration, held-out) model inputs from the Bench2Drive clips."""
    calib_clips, held_clips = common.split(args)
    make = lambda clips, n: [common.make_inputs(processor, p)
                             for p in common.b2d_payloads(clips, n, args.stride)]
    return make(calib_clips, per_clip_calib), make(held_clips, per_clip_test)


class Sim:
    """Fake-quantised (SmoothQuant + W8A8) simulation of the VLM prefill."""
    def __init__(self, model):
        self.model = model
        self.lm = model.vlm.model.language_model
        self.mods = [(f"L{i}.{n}", m) for i, layer in enumerate(self.lm.layers)
                     for n, m in decoder_linears(layer)]
        self.c = {}
        self.scale = {}
        self.lin_err = {}
        self.active = set()
        self.hooks = []

    @torch.no_grad()
    def forward(self, data, rows):
        """Prefill one prompt; returns (per-layer K/V, next-token logits)."""
        model = self.model
        tok = data["tokenized_data"]
        ids = model.fuse_traj_tokens(tok["input_ids"], {
            "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]})
        n = ids.shape[1]
        pad = torch.full((1, rows - n), model.tokenizer.pad_token_id, device=ids.device)
        ids = torch.cat([ids, pad], dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(input_ids=ids, attention_mask=torch.ones_like(ids),
                            pixel_values=tok["pixel_values"], image_grid_thw=tok["image_grid_thw"],
                            use_cache=True)
        cache = out.past_key_values
        kv = [(layer.keys[:, :, :n].float(), layer.values[:, :, :n].float()) for layer in cache.layers]
        logits = out.logits[0, n - 1].float()
        logits[model.config.traj_token_start_idx:
               model.config.traj_token_start_idx + model.config.traj_vocab_size] = float("-inf")
        return kv, logits

    def _clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calib, rows, alpha):
        """SmoothQuant factors (alpha < 0: none) and activation scales of every decoder linear."""
        self._clear()
        stat = {}

        def h0(name):
            def hook(m, inp):
                a = inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(0).float()
                stat[name] = torch.maximum(stat[name], a) if name in stat else a
            return hook

        self.hooks = [m.register_forward_pre_hook(h0(n)) for n, m in self.mods]
        for data in calib:
            self.forward(data, rows)
        self._clear()
        by = dict(self.mods)

        def factor(x_max, w_max):
            if alpha < 0:
                return torch.ones_like(x_max)
            return (x_max.clamp_min(1e-5) ** alpha / w_max.clamp_min(1e-5) ** (1 - alpha)).clamp(1e-2, 1e2)

        self.c = {}
        for i in range(len(self.lm.layers)):
            for group in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("down_proj",), ("o_proj",)):
                x_max = stat[f"L{i}.{group[0]}"]
                w_max = torch.stack([by[f"L{i}.{n}"].weight.abs().amax(0).float() for n in group]).amax(0)
                c = torch.ones_like(x_max) if group == ("o_proj",) else factor(x_max, w_max)
                for n in group:
                    self.c[f"L{i}.{n}"] = c
        cand, err, calls = {}, {}, {}

        def h1(name):
            def hook(m, inp):
                a = (inp[0].reshape(-1, inp[0].shape[-1]).float() / self.c[name]).abs().flatten()
                sub = a[torch.randint(0, a.numel(), (min(a.numel(), 2_000_000),), device=a.device)]
                q = torch.quantile(sub, torch.tensor(QUANTILES[:-1], device=a.device))
                cand[name] = cand.get(name, 0) + torch.cat((q, a.max()[None]))
                calls[name] = calls.get(name, 0) + 1
            return hook

        self.hooks = [m.register_forward_pre_hook(h1(n)) for n, m in self.mods]
        for data in calib:
            self.forward(data, rows)
        self._clear()
        subset = calib[::max(1, len(calib) // 4)][:4]

        def h2(name):
            def hook(m, inp):
                c = self.c[name]
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                x = x[torch.randint(0, x.shape[0], (256,), device=x.device)]
                w = m.weight.float()
                y = x @ w.t()
                w_q = wq(w * c[None, :])
                e = torch.stack([((fq(x / c, cc / calls[name] / 127) @ w_q.t() - y) ** 2).sum() / (y ** 2).sum()
                                 for cc in cand[name]])
                err[name] = err.get(name, 0) + e
            return hook

        self.hooks = [m.register_forward_pre_hook(h2(n)) for n, m in self.mods]
        for data in subset:
            self.forward(data, rows)
        self._clear()
        self.scale = {n: float(cand[n][int(err[n].argmin())] / calls[n] / 127) for n, _ in self.mods}
        self.lin_err = {n: float((err[n].min() / len(subset)).sqrt()) for n, _ in self.mods}

    def set(self, active):
        """Fake-quantise these linears (input and weight) from now on."""
        self._clear()
        self.active = set(active)

        def hq(name):
            def hook(m, inp, out):
                c = self.c[name]
                y = F.linear(fq(inp[0].float() / c, self.scale[name]), wq(m.weight.float() * c[None, :]))
                return y.to(out.dtype)
            return hook

        self.hooks = [m.register_forward_hook(hq(n)) for n, m in self.mods if n in self.active]


def llm_build(model, processor, args):
    """All decoder layers as Q/DQ INT8 engines in --out, a trt_dir for FastAlpamayo."""
    lm = model.vlm.model.language_model
    calib, tests = prompts(processor, args, args.calib_per_clip, 1)
    test = tests[0]
    tok = test["tokenized_data"]
    rows = tok["input_ids"].shape[1] + DRAFT_CAP
    sim = Sim(model)
    sim.calibrate(calib, rows, args.alpha)
    sim.set(())
    names = [n for n, _ in sim.mods]
    active = set(choose(sim, names, tuple(x for x in args.skip.split(",") if x), args.down_err))
    rec = real_activations(model, test, rows)
    cos, sin = (t.float() for t in rec["pos"])
    out_dir = args.out
    os.makedirs(os.path.join(out_dir, "llm"), exist_ok=True)
    for old in os.listdir(os.path.join(out_dir, "llm")):
        os.remove(os.path.join(out_dir, "llm", old))
    work = os.path.join(out_dir, "work")
    total, summary = 0.0, []
    for i in range(len(lm.layers)):
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        h_in, h_out = rec["in"][i].float(), rec["out"][i].float()
        engine = os.path.join(out_dir, "llm", f"layer_{i:02d}_int8.engine")
        module = DecoderLayer(lm.layers[i]).cuda().eval()
        layer = module.layer
        at, mlp = layer.self_attn, layer.mlp
        with torch.no_grad():
            c = sim.c[f"L{i}.q_proj"]
            layer.input_layernorm.weight.div_(c)
            for m in (at.q_proj, at.k_proj, at.v_proj):
                m.weight.mul_(c[None, :])
            c = sim.c[f"L{i}.gate_proj"]
            layer.post_attention_layernorm.weight.div_(c)
            for m in (mlp.gate_proj, mlp.up_proj):
                m.weight.mul_(c[None, :])
            c = sim.c[f"L{i}.down_proj"]
            mlp.up_proj.weight.div_(c[:, None])
            mlp.down_proj.weight.mul_(c[None, :])
            exact = rel(module(h_in, cos, sin)[0], h_out)
            torch.onnx.export(module, (h_in, cos, sin), os.path.join(work, "layer.onnx"),
                              opset_version=17, dynamo=False, input_names=["h", "cos", "sin"],
                              output_names=["h_out", "k", "v"])
        del module
        torch.cuda.empty_cache()
        kinds = [n for n in LLM_LINEARS]
        skip = tuple(k for k in kinds if f"L{i}.{k}" not in active)
        scales = {k: {"act_scale": sim.scale[f"L{i}.{k}"]} for k in kinds}
        insert_qdq(os.path.join(work, "layer.onnx"), os.path.join(work, "layer_qdq.onnx"), kinds, scales, skip)
        build_engine(os.path.join(work, "layer_qdq.onnx"), engine, args.level,
                     fp32_names=("norm",), int8=True)
        ms, err, out = time_engine(engine, {"h": h_in, "cos": cos, "sin": sin}, h_out)
        total += ms
        print(f"layer {i:02d}: {ms:6.2f} ms, rel-L2 vs the bf16 model's layer output {err:.4f} "
              f"(rebalanced fp32 module {exact:.4f}), finite {bool(torch.isfinite(out).all())}, "
              f"fp16: {','.join(skip) or 'none'}", flush=True)
        summary.append({"ms": ms, "rel": err, "fp16": list(skip)})
    shutil.rmtree(work, ignore_errors=True)
    if not [f for f in os.listdir(out_dir) if f.startswith("vit_") and f.endswith(".engine")]:
        shutil.copy(os.path.join(OUT, "vit_fp16.engine"), os.path.join(out_dir, "vit_fp16.engine"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"rows": rows, "grid": tok["image_grid_thw"].flatten().tolist(),
                   "precision": f"int8 Q/DQ on {len(active)} of {len(names)} decoder linears, "
                                f"SmoothQuant alpha {args.alpha}, {len(calib)} Bench2Drive prompts",
                   "layers": summary}, f, indent=1)
    print(f"36 layers: {total:.0f} ms in total -> {out_dir}")


def choose(sim, names, skip, down_err: float):
    """Linears to quantise: all not in ``skip``, a down_proj only if its error is below ``down_err``."""
    out = []
    for n in names:
        kind = n.split(".")[1]
        if kind in skip:
            continue
        if kind == "down_proj" and sim.lin_err[n] > down_err:
            continue
        out.append(n)
    return out


def llm_sim(model, processor, args):
    """Sweep SmoothQuant alpha and linear subsets: KV error and next-token agreement on held-out prompts."""
    calib, tests = prompts(processor, args, args.calib_per_clip, 3)
    rows = tests[0]["tokenized_data"]["input_ids"].shape[1] + DRAFT_CAP
    print(f"{len(calib)} calibration prompts, {len(tests)} held-out prompts (other clips)")
    sim = Sim(model)
    sim.set(())
    refs = [sim.forward(d, rows) for d in tests]
    names = [n for n, _ in sim.mods]
    n_layers = len(sim.lm.layers)

    def ms_of(active):
        """Estimated prefill ms from per-linear INT8 savings."""
        saved = {"q_proj": 0.25, "k_proj": 0.1, "v_proj": 0.1, "o_proj": 0.3,
                 "gate_proj": 1.25, "up_proj": 1.25, "down_proj": 1.45}
        return 12.5 * n_layers - sum(saved[n.split(".")[1]] for n in active)

    def evaluate(label, active):
        """Print the error of one quantised subset."""
        sim.set(active)
        kv_err, dl, agree = [], [], 0
        for data, (kv_ref, lg_ref) in zip(tests, refs):
            kv, lg = sim.forward(data, rows)
            kv_err.append(np.mean([rel(k, kr) + rel(v, vr) for (k, v), (kr, vr) in zip(kv, kv_ref)]) / 2)
            top = lg_ref.topk(20).indices
            dl.append(float((lg[top] - lg_ref[top]).abs().max()))
            agree += int(lg.argmax() == lg_ref.argmax())
        print(f"  {label:<40} {len(active):3d} linears  KV rel err {np.mean(kv_err):.4f}  |dlogit| top-20 "
              f"mean {np.mean(dl):4.2f} max {max(dl):5.2f}  first token {agree}/{len(tests)}  "
              f"~prefill {ms_of(active):.0f} ms", flush=True)

    for alpha in (-1.0, 0.5, 0.7, 0.85):
        sim.calibrate(calib, rows, alpha)
        worst = sorted(sim.lin_err.items(), key=lambda kv: -kv[1])[:6]
        n_down = [sum(1 for n in names if n.endswith("down_proj") and sim.lin_err[n] <= t) for t in (0.05, 0.1, 0.2)]
        print(f"\nalpha {alpha}: worst linears " + ", ".join(f"{n} {e:.2f}" for n, e in worst)
              + f"; down_proj with error <= 0.05 / 0.1 / 0.2: {n_down}")
        evaluate("all linears", names)
        evaluate("all but down_proj", choose(sim, names, ("down_proj",), 0))
        evaluate("down_proj where error <= 0.05", choose(sim, names, (), 0.05))
        evaluate("down_proj where error <= 0.1", choose(sim, names, (), 0.1))
        evaluate("down_proj where error <= 0.2", choose(sim, names, (), 0.2))
        evaluate("gate + up only", [n for n in names if n.split(".")[1] in ("gate_proj", "up_proj")])
        evaluate("gate + up + down <= 0.1", choose(sim, names, ("q_proj", "k_proj", "v_proj", "o_proj"), 0.1))
    sim.set(())


def vit_int8(model, processor, args):
    """ViT: simulate quantised linear groups, then build and time one Q/DQ engine."""
    calib, tests = prompts(processor, args, 1, 1)
    calib = [d["tokenized_data"] for d in calib[:args.vit_calib]]
    tests = [d["tokenized_data"] for d in tests[:3]]
    grid = tests[0]["image_grid_thw"]
    vit = Vit(model.vlm.model.visual, grid).cuda().eval()
    mods = vit_linears(vit)
    names = [n for n, _ in mods]
    with torch.no_grad():
        scales = calibrate(mods, lambda: [vit(t["pixel_values"].float()) for t in calib])
        truth = [vit(t["pixel_values"].float()) for t in tests]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = [model.vlm.model.visual(t["pixel_values"], grid_thw=grid)[0].float() for t in tests]
    print(f"yardstick: the bf16 model vs exact fp32: rel-L2 "
          f"{np.mean([rel(b, t[0]) for b, t in zip(bf, truth)]):.4f}")
    worst = sorted(scales.items(), key=lambda kv: -kv[1]["rel_err"])[:6]
    print("largest single-linear output errors: " + ", ".join(f"{n} {v['rel_err']:.3f}" for n, v in worst))
    by = dict(mods)
    orig = {n: m.weight.data.clone() for n, m in mods}

    def simulate(active):
        """Mean rel-L2 (merged, worst deepstack) of the fake-quantised ViT."""
        hooks = []
        for n, m in mods:
            m.weight.data.copy_(orig[n])
            if n in active:
                m.weight.data.copy_(wq(orig[n]))
                hooks.append(m.register_forward_pre_hook(
                    lambda mod, inp, s=scales[n]["act_scale"]: (fq(inp[0], s),)))
        with torch.no_grad():
            outs = [vit(t["pixel_values"].float()) for t in tests]
        for h in hooks:
            h.remove()
        for n, m in mods:
            m.weight.data.copy_(orig[n])
        return (np.mean([rel(o[0], t[0]) for o, t in zip(outs, truth)]),
                np.mean([max(rel(a, b) for a, b in zip(o[1:], t[1:])) for o, t in zip(outs, truth)]))

    groups = {
        "all linears": names,
        "blocks only (no mergers)": [n for n in names if n.startswith("B")],
        "all but fc2": [n for n in names if not n.endswith(".fc2")],
        "blocks but fc2": [n for n in names if n.startswith("B") and not n.endswith(".fc2")],
        "only qkv": [n for n in names if n.endswith(".qkv")],
        "only fc1": [n for n in names if n.startswith("B") and n.endswith(".fc1")],
        "only fc2": [n for n in names if n.startswith("B") and n.endswith(".fc2")],
        "only mergers": [n for n in names if not n.startswith("B")],
    }
    print(f"{'quantised (simulation)':<28} n    rel-L2 vs fp32: merged   deepstack (worst)")
    for label, ns in groups.items():
        a, b = simulate(set(ns))
        print(f"  {label:<26} {len(ns):3d}        {a:.4f}        {b:.4f}", flush=True)

    skip = tuple(x for x in args.skip.split(",") if x)
    work = os.path.join(OUT, "vit_int8_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    x = tests[0]["pixel_values"].float()
    with torch.no_grad():
        torch.onnx.export(vit, (x,), os.path.join(work, "vit.onnx"), opset_version=17, dynamo=False,
                          input_names=["pixels"], output_names=[f"out{i}" for i in range(4)])
    del vit
    torch.cuda.empty_cache()
    done = insert_qdq(os.path.join(work, "vit.onnx"), os.path.join(work, "vit_qdq.onnx"), names, scales, skip)
    os.makedirs(args.out, exist_ok=True)
    engine = os.path.join(args.out, "vit_int8.engine")
    took = build_engine(os.path.join(work, "vit_qdq.onnx"), engine, args.level, int8=True)
    shutil.rmtree(work, ignore_errors=True)
    ms, err, out = time_engine(engine, {"pixels": x}, truth[0][0])
    ms16, err16, _ = time_engine(os.path.join(OUT, "vit_fp16.engine"), {"pixels": x}, truth[0][0])
    print(f"vit INT8 engine ({done} of {len(names)} linears with Q/DQ, skip {skip or 'none'}): {ms:.1f} ms, "
          f"rel-L2 vs fp32 {err:.4f}, built in {took:.0f} s | fp16 engine {ms16:.1f} ms, rel-L2 {err16:.4f}")
    print("    " + engine_summary(engine))


def main() -> None:
    """Dispatch the subcommand."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["llm-build", "llm-sim", "vit-int8"])
    ap.add_argument("--out", default=os.environ.get("ALPAMAYO_TRT_INT8_DIR", OUT + "-int8"))
    ap.add_argument("--skip", default="", help="linears left in fp16, e.g. 'down_proj,up_proj'")
    ap.add_argument("--down-err", type=float, default=0.0,
                    help="quantise a down_proj only if its calibrated output error is below this")
    ap.add_argument("--calib-per-clip", type=int, default=3)
    ap.add_argument("--vit-calib", type=int, default=9, help="vit-int8: calibration frames")
    ap.add_argument("--alpha", type=float, default=-1.0, help="SmoothQuant strength; < 0: off")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--b2d-dir", default=common.b2d.B2D_DIR)
    ap.add_argument("--clips", nargs="*", default=None, help="held-out clip names (default: the split)")
    ap.add_argument("--stride", type=int, default=0, help="frame spacing (0: spread evenly)")
    args = ap.parse_args()
    model, processor = common.load_model()
    if args.cmd == "vit-int8":
        vit_int8(model, processor, args)
    elif args.cmd == "llm-sim":
        llm_sim(model, processor, args)
    else:
        llm_build(model, processor, args)


if __name__ == "__main__":
    main()
