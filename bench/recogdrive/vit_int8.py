#!/usr/bin/env python3
"""INT8 (W8A8, static, Q/DQ) for the linear layers of ReCogDrive's vision encoder.

  vit_int8.py calib            activation ranges -> trt/vit_int8_scales.pt, plus a
                               PyTorch simulation of the quantised encoder (accuracy
                               before any engine is built) and a per-group sensitivity
  vit_int8.py onnx [--skip ..] Q/DQ ONNX from the fp32 ONNX and the scales

Quantised: the input and the weight of every nn.Linear (qkv, proj, fc1, fc2, mlp1).
Left in fp16: patch embedding, LayerNorm, attention matmuls / softmax, GELU, residuals.
Weights: per output channel, max-abs.  Activations: one symmetric scale per layer,
the clip value (a high percentile of |x|) that minimises that layer's output error.
"""
import argparse
import copy
import glob
import os

import numpy as np
import torch

import common as C
from trt_vit import ONNX, OUT, pixel_batches

SCALES = os.path.join(OUT, "vit_int8_scales.pt")
QUANTILES = (0.999, 0.9999, 0.99999, 1.0)


def build_fp32(agent):
    vlm = agent.backbone.model

    class Vit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = copy.deepcopy(vlm.vision_model).float()
            self.mlp1 = copy.deepcopy(vlm.mlp1).float()
            for layer in self.vision.encoder.layers:
                layer.attn.use_flash_attn = False
            self.vision.embeddings._get_pos_embed = lambda pos, h, w: pos
            self.scale = vlm.downsample_ratio

        def forward(self, pixels):
            x = self.vision(pixel_values=pixels, output_hidden_states=False, return_dict=True).last_hidden_state[:, 1:, :]
            side = int(x.shape[1] ** 0.5)
            x = vlm.pixel_shuffle(x.reshape(x.shape[0], side, side, -1), scale_factor=self.scale)
            return self.mlp1(x.reshape(x.shape[0], -1, x.shape[-1]))

    return Vit().cuda().eval()


def linears(model):
    """(name, module) in execution order."""
    out = []
    for i, layer in enumerate(model.vision.encoder.layers):
        out += [(f"L{i}.qkv", layer.attn.qkv), (f"L{i}.proj", layer.attn.proj),
                (f"L{i}.fc1", layer.mlp.fc1), (f"L{i}.fc2", layer.mlp.fc2)]
    out += [("mlp1.1", model.mlp1[1]), ("mlp1.3", model.mlp1[3])]
    return out


@torch.no_grad()
def smooth(model, batches, alpha):
    """SmoothQuant-style rebalancing, exact in fp32: divide each input channel of a
    linear by c_j and multiply the matching weight column by c_j, with
    c_j = max|x_j|^alpha / max|W_:j|^(1-alpha).  The division is folded into what
    produces the input: LayerNorm's affine (qkv, fc1) or the v rows of qkv (proj,
    whose input is attention-weighted v).  fc2's input is a GELU output and cannot
    be rescaled this way."""
    layers = model.vision.encoder.layers
    stat = {}
    def hook(key):
        def h(m, inp):
            a = inp[0].abs().reshape(-1, inp[0].shape[-1]).amax(0)
            stat[key] = torch.maximum(stat[key], a) if key in stat else a
        return h
    hs = []
    for i, layer in enumerate(layers):
        hs += [layer.attn.qkv.register_forward_pre_hook(hook((i, "qkv"))),
               layer.attn.proj.register_forward_pre_hook(hook((i, "proj"))),
               layer.mlp.fc1.register_forward_pre_hook(hook((i, "fc1")))]
    for b in batches: model(b.cuda())
    for h in hs: h.remove()
    def factor(x_max, weight):
        c = x_max.clamp_min(1e-5) ** alpha / weight.abs().amax(0).clamp_min(1e-5) ** (1 - alpha)
        return c.clamp(1e-3, 1e3)
    for i, layer in enumerate(layers):
        d = layer.attn.embed_dim
        for norm, lin, key in ((layer.norm1, layer.attn.qkv, "qkv"), (layer.norm2, layer.mlp.fc1, "fc1")):
            c = factor(stat[(i, key)], lin.weight)
            norm.weight.div_(c); norm.bias.div_(c); lin.weight.mul_(c[None, :])
        c = factor(stat[(i, "proj")], layer.attn.proj.weight)
        layer.attn.qkv.weight[2 * d:].div_(c[:, None]); layer.attn.qkv.bias[2 * d:].div_(c)
        layer.attn.proj.weight.mul_(c[None, :])


def wq(weight):
    """Per-output-channel symmetric int8 weight, dequantised."""
    s = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return (weight / s).round().clamp_(-127, 127) * s, s.squeeze(1)


def fq(x, s):
    return (x / s).round().clamp_(-127, 127) * s


@torch.no_grad()
def calibrate(model, batches):
    mods = linears(model)
    cand = {n: torch.zeros(len(QUANTILES), device="cuda") for n, _ in mods}
    def h1(name):
        def hook(m, inp):
            a = inp[0].abs().flatten()
            a = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)]
            q = torch.quantile(a, torch.tensor(QUANTILES[:-1], device=a.device))
            cand[name] += torch.cat((q, inp[0].abs().max()[None]))
        return hook
    hs = [m.register_forward_pre_hook(h1(n)) for n, m in mods]
    for b in batches: model(b.cuda())
    for h in hs: h.remove()
    for n in cand: cand[n] /= len(batches)
    err = {n: torch.zeros(len(QUANTILES), device="cuda") for n, _ in mods}
    wqs = {n: wq(m.weight)[0] for n, m in mods}
    def h2(name):
        def hook(m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            x = x[torch.randint(0, x.shape[0], (1024,), device=x.device)]
            y = x @ m.weight.t()
            for i, c in enumerate(cand[name]):
                err[name][i] += ((fq(x, c / 127) @ wqs[name].t() - y) ** 2).sum() / (y ** 2).sum()
        return hook
    hs = [m.register_forward_pre_hook(h2(n)) for n, m in mods]
    for b in batches: model(b.cuda())
    for h in hs: h.remove()
    scales = {}
    for n, _ in mods:
        i = int(err[n].argmin())
        scales[n] = {"act_scale": float(cand[n][i] / 127), "clip": float(cand[n][i]), "quantile": QUANTILES[i],
                     "max": float(cand[n][-1]), "layer_rel_err": float((err[n][i] / len(batches)).sqrt())}
    return scales


class Sim:
    """The fp32 encoder with chosen linears fake-quantised (what the engine computes)."""
    def __init__(self, model, scales):
        self.model, self.scales = model, scales
        self.mods = dict(linears(model))
        self.orig = {n: m.weight.data.clone() for n, m in self.mods.items()}
        self.hooks = []

    def set(self, names):
        for h in self.hooks: h.remove()
        self.hooks = []
        for n, m in self.mods.items():
            m.weight.data.copy_(self.orig[n])
            if n in names:
                m.weight.data.copy_(wq(self.orig[n])[0])
                s = self.scales[n]["act_scale"]
                self.hooks.append(m.register_forward_pre_hook(lambda mod, inp, s=s: (fq(inp[0], s),)))

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["calib", "onnx"])
    p.add_argument("--alpha", type=float, default=-1.0, help="SmoothQuant strength; < 0: no smoothing")
    p.add_argument("--export", action="store_true", help="calib: also export the (smoothed) fp32 ONNX")
    p.add_argument("--onnx", default=ONNX, help="onnx: the fp32 ONNX the scales belong to")
    p.add_argument("--scales", default=SCALES)
    p.add_argument("--calib-aug", type=int, default=11)
    p.add_argument("--skip", default="", help="comma-separated name fragments to leave in fp16, e.g. 'L0.,L23.,.fc2'")
    p.add_argument("--tag", default="int8qdq")
    a = p.parse_args()

    if a.cmd == "calib":
        agent = C.build_agent()
        vlm = agent.backbone.model
        model = build_fp32(agent)
        batches = pixel_batches(agent, n_aug=a.calib_aug)
        print(f"calibrating on {len(batches)} batches ({len(batches) * 9} tiles) from {len(glob.glob('frames/*.bmp'))} frames")
        real = pixel_batches(agent)
        with torch.no_grad():
            truth_r = [model(b.cuda()) for b in real]
        if a.alpha >= 0:
            smooth(model, batches, a.alpha)
            with torch.no_grad():
                print(f"smoothing alpha={a.alpha}: fp32 output changed by rel-L2 {np.mean([rel(model(b.cuda()), t) for b, t in zip(real, truth_r)]):.2e} (exact reparameterisation)")
        scales = calibrate(model, batches)
        torch.save(scales, a.scales)
        if a.export:
            path = a.scales.replace(".pt", ".onnx")
            torch.onnx.export(model, real[0].cuda(), path, opset_version=17, dynamo=False, input_names=["pixels"],
                              output_names=["features"], do_constant_folding=True)
            print("wrote", path)
        picks = {}
        for v in scales.values(): picks[v["quantile"]] = picks.get(v["quantile"], 0) + 1
        print("clip quantile chosen per layer:", picks)
        worst = sorted(scales.items(), key=lambda kv: -kv[1]["layer_rel_err"])[:6]
        print("largest single-layer output errors:", ", ".join(f"{n} {v['layer_rel_err']:.3f} (clip {v['clip']:.1f} of max {v['max']:.1f})" for n, v in worst))

        held = pixel_batches(agent, n_aug=3, seed=999)
        held = [b for i, b in enumerate(held) if i % 4][::2]
        with torch.no_grad():
            truth_h = [model(b.cuda()) for b in held]
            bf_r = [vlm.extract_feature(b.cuda().to(torch.bfloat16)).float() for b in real]
        print(f"\nyardstick: the reference (bf16) vs exact fp32 on the real frames: rel-L2 {np.mean([rel(x, t) for x, t in zip(bf_r, truth_r)]):.4f}")
        sim = Sim(model, scales)
        names = list(scales)
        hard = ("L0.", "L1.", "L12.")
        groups = {
            "all 98 linears": names,
            "all but fc2": [n for n in names if not n.endswith(".fc2")],
            "all but fc2, blocks 0,1,12": [n for n in names if not n.endswith(".fc2") and not n.startswith(hard)],
            "all but blocks 0,1,12": [n for n in names if not n.startswith(hard)],
            "all but fc2, blocks 0,1,12, mlp1": [n for n in names if not n.endswith(".fc2") and not n.startswith(hard + ("mlp1",))],
            "only qkv": [n for n in names if n.endswith(".qkv")],
            "only proj": [n for n in names if n.endswith(".proj")],
            "only fc1": [n for n in names if n.endswith(".fc1")],
            "only fc2": [n for n in names if n.endswith(".fc2")],
            "only mlp1": [n for n in names if n.startswith("mlp1")],
        }
        print(f"{'quantised':36s} n   rel-L2 vs fp32: real frames  held-out   | vs bf16 reference (real)")
        for g, ns in groups.items():
            sim.set(set(ns))
            er = np.mean([rel(sim(b.cuda()), t) for b, t in zip(real, truth_r)])
            eh = np.mean([rel(sim(b.cuda()), t) for b, t in zip(held, truth_h)])
            eb = np.mean([rel(sim(b.cuda()), t) for b, t in zip(real, bf_r)])
            print(f"{g:36s} {len(ns):3d}        {er:.4f}        {eh:.4f}     |   {eb:.4f}")
        sens = []
        for i in range(24):
            sim.set({n for n in names if n.startswith(f"L{i}.")})
            sens.append(np.mean([rel(sim(b.cuda()), t) for b, t in zip(real, truth_r)]))
        print("per-block error when only that block is quantised:", " ".join(f"{i}:{e:.3f}" for i, e in enumerate(sens)))
    else:
        import onnx
        from onnx import helper, numpy_helper
        scales = torch.load(a.scales)
        skip = [s for s in a.skip.split(",") if s]
        m = onnx.load(a.onnx)
        g = m.graph
        init = {i.name: i for i in g.initializer}
        names = list(scales)
        new_nodes, k, done = [], 0, 0
        for node in g.node:
            if node.op_type == "MatMul" and node.input[1] in init and len(init[node.input[1]].dims) == 2:
                name = names[k]; k += 1
                w = numpy_helper.to_array(init[node.input[1]])
                if not any(s in name for s in skip):
                    a_in, w_in = node.input[0], node.input[1]
                    s_act = np.array(scales[name]["act_scale"], np.float32)
                    s_w = (np.abs(w).max(axis=0).clip(1e-8) / 127).astype(np.float32)
                    for nm, arr in ((f"{name}_as", s_act), (f"{name}_az", np.array(0, np.int8)),
                                    (f"{name}_ws", s_w), (f"{name}_wz", np.zeros(w.shape[1], np.int8))):
                        g.initializer.append(numpy_helper.from_array(arr, nm))
                    new_nodes += [
                        helper.make_node("QuantizeLinear", [a_in, f"{name}_as", f"{name}_az"], [f"{name}_aq"]),
                        helper.make_node("DequantizeLinear", [f"{name}_aq", f"{name}_as", f"{name}_az"], [f"{name}_adq"]),
                        helper.make_node("QuantizeLinear", [w_in, f"{name}_ws", f"{name}_wz"], [f"{name}_wq"], axis=1),
                        helper.make_node("DequantizeLinear", [f"{name}_wq", f"{name}_ws", f"{name}_wz"], [f"{name}_wdq"], axis=1),
                    ]
                    node.input[0], node.input[1] = f"{name}_adq", f"{name}_wdq"
                    done += 1
            new_nodes.append(node)
        assert k == len(names), (k, len(names))
        del g.node[:]
        g.node.extend(new_nodes)
        path = os.path.join(OUT, f"vit_9x448_{a.tag}.onnx")
        onnx.save(m, path)
        print(f"wrote {path}: Q/DQ on {done} of {k} linears" + (f" (fp16: {skip})" if skip else ""))
