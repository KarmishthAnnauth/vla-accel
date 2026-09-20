#!/usr/bin/env python3
"""Export ORION's EVA-ViT-L image backbone to ONNX for a TensorRT engine.

Run inside the orion_ros container (needs the compiled mmcv ops):

  docker exec orion_ros bash -c 'cd /root/Orion && python3 \
      /benchmarking/alpamayo-autoware/src/orion_ros/tools/export_vit_onnx.py \
      --out /benchmarking/alpamayo-autoware/src/orion_ros/engines/orion_vit.onnx'
  # then, still in the container:
  /usr/src/tensorrt/bin/trtexec --onnx=.../orion_vit.onnx --saveEngine=.../orion_vit_fp16.plan \
      --fp16 --builderOptimizationLevel=5 --memPoolSize=workspace:8192M

What is exported: exactly the module ORION calls in `extract_img_feat`
(`img_backbone(img)` -> [feat]) at the fixed inference shape 6x3x640x640,
with two export-only substitutions that the parity check below covers:
  * `flash_attn=False`: attention goes through the backbone's own
    matmul/softmax branch instead of the flash_attn custom op (TensorRT fuses
    the standard pattern itself);
  * the absolute position embedding is baked at the 40x40 token grid so no
    interpolate runs per forward.
Weights are exported in fp32 (the same values ORION halves for fp16_infer);
trtexec --fp16 picks the per-layer precision.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

IMG_SHAPE = (6, 3, 640, 640)


def load_backbone_state(ckpt: str, cache: str) -> dict:
    """img_backbone.* tensors from Orion.pth, cached to `cache` after the first
    run. A plain sequential load (~200 s, 38 GB RAM) is used on purpose:
    torch.load(mmap=True) page-faults 4 KB at a time through the exfat-FUSE
    mount and was still reading after 10 minutes."""
    if cache and os.path.exists(cache):
        return torch.load(cache, map_location="cpu")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    bb = {k[len("img_backbone."):]: v.clone() for k, v in sd.items() if k.startswith("img_backbone.")}
    if cache:
        torch.save(bb, cache)
    return bb


class ExportVit(nn.Module):
    """EVAViT.forward with the position embedding pre-resolved (see module doc)."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        from mmcv.models.backbones.eva_vit import get_abs_pos
        self.vit = vit
        with torch.no_grad():
            hw = (IMG_SHAPE[2] // vit.patch_embed.proj.kernel_size[0],) * 2
            pos = get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, hw)
        self.register_buffer("pos", pos.clone())

    def forward(self, x):
        v = self.vit
        x = v.patch_embed(x) + self.pos
        for blk in v.blocks:
            x = blk(x)
        return x.permute(0, 3, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/Orion/adzoo/orion/configs/orion_stage3_agent.py")
    ap.add_argument("--checkpoint", default="/models/Orion/Orion.pth")
    ap.add_argument("--state-cache", default="/benchmarking/.orion_vit_sd.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from mmcv import Config
    from mmcv.models import build_backbone

    cfg = Config.fromfile(args.config)
    bb_cfg = dict(cfg.model.img_backbone)
    bb_cfg.update(flash_attn=False, with_cp=False, drop_path_rate=0.0)
    ref_cfg = dict(cfg.model.img_backbone)
    ref_cfg.update(with_cp=False, drop_path_rate=0.0)

    t0 = time.time()
    state = load_backbone_state(args.checkpoint, args.state_cache)
    print(f"backbone state: {len(state)} tensors in {time.time() - t0:.1f}s")

    vit = build_backbone(bb_cfg)
    missing, unexpected = vit.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={missing} unexpected={unexpected}")
    assert not [m for m in missing if "rope" not in m and "freqs" not in m], "unexpected missing weights"
    vit.eval().cuda()
    export_model = ExportVit(vit).eval().cuda()

    ref = build_backbone(ref_cfg)
    ref.load_state_dict(state, strict=False)
    ref.eval().cuda().half()

    torch.manual_seed(0)
    x = torch.rand(*IMG_SHAPE, device="cuda") * 255.0
    x = (x - 127.5) / 58.0
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            y_ref = ref(x.half())[0].float()
        y_exp = export_model(x)
        y_pos = vit(x)[0]
    def rel(a, b):
        return float((a - b).abs().max()), float((a - b).norm() / b.norm())
    print(f"parity fp32/flash-off vs fp16/flash (ORION path): max|d|={rel(y_exp, y_ref)[0]:.4f} rel={rel(y_exp, y_ref)[1]:.2e}")
    print(f"parity baked-pos vs unbaked (both fp32):         max|d|={rel(y_exp, y_pos)[0]:.2e}")
    print(f"feature shape {tuple(y_exp.shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            export_model, x, args.out, opset_version=args.opset,
            input_names=["img"], output_names=["feat"],
            do_constant_folding=True, dynamic_axes=None,
        )
    print(f"exported {args.out} ({os.path.getsize(args.out) / 2**20:.0f} MiB) in {time.time() - t0:.0f}s")
    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("onnx ok; ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    torch.save({"x": x.cpu(), "y_ref": y_ref.cpu(), "y_exp": y_exp.cpu()},
               os.path.splitext(args.out)[0] + "_parity.pt")


if __name__ == "__main__":
    main()
