"""Export ORION's EVA-ViT-L backbone (6x3x640x640) to ONNX for a TensorRT engine.

Flash attention is off and the position embedding baked; a *_parity.pt file is written for check_vit_engine.py.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from vla_accel.orion import model as om

IMG_SHAPE = (6, 3, 640, 640)


def load_backbone_state(ckpt: str, cache: str) -> dict:
    """img_backbone.* tensors from Orion.pth, cached to `cache` after the first run."""
    if cache and os.path.exists(cache):
        return torch.load(cache, map_location="cpu")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    bb = {k[len("img_backbone."):]: v.clone() for k, v in sd.items() if k.startswith("img_backbone.")}
    if cache:
        torch.save(bb, cache)
    return bb


class ExportVit(nn.Module):
    """EVAViT.forward with the position embedding pre-resolved."""

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
    """Build the backbone, check parity, export and validate the ONNX graph."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=om.CONFIG)
    ap.add_argument("--checkpoint", default=om.CHECKPOINT)
    ap.add_argument("--state-cache", default=os.path.join(om.ENGINES_DIR, "orion_vit_sd.pt"))
    ap.add_argument("--out", default=os.path.join(om.ENGINES_DIR, "orion_vit.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    args.out, args.state_cache = os.path.abspath(args.out), os.path.abspath(args.state_cache)
    os.makedirs(os.path.dirname(args.state_cache), exist_ok=True)
    om.use_repo()

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
        """Max abs and relative L2 difference."""
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
