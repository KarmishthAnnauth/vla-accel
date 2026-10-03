#!/usr/bin/env python3
"""TensorRT experiments for ReCogDrive's vision encoder (InternViT-300M + mlp1).

  trt_vit.py export                 fp32 ONNX of InternVLChatModel.extract_feature, 9 tiles
  trt_vit.py build fp16|int8 [...]  engine (int8: entropy calibration on calib batches)
  trt_vit.py bench <engine> ...     latency, and accuracy against the PyTorch reference

Everything lands in /media/USER/EXTSSD/ReCogDrive/trt (the SSD).
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import torch

import common as C

OUT = os.path.join(C.SSD, "trt")
ONNX = os.path.join(OUT, "vit_9x448.onnx")
N_TILES = 9
HERE = os.path.dirname(os.path.abspath(__file__))


def pixel_batches(agent, n_aug=0, seed=0, frame_glob="frames/*.bmp"):
    """(9, 3, 448, 448) fp32 batches exactly as the model sees them, from the
    frames on disk; n_aug extra variants per frame (flip, zoom, gain)."""
    from PIL import Image, ImageEnhance
    from recogdrive_ros.recogdrive_speedups import FastReCogDrive
    fast = FastReCogDrive.__new__(FastReCogDrive)
    from concurrent.futures import ThreadPoolExecutor
    from navsim.agents.recogdrive.utils.internvl_preprocess import IMAGENET_MEAN, IMAGENET_STD, find_closest_aspect_ratio
    fast._pool = ThreadPoolExecutor(6); fast._ratio_cache = {}; fast._find_ratio = find_closest_aspect_ratio; fast.n_tiles = N_TILES
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rng = np.random.default_rng(seed)
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, frame_glob))):
        base = Image.open(path).convert("RGB")
        variants = [base]
        for _ in range(n_aug):
            im = base
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            z = rng.uniform(1.0, 1.35)
            w, h = im.size; cw, ch = int(w / z), int(h / z)
            x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
            variants.append(im)
        for im in variants:
            tiles = fast._load_tiles(im)
            x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float().div(255)
            out.append((x - mean) / std)
    return out


def export(agent):
    vlm = agent.backbone.model

    class Vit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = copy.deepcopy(vlm.vision_model).float()
            self.mlp1 = copy.deepcopy(vlm.mlp1).float()
            for layer in self.vision.encoder.layers:
                layer.attn.use_flash_attn = False
            emb = self.vision.embeddings
            emb._get_pos_embed = lambda pos, h, w: pos
            self.scale = vlm.downsample_ratio

        def forward(self, pixels):
            x = self.vision(pixel_values=pixels, output_hidden_states=False, return_dict=True).last_hidden_state
            x = x[:, 1:, :]
            side = int(x.shape[1] ** 0.5)
            x = vlm.pixel_shuffle(x.reshape(x.shape[0], side, side, -1), scale_factor=self.scale)
            return self.mlp1(x.reshape(x.shape[0], -1, x.shape[-1]))

    model = Vit().cuda().eval()
    x = pixel_batches(agent)[0].cuda()
    with torch.no_grad():
        ref32 = model(x)
        ref_bf = vlm.extract_feature(x.to(torch.bfloat16)).float()
    print("fp32 export model vs the bf16 reference: rel-L2", ((ref_bf - ref32).norm() / ref32.norm()).item())
    os.makedirs(OUT, exist_ok=True)
    torch.onnx.export(model, x, ONNX, opset_version=17, dynamo=False, input_names=["pixels"],
                      output_names=["features"], do_constant_folding=True)
    print("wrote", ONNX, os.path.getsize(ONNX) >> 20, "MiB")


def build(agent, precision, calib_aug, level, tag, onnx_path=ONNX):
    """precision: fp16 | int8 (TensorRT's calibrator) | qdq (an ONNX that already
    carries Q/DQ nodes: INT8 where they are, fp16 elsewhere)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.builder_optimization_level = level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache_path = os.path.join(OUT, "timing.cache")
    cache = config.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "qdq":
        config.set_flag(trt.BuilderFlag.INT8)
    keep = None
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        batches = pixel_batches(agent, n_aug=calib_aug)
        print(f"calibrating on {len(batches)} batches of {N_TILES} tiles")

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.i = 0

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.i >= len(batches):
                    return None
                self.cur = batches[self.i].contiguous().cuda()
                self.i += 1
                return [int(self.cur.data_ptr())]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, data):
                open(os.path.join(OUT, f"calib_{tag}.cache"), "wb").write(data)

        keep = Calibrator()
        config.int8_calibrator = keep
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    path = os.path.join(OUT, f"vit_{tag}.engine")
    open(path, "wb").write(blob)
    open(cache_path, "wb").write(config.get_timing_cache().serialize())
    print(f"built {path} ({os.path.getsize(path) >> 20} MiB) in {time.time() - t0:.0f} s")
    return path


class Engine:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(open(path, "rb").read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inp = torch.zeros(tuple(self.engine.get_tensor_shape("pixels")), device="cuda")
        self.out = torch.zeros(tuple(self.engine.get_tensor_shape("features")), device="cuda")
        self.context.set_tensor_address("pixels", self.inp.data_ptr())
        self.context.set_tensor_address("features", self.out.data_ptr())

    def __call__(self, x):
        self.inp.copy_(x)
        self.run()
        return self.out

    def run(self):
        assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

    def precisions(self):
        import json
        insp = self.engine.create_engine_inspector()
        info = json.loads(insp.get_engine_information(self.trt.LayerInformationFormat.JSON))
        counts = {}
        for layer in info.get("Layers", []):
            if isinstance(layer, dict):
                outs = layer.get("Outputs", [])
                fmt = outs[0].get("Format/Datatype", "?") if outs else "?"
                key = (layer.get("LayerType", "?"), fmt.split()[-1] if fmt else "?")
                counts[key] = counts.get(key, 0) + 1
        return counts


def bench(agent, paths):
    vlm = agent.backbone.model
    evalb = pixel_batches(agent)
    held = pixel_batches(agent, n_aug=2, seed=123)[1::3] + pixel_batches(agent, n_aug=2, seed=123)[2::3]
    with torch.no_grad():
        refs = [vlm.extract_feature(b.cuda().to(torch.bfloat16)).float() for b in evalb + held]
    x = evalb[0].cuda().to(torch.bfloat16)
    with torch.no_grad():
        for _ in range(3): vlm.extract_feature(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): vlm.extract_feature(x)
        torch.cuda.synchronize()
    print(f"PyTorch bf16 eager extract_feature: {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms (fast path graph: ~280 ms)")
    for path in paths:
        eng = Engine(path)
        for _ in range(5): eng.run()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): eng.run()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 20 * 1e3
        errs, coss = [], []
        for b, r in zip(evalb + held, refs):
            o = eng(b.cuda()).float()
            errs.append(((o - r).norm() / r.norm()).item())
            coss.append(torch.nn.functional.cosine_similarity(o, r, dim=-1).min().item())
        n = len(evalb)
        print(f"{os.path.basename(path):28s} {ms:7.1f} ms | vs bf16 reference: rel-L2 real frames {np.mean(errs[:n]):.4f}, "
              f"held-out variants {np.mean(errs[n:]):.4f}; worst token cosine {min(coss):.4f}")
        prec = eng.precisions()
        print("     layers by (type, output dtype):", ", ".join(f"{k[0]}/{k[1]}x{v}" for k, v in sorted(prec.items(), key=lambda kv: -kv[1])[:8]))
        del eng


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["export", "build", "bench"])
    p.add_argument("args", nargs="*")
    p.add_argument("--calib-aug", type=int, default=7)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--tag", default="")
    p.add_argument("--onnx", default=ONNX)
    a = p.parse_args()
    agent = C.build_agent()
    if a.cmd == "export":
        export(agent)
    elif a.cmd == "build":
        build(agent, a.args[0], a.calib_aug, a.level, a.tag or a.args[0], a.onnx)
    else:
        bench(agent, a.args)
