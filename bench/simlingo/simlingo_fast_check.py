"""Correctness + speed check for simlingo_training.models.fast_inference.

Loads the real checkpoint, builds an input matching what the ROS node feeds the
model (1024x359 frame -> 2 tiles -> 543-token prompt), and compares the fast
path against the untouched original.  Run inside the sim_ros container.
"""
import importlib.util
import math
import sys
import time

sys.path.insert(0, "/benchmarking/simlingo")
sys.path.insert(0, "/benchmarking/simlingo/team_code")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image as PILImage
from transformers import AutoConfig, AutoProcessor

SIM = "/benchmarking/simlingo"
CKPT = "/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
DEV = torch.device("cuda")
FREQ = "/sys/class/devfreq/17000000.gpu/cur_freq"


def gpu_mhz():
    try:
        return int(open(FREQ).read().strip()) // 10 ** 6
    except Exception:
        return -1


def build_model():
    cfg = OmegaConf.load("/models/simlingo/simlingo/.hydra/config.yaml")
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    variant = cfg.model.vision_model.variant
    cache = f"{SIM}/pretrained/{variant.split('/')[1]}"
    proc = AutoProcessor.from_pretrained(variant, trust_remote_code=True, cache_dir=cache)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    tok.add_special_tokens({"additional_special_tokens": [
        "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
        "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"]})
    tok.padding_side = "left"

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(
        cfg.model, cfg_data_module=cfg.data_module, processor=proc,
        cache_dir=cache, _recursive_=False).to(DEV)
    torch.set_default_dtype(default)
    model.load_state_dict(torch.load(CKPT, map_location=DEV))
    model.eval()
    return model, cfg, tok, cache, variant


def build_example(cfg, tok, cache_dir, variant, speed=5.0):
    """Same shape the ROS node produces: 1024x359 frame, 2 tiles, 543 tokens."""
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

    vcfg = AutoConfig.from_pretrained(variant, trust_remote_code=True, cache_dir=cache_dir)
    n_img_tok = int(((vcfg.force_image_size or vcfg.vision_config.image_size)
                     // vcfg.vision_config.patch_size) ** 2 * (vcfg.downsample_ratio ** 2))
    spec = importlib.util.spec_from_file_location("_conv", f"{cache_dir}/conversation.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    rng = np.random.default_rng(0)
    img = (np.linspace(0, 255, 359 * 1024 * 3).reshape(359, 1024, 3) % 255).astype(np.uint8)
    img[120:260, 300:700] = rng.integers(0, 255, (140, 400, 3))

    transform = build_transform(input_size=448)
    tiles = dynamic_preprocess(PILImage.fromarray(img), image_size=448,
                               use_thumbnail=cfg.model.vision_model.use_global_img, max_num=2)
    pixel = torch.stack([transform(t) for t in tiles])
    processed = pixel.unsqueeze(0).unsqueeze(0)

    tps = np.array([[8.0, 0.5], [16.0, 1.0]], dtype=np.float32)
    prompt = (f"Current speed: {round(speed, 1)} m/s. "
              "Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
    tpl = conv.get_conv_template("internlm2-chat")
    tpl.append_message(tpl.roles[0], f"<image>\n{prompt}")
    tpl.append_message(tpl.roles[1], None)
    query = tpl.get_prompt()
    query = query.replace(
        tpl.system_template.replace("{system_message}", tpl.system_message) + tpl.sep, "")
    query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * n_img_tok * pixel.shape[0]
                          + "</img>", 1)
    enc = tok([query], padding=True, return_tensors="pt",
              return_offsets_mapping=True, add_special_tokens=False)
    label = LanguageLabel(
        phrase_ids=enc["input_ids"].to(DEV),
        phrase_valid=(enc["input_ids"] != tok.pad_token_id).to(DEV),
        phrase_mask=(enc["input_ids"] != tok.pad_token_id).to(DEV),
        placeholder_values=[{tok.convert_tokens_to_ids("<TARGET_POINT>"): tps}],
        language_string=[query], loss_masking=None)

    _, _, _, _, H, W = processed.shape
    focal = W / (2.0 * math.tan(110 * math.pi / 360.0))
    K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]],
                     dtype=torch.float32).unsqueeze(0).to(DEV)
    E = torch.eye(4, dtype=torch.float32)
    E[:3, 3] = torch.tensor([-1.5, 0.0, 2.0])
    E = E.unsqueeze(0).to(DEV)

    return DrivingInput(
        camera_images=processed.to(DEV).bfloat16(), image_sizes=None,
        camera_intrinsics=K, camera_extrinsics=E,
        vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=DEV),
        target_point=torch.from_numpy(tps[:1]).float().to(DEV),
        prompt=label, prompt_inference=label), enc["input_ids"].shape[1]


def bench(fn, n=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    print("loading model…", flush=True)
    model, cfg, tok, cache_dir, variant = build_model()
    example, n_tok = build_example(cfg, tok, cache_dir, variant)
    print(f"prompt tokens: {n_tok}   gpu: {gpu_mhz()} MHz\n")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_ms = bench(lambda: model(example))
        b_speed, b_route, b_lang = model(example)
    print(f"BASELINE (untouched)   {base_ms:8.1f} ms   language={b_lang[0]!r}")

    from simlingo_training.models.fast_inference import optimize_for_inference
    report = optimize_for_inference(model)
    print(f"\noptimize_for_inference -> {report}")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fast_ms = bench(lambda: model(example))
        f_speed, f_route, f_lang = model(example)
    fast = model._fast_inference
    print(f"\nOPTIMISED              {fast_ms:8.1f} ms   language={f_lang[0]!r}")
    print(f"  cuda graph active: {fast._graph is not None}")

    route_d = float((f_route.float() - b_route.float()).abs().max())
    speed_d = float((f_speed.float() - b_speed.float()).abs().max())
    print(f"\nvs BASELINE (includes the bf16 LoRA-merge perturbation):")
    print(f"  route max |diff| : {route_d:.4f} m")
    print(f"  speed max |diff| : {speed_d:.4f} m")
    print(f"  language match   : {f_lang[0].strip() == b_lang[0].strip()}")
    print(f"\n  SPEEDUP: {base_ms / fast_ms:.2f}x   ({base_ms:.0f} -> {fast_ms:.0f} ms)")

    print("\nrepeat-call stability (same input, 5 calls):")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        firsts = []
        for i in range(5):
            s, r, l = model(example)
            firsts.append(float(r[0, 0, 0]))
            print(f"  call {i}: route[0,0]={r[0,0,0].float():+.5f}  lang={l[0]!r}")
    print(f"  all identical: {len(set(f'{x:.6f}' for x in firsts)) == 1}")

    print("\nvarying prompt length (speed changes token count):")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for spd in (0.0, 5.0, 12.5, 0.0):
            ex, nt = build_example(cfg, tok, cache_dir, variant, speed=spd)
            s, r, l = model(ex)
            print(f"  speed={spd:5.1f}  tokens={nt}  route[0,0]={r[0,0,0].float():+.5f}  lang={l[0]!r}")


if __name__ == "__main__":
    main()
