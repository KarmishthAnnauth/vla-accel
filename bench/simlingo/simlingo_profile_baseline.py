"""Profile SimLingo inference on the Orin: where does the per-frame time go?

Replicates simlingo_node._load_model + _run_inference exactly, but with a
synthetic image, and instruments the three phases:
   vision encode | autoregressive language decode | driving-head forward
Run inside the sim_ros container.
"""
import importlib.util, math, os, sys, time
from pathlib import Path

SIMLINGO = "/benchmarking/simlingo"
sys.path.insert(0, SIMLINGO)
sys.path.insert(0, SIMLINGO + "/team_code")

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
from PIL import Image as PILImage

CKPT = "/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CFG  = "/models/simlingo/simlingo/.hydra/config.yaml"
DEV  = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

cfg = OmegaConf.load(CFG)
cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
variant = cfg.model.vision_model.variant
vlm_cache = f"{SIMLINGO}/pretrained/{variant.split('/')[1]}"

processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True, cache_dir=vlm_cache)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
tokenizer.add_special_tokens({"additional_special_tokens": [
    "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"]})
tokenizer.padding_side = "left"

vlm_cfg = AutoConfig.from_pretrained(variant, trust_remote_code=True, cache_dir=vlm_cache)
image_size = vlm_cfg.force_image_size or vlm_cfg.vision_config.image_size
num_image_token = int((image_size // vlm_cfg.vision_config.patch_size) ** 2
                      * (vlm_cfg.downsample_ratio ** 2))

spec = importlib.util.spec_from_file_location("_conv", f"{vlm_cache}/conversation.py")
conv_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv_module)

from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel

transform = build_transform(input_size=448)

print("loading model…", flush=True)
t0 = time.time()
dd = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
model = hydra.utils.instantiate(cfg.model, cfg_data_module=cfg.data_module,
                                processor=processor, cache_dir=vlm_cache,
                                _recursive_=False).to(DEV)
torch.set_default_dtype(dd)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

def nparams(m): return sum(p.numel() for p in m.parameters())
print(f"  vision_model : {nparams(model.vision_model)/1e6:7.1f} M")
print(f"  language_model:{nparams(model.language_model)/1e6:7.1f} M")
print(f"  total        : {nparams(model)/1e6:7.1f} M")
print(f"  predict_language = {model.predict_language}")

rng = np.random.default_rng(0)
img = rng.integers(0, 255, (600, 1024, 3), dtype=np.uint8)
speed = 5.0
target_points_np = np.array([[8.0, 0.5], [16.0, 1.0]], dtype=np.float32)
target_point_torch = torch.from_numpy(target_points_np[:1]).float()

def preprocess():
    pil = PILImage.fromarray(img)
    tiles = dynamic_preprocess(pil, image_size=448,
                               use_thumbnail=cfg.model.vision_model.use_global_img, max_num=2)
    pv = torch.stack([transform(t) for t in tiles])
    return pv.unsqueeze(0).unsqueeze(0), pv.shape[0]

processed_image, num_patches = preprocess()
print(f"  tiles per frame: {num_patches}, image tokens/tile: {num_image_token}")

def build_input():
    prompt = (f"Current speed: {round(speed,1)} m/s. "
              "Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
    tp_id = tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
    question = f"<image>\n{prompt}"
    tpl = conv_module.get_conv_template("internlm2-chat")
    tpl.append_message(tpl.roles[0], question); tpl.append_message(tpl.roles[1], None)
    query = tpl.get_prompt()
    hdr = tpl.system_template.replace("{system_message}", tpl.system_message) + tpl.sep
    query = query.replace(hdr, "")
    query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * num_image_token * num_patches + "</img>", 1)
    tok = tokenizer([query], padding=True, return_tensors="pt",
                    return_offsets_mapping=True, add_special_tokens=False)
    ll = LanguageLabel(phrase_ids=tok["input_ids"].to(DEV),
                       phrase_valid=(tok["input_ids"] != tokenizer.pad_token_id).to(DEV),
                       phrase_mask=(tok["input_ids"] != tokenizer.pad_token_id).to(DEV),
                       placeholder_values=[{tp_id: target_points_np}],
                       language_string=[query], loss_masking=None)
    _,_,n_tiles,C,H,W = processed_image.shape
    focal = W / (2.0 * math.tan(110 * math.pi / 360.0))
    K = torch.tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]], dtype=torch.float32).unsqueeze(0).to(DEV)
    E = torch.eye(4, dtype=torch.float32); E[:3,3] = torch.tensor([-1.5,0.,2.]); E = E.unsqueeze(0).to(DEV)
    return DrivingInput(camera_images=processed_image.to(DEV).bfloat16(), image_sizes=None,
                        camera_intrinsics=K, camera_extrinsics=E,
                        vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=DEV),
                        target_point=target_point_torch.to(DEV), prompt=ll, prompt_inference=ll), tok["input_ids"].shape[1]

model_input, n_prompt_tokens = build_input()
print(f"  prompt token length: {n_prompt_tokens}")

import simlingo_training.models.language_model.llm as llmmod
_orig_fwd = llmmod.LLM.forward
STATS = {"calls": 0, "t": 0.0, "seqlens": []}
def timed_fwd(self, embeddings, *a, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = _orig_fwd(self, embeddings, *a, **kw)
    torch.cuda.synchronize()
    STATS["calls"] += 1; STATS["t"] += time.perf_counter() - t
    STATS["seqlens"].append(embeddings.shape[1])
    return out
llmmod.LLM.forward = timed_fwd

ie = model.vision_model.image_encoder
_orig_rp = ie.replace_placeholder_tokens
VSTATS = {"calls": 0, "t": 0.0}
def timed_rp(*a, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = _orig_rp(*a, **kw)
    torch.cuda.synchronize()
    VSTATS["calls"] += 1; VSTATS["t"] += time.perf_counter() - t
    return out
ie.replace_placeholder_tokens = timed_rp

def run(tag, n=3):
    for i in range(n):
        for d in (STATS, VSTATS):
            d["calls"] = 0; d["t"] = 0.0
        STATS["seqlens"] = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wps, route, lang = model(model_input)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        sl = STATS["seqlens"]
        print(f"[{tag}] iter {i}: total {total*1000:8.1f} ms | "
              f"vision {VSTATS['t']*1000:7.1f} ms ({VSTATS['calls']}x) | "
              f"llm {STATS['t']*1000:8.1f} ms in {STATS['calls']} fwd "
              f"(seqlen {min(sl) if sl else 0}->{max(sl) if sl else 0}) | "
              f"other {(total-STATS['t']-VSTATS['t'])*1000:6.1f} ms", flush=True)
        if i == n-1 and lang:
            print(f"        language: {lang[0][:160]!r}")

t = time.perf_counter()
for _ in range(5): preprocess()
print(f"\n[cpu] image preprocess (tile+normalise): {(time.perf_counter()-t)/5*1000:.1f} ms")

print()
run("as-is", 4)

print("\n--- predict_language = False (driving head only, no CoT) ---")
model.predict_language = False
try:
    run("no-lang", 4)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
