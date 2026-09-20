"""Optimised SimLingo inference, measured end-to-end against the as-is baseline.

Optimisations, all inside PyTorch (no ONNX / TensorRT):
  1. merge the LoRA adapters into the base weights   (3 matmuls -> 1 per linear)
  2. swap FlashAttention2 -> SDPA                    (FA2 is not CUDA-graph capturable)
  3. give greedy_sample a KV cache                   (was re-prefilling the whole
                                                      543-token prompt for EVERY token)
  4. capture the decode step in a CUDA graph         (decode is ~93% launch overhead)
  5. run the driving head as a cache continuation    (30 tokens, not a 3rd full forward)

Correctness is checked against the unoptimised model on the same input.
"""
import sys, time, math, importlib.util
sys.path.insert(0,"/benchmarking/simlingo"); sys.path.insert(0,"/benchmarking/simlingo/team_code")
import numpy as np, torch, hydra
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig, StaticCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
from PIL import Image as PILImage

DEV = torch.device("cuda")
SIM = "/benchmarking/simlingo"
torch.backends.cuda.matmul.allow_tf32 = True

cfg = OmegaConf.load("/models/simlingo/simlingo/.hydra/config.yaml")
cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
variant = cfg.model.vision_model.variant
vcache  = f"{SIM}/pretrained/{variant.split('/')[1]}"

processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True, cache_dir=vcache)
tokenizer = processor.tokenizer if hasattr(processor,"tokenizer") else processor
tokenizer.add_special_tokens({"additional_special_tokens":[
    "<WAYPOINTS>","<WAYPOINTS_DIFF>","<ORG_WAYPOINTS_DIFF>","<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>","<ROUTE>","<ROUTE_DIFF>","<TARGET_POINT>"]})
tokenizer.padding_side = "left"
vcfg = AutoConfig.from_pretrained(variant, trust_remote_code=True, cache_dir=vcache)
NIT = int((( vcfg.force_image_size or vcfg.vision_config.image_size)//vcfg.vision_config.patch_size)**2
          * (vcfg.downsample_ratio**2))
spec = importlib.util.spec_from_file_location("_conv", f"{vcache}/conversation.py")
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)

from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
transform = build_transform(input_size=448)

dd = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
model = hydra.utils.instantiate(cfg.model, cfg_data_module=cfg.data_module, processor=processor,
                                cache_dir=vcache, _recursive_=False).to(DEV)
torch.set_default_dtype(dd)
model.load_state_dict(torch.load("/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
                                 map_location=DEV))
model.eval()

rng = np.random.default_rng(0)
img = (np.linspace(0,255,600*1024*3).reshape(600,1024,3) % 255).astype(np.uint8)
img[200:400, 300:700] = rng.integers(0,255,(200,400,3))
speed = 5.0
tps = np.array([[8.0,0.5],[16.0,1.0]], dtype=np.float32)

pil   = PILImage.fromarray(img)
tiles = dynamic_preprocess(pil, image_size=448, use_thumbnail=cfg.model.vision_model.use_global_img, max_num=2)
pv    = torch.stack([transform(t) for t in tiles])
processed = pv.unsqueeze(0).unsqueeze(0)
npatch = pv.shape[0]

def make_input():
    prompt = (f"Current speed: {round(speed,1)} m/s. "
              "Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
    tpid = tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
    tpl = conv.get_conv_template("internlm2-chat")
    tpl.append_message(tpl.roles[0], f"<image>\n{prompt}"); tpl.append_message(tpl.roles[1], None)
    q = tpl.get_prompt()
    q = q.replace(tpl.system_template.replace("{system_message}", tpl.system_message)+tpl.sep, "")
    q = q.replace("<image>", "<img>"+"<IMG_CONTEXT>"*NIT*npatch+"</img>", 1)
    tok = tokenizer([q], padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    ll = LanguageLabel(phrase_ids=tok["input_ids"].to(DEV),
                       phrase_valid=(tok["input_ids"]!=tokenizer.pad_token_id).to(DEV),
                       phrase_mask=(tok["input_ids"]!=tokenizer.pad_token_id).to(DEV),
                       placeholder_values=[{tpid: tps}], language_string=[q], loss_masking=None)
    _,_,_,C,H,W = processed.shape
    focal = W/(2.0*math.tan(110*math.pi/360.0))
    K = torch.tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]],dtype=torch.float32).unsqueeze(0).to(DEV)
    E = torch.eye(4,dtype=torch.float32); E[:3,3]=torch.tensor([-1.5,0.,2.]); E=E.unsqueeze(0).to(DEV)
    return DrivingInput(camera_images=processed.to(DEV).bfloat16(), image_sizes=None,
                        camera_intrinsics=K, camera_extrinsics=E,
                        vehicle_speed=torch.tensor([[speed]],dtype=torch.float32,device=DEV),
                        target_point=torch.from_numpy(tps[:1]).float().to(DEV), prompt=ll, prompt_inference=ll)

mi = make_input()

spin = torch.randn(2048,2048,device=DEV,dtype=torch.bfloat16)
def warm(sec=15):
    t=time.time()
    while time.time()-t<sec:
        for _ in range(20): torch.mm(spin,spin)
    torch.cuda.synchronize()
def bench(fn,n,warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

print("warming GPU clock…"); warm(15)
print("gpu clock:", int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read())//10**6, "MHz")

print("\n" + "="*66 + "\nBASELINE (as-is)\n" + "="*66)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    base_t = bench(lambda: model(mi), n=4)
    wps0, route0, lang0 = model(mi)
print(f"  end-to-end: {base_t:8.1f} ms")
print(f"  language  : {lang0[0]!r}")

print("\n" + "="*66 + "\nOPTIMISED\n" + "="*66)
lm_wrap = model.language_model
merged  = lm_wrap.model.merge_and_unload().eval()
merged.config._attn_implementation = "sdpa"
for l in merged.model.layers:
    l.self_attn.__class__ = Qwen2SdpaAttention
    l.self_attn.config._attn_implementation = "sdpa"
lm_wrap.model = merged
model.adaptors.language.lm_head = merged.lm_head
print("  LoRA merged, attention = SDPA")

HID    = lm_wrap.hidden_size
MAXLEN = 768
EOSID  = tokenizer.eos_token_id
embed_w = model.adaptors.language.embed_tokens.weight
logit_w = model.adaptors.language.lm_head.weight

s_emb  = torch.zeros(1,1,HID, device=DEV, dtype=torch.bfloat16)
s_pos  = torch.zeros(1, dtype=torch.long, device=DEV)
s_mask = torch.zeros(1, MAXLEN, device=DEV, dtype=torch.bool)
kv     = StaticCache(config=merged.config, max_batch_size=1, max_cache_len=MAXLEN,
                     device=DEV, dtype=torch.bfloat16)
_graph = {"g": None, "out": None}

def decode_eager():
    return merged(inputs_embeds=s_emb, attention_mask=s_mask, past_key_values=kv,
                  cache_position=s_pos, use_cache=True, output_hidden_states=True, return_dict=True)

def fast_forward(mi):
    """Replacement for DrivingModel.forward: cached generation + cached driving head."""
    adaptor_dict = model.adaptors(mi, inference=True)
    adaptor_dict = model.vision_model.image_encoder.replace_placeholder_tokens(
        adaptor_dict=adaptor_dict, pixel_values=mi.camera_images,
        placeholder_values=mi.prompt_inference.placeholder_values, wp_encoder=model.wp_encoder)
    inp  = adaptor_dict["language_inputs"][0].unsqueeze(0)
    L    = inp.size(1)

    kv.reset()
    s_mask.zero_(); s_mask[:, :L] = True
    out = merged(inputs_embeds=inp.to(torch.bfloat16),
                 attention_mask=s_mask[:, :L].contiguous(),
                 past_key_values=kv, cache_position=torch.arange(L, device=DEV),
                 use_cache=True, output_hidden_states=True, return_dict=True)

    toks, pos = [], L
    h = out.hidden_states[-1][:, -1]
    for _ in range(100):
        nt = F.linear(h, logit_w).argmax(-1)
        toks.append(nt)
        if int(nt) == EOSID: break
        s_emb.copy_(F.embedding(nt.unsqueeze(1), embed_w).to(torch.bfloat16))
        s_pos.fill_(pos); s_mask[0, pos] = True
        if _graph["g"] is None:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): decode_eager()
            torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
            s_pos.fill_(pos)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                o = decode_eager()
            _graph["g"], _graph["out"] = g, o
            torch.cuda.synchronize()
        _graph["g"].replay()
        h = _graph["out"].hidden_states[-1][:, -1]
        pos += 1

    drv  = model.adaptors.driving(mi)["inputs"]
    nd   = drv.size(1)
    last = F.embedding(toks[-1].unsqueeze(1), embed_w).to(torch.bfloat16)
    tail = torch.cat([last, drv.to(torch.bfloat16)], dim=1)
    s_mask[0, pos:pos+tail.size(1)] = True
    o = merged(inputs_embeds=tail, attention_mask=s_mask[:, :pos+tail.size(1)].contiguous(),
               past_key_values=kv, cache_position=torch.arange(pos, pos+tail.size(1), device=DEV),
               use_cache=True, output_hidden_states=True, return_dict=True)
    feats = o.hidden_states[-1][:, -nd:]
    preds = model.adaptors.driving.get_predictions(feats, o[0][:, -nd:])
    text  = tokenizer.batch_decode(torch.cat(toks).unsqueeze(0), skip_special_tokens=True)[0]
    return preds["speed_wps"], preds["route"], [text], len(toks)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    wps1, route1, lang1, ntok = fast_forward(mi)
    opt_t = bench(lambda: fast_forward(mi), n=6)

print(f"  end-to-end: {opt_t:8.1f} ms   ({ntok} tokens generated)")
print(f"  language  : {lang1[0]!r}")
print(f"\n  route  max abs diff vs baseline: {(route1.float()-route0.float()).abs().max():.5f} m")
print(f"  speed  max abs diff vs baseline: {(wps1.float()-wps0.float()).abs().max():.5f} m")
print(f"  language identical: {lang1[0].strip() == lang0[0].strip()}")
print(f"\n  SPEEDUP: {base_t/opt_t:.1f}x   ({base_t:.0f} ms -> {opt_t:.0f} ms)")
