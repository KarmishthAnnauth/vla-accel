"""Does the devfreq governor explain the 794ms -> 3700ms gap?

The decode is ~93% kernel-launch overhead, so the GPU is mostly IDLE during
inference.  An idle-looking GPU is exactly what the governor clocks down.
Compare: clock force-warmed (my earlier benchmarks) vs cold, and with the
node's real duty cycle (a gap between inferences) vs back-to-back.
"""
import sys, time, math, importlib.util
sys.path.insert(0,"/benchmarking/simlingo"); sys.path.insert(0,"/benchmarking/simlingo/team_code")
import numpy as np, torch, hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
from PIL import Image as PILImage

DEV=torch.device("cuda"); SIM="/benchmarking/simlingo"
FREQ="/sys/class/devfreq/17000000.gpu/cur_freq"
def mhz():
    try: return int(open(FREQ).read().strip())//10**6
    except Exception: return -1

cfg=OmegaConf.load("/models/simlingo/simlingo/.hydra/config.yaml")
cfg.model.vision_model.use_global_img=cfg.data_module.use_global_img
v=cfg.model.vision_model.variant; vc=f"{SIM}/pretrained/{v.split('/')[1]}"
proc=AutoProcessor.from_pretrained(v,trust_remote_code=True,cache_dir=vc)
tk=proc.tokenizer if hasattr(proc,"tokenizer") else proc
tk.add_special_tokens({"additional_special_tokens":["<WAYPOINTS>","<WAYPOINTS_DIFF>","<ORG_WAYPOINTS_DIFF>","<ORG_WAYPOINTS>","<WAYPOINT_LAST>","<ROUTE>","<ROUTE_DIFF>","<TARGET_POINT>"]})
tk.padding_side="left"
vcfg=AutoConfig.from_pretrained(v,trust_remote_code=True,cache_dir=vc)
NIT=int(((vcfg.force_image_size or vcfg.vision_config.image_size)//vcfg.vision_config.patch_size)**2*(vcfg.downsample_ratio**2))
sp=importlib.util.spec_from_file_location("_c",f"{vc}/conversation.py"); conv=importlib.util.module_from_spec(sp); sp.loader.exec_module(conv)
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
tr=build_transform(input_size=448)
dd=torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
model=hydra.utils.instantiate(cfg.model,cfg_data_module=cfg.data_module,processor=proc,cache_dir=vc,_recursive_=False).to(DEV)
torch.set_default_dtype(dd)
model.load_state_dict(torch.load("/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",map_location=DEV)); model.eval()

img=(np.linspace(0,255,359*1024*3).reshape(359,1024,3)%255).astype(np.uint8)
tiles=dynamic_preprocess(PILImage.fromarray(img),image_size=448,use_thumbnail=False,max_num=2)
pv=torch.stack([tr(t) for t in tiles]); processed=pv.unsqueeze(0).unsqueeze(0)
tps=np.array([[8.,0.5],[16.,1.]],dtype=np.float32)
tpl=conv.get_conv_template("internlm2-chat")
tpl.append_message(tpl.roles[0],"<image>\nCurrent speed: 0.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
tpl.append_message(tpl.roles[1],None); q=tpl.get_prompt()
q=q.replace(tpl.system_template.replace("{system_message}",tpl.system_message)+tpl.sep,"")
q=q.replace("<image>","<img>"+"<IMG_CONTEXT>"*NIT*pv.shape[0]+"</img>",1)
t=tk([q],padding=True,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False)
ll=LanguageLabel(phrase_ids=t["input_ids"].to(DEV),phrase_valid=(t["input_ids"]!=tk.pad_token_id).to(DEV),
                 phrase_mask=(t["input_ids"]!=tk.pad_token_id).to(DEV),
                 placeholder_values=[{tk.convert_tokens_to_ids("<TARGET_POINT>"):tps}],language_string=[q],loss_masking=None)
_,_,_,C,H,W=processed.shape; f_=W/(2*math.tan(110*math.pi/360))
K=torch.tensor([[f_,0,W/2],[0,f_,H/2],[0,0,1]],dtype=torch.float32).unsqueeze(0).to(DEV)
E=torch.eye(4,dtype=torch.float32); E[:3,3]=torch.tensor([-1.5,0.,2.]); E=E.unsqueeze(0).to(DEV)
mi=DrivingInput(camera_images=processed.to(DEV).bfloat16(),image_sizes=None,camera_intrinsics=K,camera_extrinsics=E,
                vehicle_speed=torch.tensor([[0.]],dtype=torch.float32,device=DEV),
                target_point=torch.from_numpy(tps[:1]).float().to(DEV),prompt=ll,prompt_inference=ll)

def infer():
    torch.cuda.synchronize(); a=time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        model(mi)
    torch.cuda.synchronize(); return (time.perf_counter()-a)*1000

print(f"\n--- A. cold clock, back-to-back (no warm-up spin) ---   start gpu={mhz()}MHz")
for i in range(8):
    t=infer(); print(f"   {i}: {t:7.0f} ms   gpu={mhz():5d} MHz")

print(f"\n--- B. force the clock up with a 15s matmul spin ---")
spin=torch.randn(2048,2048,device=DEV,dtype=torch.bfloat16)
t0=time.time()
while time.time()-t0<15:
    for _ in range(20): torch.mm(spin,spin)
torch.cuda.synchronize()
print(f"   gpu after spin = {mhz()} MHz")
for i in range(5):
    t=infer(); print(f"   {i}: {t:7.0f} ms   gpu={mhz():5d} MHz")

print(f"\n--- C. the node's duty cycle: 1.5s idle gap between inferences ---")
for i in range(6):
    time.sleep(1.5)
    t=infer(); print(f"   {i}: {t:7.0f} ms   gpu={mhz():5d} MHz")
