"""Is the ROS gap GIL contention? Run the identical model work with and without
a competing Python thread at 20 Hz (the node's control_hz)."""
import sys, time, math, threading, importlib.util
sys.path.insert(0,"/benchmarking/simlingo"); sys.path.insert(0,"/benchmarking/simlingo/team_code")
import numpy as np, torch, hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
from PIL import Image as PILImage
from scipy.interpolate import PchipInterpolator

DEV=torch.device("cuda"); SIM="/benchmarking/simlingo"
cfg=OmegaConf.load("/models/simlingo/simlingo/.hydra/config.yaml")
cfg.model.vision_model.use_global_img=cfg.data_module.use_global_img
v=cfg.model.vision_model.variant; vc=f"{SIM}/pretrained/{v.split('/')[1]}"
proc=AutoProcessor.from_pretrained(v,trust_remote_code=True,cache_dir=vc)
tok_=proc.tokenizer if hasattr(proc,"tokenizer") else proc
tok_.add_special_tokens({"additional_special_tokens":["<WAYPOINTS>","<WAYPOINTS_DIFF>","<ORG_WAYPOINTS_DIFF>","<ORG_WAYPOINTS>","<WAYPOINT_LAST>","<ROUTE>","<ROUTE_DIFF>","<TARGET_POINT>"]})
tok_.padding_side="left"
vcfg=AutoConfig.from_pretrained(v,trust_remote_code=True,cache_dir=vc)
NIT=int(((vcfg.force_image_size or vcfg.vision_config.image_size)//vcfg.vision_config.patch_size)**2*(vcfg.downsample_ratio**2))
spec=importlib.util.spec_from_file_location("_c",f"{vc}/conversation.py"); conv=importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
tr=build_transform(input_size=448)
dd=torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
model=hydra.utils.instantiate(cfg.model,cfg_data_module=cfg.data_module,processor=proc,cache_dir=vc,_recursive_=False).to(DEV)
torch.set_default_dtype(dd)
model.load_state_dict(torch.load("/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",map_location=DEV)); model.eval()

img=(np.linspace(0,255,600*1024*3).reshape(600,1024,3)%255).astype(np.uint8)
tiles=dynamic_preprocess(PILImage.fromarray(img),image_size=448,use_thumbnail=cfg.model.vision_model.use_global_img,max_num=2)
pv=torch.stack([tr(t) for t in tiles]); processed=pv.unsqueeze(0).unsqueeze(0); npatch=pv.shape[0]
tps=np.array([[8.,0.5],[16.,1.]],dtype=np.float32)
tpl=conv.get_conv_template("internlm2-chat")
tpl.append_message(tpl.roles[0],f"<image>\nCurrent speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
tpl.append_message(tpl.roles[1],None); q=tpl.get_prompt()
q=q.replace(tpl.system_template.replace("{system_message}",tpl.system_message)+tpl.sep,"")
q=q.replace("<image>","<img>"+"<IMG_CONTEXT>"*NIT*npatch+"</img>",1)
t=tok_([q],padding=True,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False)
ll=LanguageLabel(phrase_ids=t["input_ids"].to(DEV),phrase_valid=(t["input_ids"]!=tok_.pad_token_id).to(DEV),
                 phrase_mask=(t["input_ids"]!=tok_.pad_token_id).to(DEV),
                 placeholder_values=[{tok_.convert_tokens_to_ids("<TARGET_POINT>"):tps}],language_string=[q],loss_masking=None)
_,_,_,C,H,W=processed.shape; f_=W/(2*math.tan(110*math.pi/360))
K=torch.tensor([[f_,0,W/2],[0,f_,H/2],[0,0,1]],dtype=torch.float32).unsqueeze(0).to(DEV)
E=torch.eye(4,dtype=torch.float32); E[:3,3]=torch.tensor([-1.5,0.,2.]); E=E.unsqueeze(0).to(DEV)
mi=DrivingInput(camera_images=processed.to(DEV).bfloat16(),image_sizes=None,camera_intrinsics=K,camera_extrinsics=E,
                vehicle_speed=torch.tensor([[5.]],dtype=torch.float32,device=DEV),
                target_point=torch.from_numpy(tps[:1]).float().to(DEV),prompt=ll,prompt_inference=ll)

spin=torch.randn(2048,2048,device=DEV,dtype=torch.bfloat16)
t0=time.time()
while time.time()-t0<15:
    for _ in range(20): torch.mm(spin,spin)
torch.cuda.synchronize()
print("gpu:",int(open("/sys/class/devfreq/17000000.gpu/cur_freq").read())//10**6,"MHz")

def infer():
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        model(mi)

def timed(n=6):
    for _ in range(2): infer()
    ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); a=time.perf_counter(); infer(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-a)*1000)
    return sum(ts)/len(ts), min(ts), max(ts)

print("\n--- A. inference alone (no competing thread) ---")
a,mn,mx=timed(); print(f"  {a:7.1f} ms  (min {mn:.0f}, max {mx:.0f})")

route=np.random.randn(21,2).cumsum(0)*2
def control_body():
    wp=np.concatenate((np.zeros_like(route[:1]),route))
    sh=np.roll(wp,1,axis=0); sh[0]=sh[1]
    d=np.cumsum(np.linalg.norm(wp-sh,axis=1)); d+=np.arange(len(d))*1e-4
    x=np.arange(0.1,d[-1],0.1)
    if len(x): PchipInterpolator(d,wp,axis=0)(x)

stop=threading.Event()
def ctrl_loop(hz):
    per=1.0/hz
    while not stop.is_set():
        s=time.perf_counter(); control_body()
        r=per-(time.perf_counter()-s)
        if r>0: time.sleep(r)

for hz in (2.0, 20.0):
    stop.clear()
    th=threading.Thread(target=ctrl_loop,args=(hz,),daemon=True); th.start()
    time.sleep(0.5)
    a,mn,mx=timed()
    stop.set(); th.join(timeout=2)
    print(f"\n--- B. inference + control thread at {hz:g} Hz ---")
    print(f"  {a:7.1f} ms  (min {mn:.0f}, max {mx:.0f})")

s=time.perf_counter()
for _ in range(200): control_body()
print(f"\n  one control_body(): {(time.perf_counter()-s)/200*1000:.2f} ms  "
      f"-> {20*(time.perf_counter()-s)/200*100:.1f}% of one core at 20 Hz")
