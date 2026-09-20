"""Reproduce simlingo_node's concurrency structure and measure what each part costs.

Configs (env PROBE):
  bare      inference only, no rclpy spinning at all           -> the model's true cost
  full      MultiThreadedExecutor(2) + 10Hz image cb + 20Hz control + logging
  noctrl    full minus the 20 Hz control timer
  noimg     full minus the image subscription
  nolog     full minus the per-frame get_logger().info calls
  single    SingleThreadedExecutor, no control timer  (the 0.89s-era shape)
"""
import os, sys, time, threading, math, importlib.util
sys.path.insert(0,"/benchmarking/simlingo"); sys.path.insert(0,"/benchmarking/simlingo/team_code")
import numpy as np, cv2, torch, hydra, rclpy
from concurrent.futures import ThreadPoolExecutor
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
from PIL import Image as PILImage
from scipy.interpolate import PchipInterpolator

MODE = os.environ.get("PROBE", "full")
DEV  = torch.device("cuda"); SIM="/benchmarking/simlingo"

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

base_img=(np.linspace(0,255,600*1024*3).reshape(600,1024,3)%255).astype(np.uint8)
tps=np.array([[8.,0.5],[16.,1.]],dtype=np.float32)

def build_model_input(img):
    tiles=dynamic_preprocess(PILImage.fromarray(img),image_size=448,
                             use_thumbnail=cfg.model.vision_model.use_global_img,max_num=2)
    pv=torch.stack([tr(t) for t in tiles]); processed=pv.unsqueeze(0).unsqueeze(0)
    tpl=conv.get_conv_template("internlm2-chat")
    tpl.append_message(tpl.roles[0],"<image>\nCurrent speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.")
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
    return DrivingInput(camera_images=processed.to(DEV).bfloat16(),image_sizes=None,camera_intrinsics=K,
                        camera_extrinsics=E,vehicle_speed=torch.tensor([[5.]],dtype=torch.float32,device=DEV),
                        target_point=torch.from_numpy(tps[:1]).float().to(DEV),prompt=ll,prompt_inference=ll)

spin=torch.randn(2048,2048,device=DEV,dtype=torch.bfloat16)
t0=time.time()
while time.time()-t0<15:
    for _ in range(20): torch.mm(spin,spin)
torch.cuda.synchronize()

TIMES=[]
def infer_once(mi):
    torch.cuda.synchronize(); a=time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        model(mi)
    torch.cuda.synchronize()
    TIMES.append((time.perf_counter()-a)*1000)

N_SAMPLES = 12

if MODE == "bare":
    mi=build_model_input(base_img)
    for _ in range(2): infer_once(mi)
    TIMES.clear()
    for _ in range(N_SAMPLES): infer_once(mi)
else:
    rclpy.init()
    node=Node("probe")
    log = (MODE != "nolog")
    pool=ThreadPoolExecutor(max_workers=1)
    state={"img":base_img,"fut":None,"n":0}

    if MODE != "noimg":
        qos=QoSProfile(reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST,depth=5)
        def img_cb(msg):
            arr=np.frombuffer(bytes(msg.data),dtype=np.uint8)
            im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im=im[:int(im.shape[0]*0.75)]
            state["img"]=im
            state["n"]+=1
            if log: node.get_logger().info(f"[IMG] n={state['n']}",throttle_duration_sec=5.0)
        node.create_subscription(CompressedImage,"/carla/hero/rgb_0/compressed",img_cb,qos)

    if MODE not in ("noctrl","single"):
        ctrl_group=MutuallyExclusiveCallbackGroup()
        route=np.random.randn(21,2).cumsum(0)*2
        def ctrl_cb():
            wp=np.concatenate((np.zeros_like(route[:1]),route))
            sh=np.roll(wp,1,axis=0); sh[0]=sh[1]
            d=np.cumsum(np.linalg.norm(wp-sh,axis=1)); d+=np.arange(len(d))*1e-4
            x=np.arange(0.1,d[-1],0.1)
            if len(x): PchipInterpolator(d,wp,axis=0)(x)
        node.create_timer(1.0/20.0, ctrl_cb, callback_group=ctrl_group)

    def timer_cb():
        if state["fut"] and not state["fut"].done(): return
        if log: node.get_logger().info("Submitting inference job…")
        img=state["img"].copy()
        mi=build_model_input(img)
        state["fut"]=pool.submit(infer_once,mi)
    node.create_timer(0.25, timer_cb)

    ex = SingleThreadedExecutor() if MODE=="single" else MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    stop=threading.Event()
    def spinner():
        while not stop.is_set(): ex.spin_once(timeout_sec=0.1)
    th=threading.Thread(target=spinner,daemon=True); th.start()
    t_end=time.time()+90
    while time.time()<t_end and len(TIMES)<N_SAMPLES+3: time.sleep(0.5)
    stop.set(); th.join(timeout=3)
    TIMES[:] = TIMES[3:]
    rclpy.shutdown()

if TIMES:
    a=sum(TIMES)/len(TIMES)
    print(f"RESULT {MODE:8s} n={len(TIMES):2d}  mean={a:7.1f} ms  min={min(TIMES):7.1f}  max={max(TIMES):7.1f}", flush=True)
else:
    print(f"RESULT {MODE:8s} no samples", flush=True)
