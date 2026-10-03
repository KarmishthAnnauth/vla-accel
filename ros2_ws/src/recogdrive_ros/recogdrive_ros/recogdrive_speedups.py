"""Inference-time speedups for ReCogDrive, applied to the *built agent*.

Nothing in here edits ReCogDrive sources: FastReCogDrive wraps the agent the
node already has, reads its sub-modules, and reproduces what
``ReCogDriveAgent.compute_trajectory`` computes with less work.  The agent is
left untouched and remains the fallback for any input the fast path does not
cover.

What the reference path spends its time on (Orin, clocks pinned, 1262 ms):

  image load + tiling, CPU        72 ms   PIL resizes, 9x ToTensor/Normalize
  tokenisation                    27 ms   slow tokenizer over 2800 tokens
  vision encoder, 9 tiles        298 ms   compute-bound
  LLM body, 2800 tokens          486 ms   compute-bound + ~60 ms dispatch
  lm_head + logits.float()        80 ms   logits over 2800 x 151682 nobody reads
  diffusion planner, 5 steps     283 ms   pure dispatch (64-token context: 267)

and what is done about each:

  * image: the two PIL resizes run in threads (the big one in strips, same
    pixels), ToTensor/Normalize once on the GPU.
  * tokenisation: everything up to and including the image tokens is constant;
    only the text after ``</img>`` is tokenised per frame.
  * LLM: the lm_head is skipped (only hidden_states[-1] is used).  The 291
    tokens before the image (system prompt) are the same every frame: their
    keys/values and final hidden states are computed once and reused (KV-cache
    reuse).  The rest runs in a fixed-size window under a CUDA graph; padding
    rows, which the reference computes as a constant, are filled in from a
    stored copy.
  * planner: the denoising loop is captured into one CUDA graph; the
    cross-attention keys/values of the VLM tokens, identical in all 5 steps,
    are computed once per frame.
  * vision encoder: CUDA graph.
  * both transformers: the big matmuls go through a pre-transposed weight
    (F.linear's transposed bf16 GEMM runs at half speed for 8960 -> 1536), and
    the elementwise chains (RMSNorm, RoPE, SiLU-gate, layer-scale) are fused
    into one kernel each with torch.compile.

There is no LoRA to merge: the released VLM has use_llm_lora = 0.

The VLM half is bit-identical to the reference, not merely close.  That takes
care: bf16 results depend on where values are rounded and on which GEMM kernel
runs, and a last-bit difference grows to ~5 % of the final hidden state.  So
every fused kernel rounds where eager rounds (_rne), cached rows come from a
full-length pass, and each replaced op is compared against the stock op at
start-up and dropped if a single bit differs.  The planner (fp32) agrees to
~1e-6.  ``verify()`` measures all of it against the unmodified agent.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = 448
TOKENS_PER_TILE = 256
SOURCE_MAX_LENGTH = 2800
TAIL_CAPACITY = 232
RESIZE_STRIPS = 4
THUMBNAIL_STRIPS = 2


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


_F32, _BF16 = torch.float32, torch.bfloat16


def _rne(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> the bf16 value eager's cast gives (round to nearest even), kept
    as fp32.  An eager bf16 op is "compute in fp32, round to bf16"; a fused
    kernel computes the whole chain in fp32 and rounds once at the store, so
    the intermediate roundings have to be put back by hand."""
    i = v.view(torch.int32)
    return ((i + 0x7FFF + ((i >> 16) & 1)) & -65536).view(_F32)


def _rope_ref(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


def _rope_fused(x, cos, sin):
    xf = x.to(_F32)
    return (_rne(xf * cos.to(_F32)) + _rne(_rotate_half(xf) * sin.to(_F32))).to(_BF16)


def _silu_mul_ref(gate, up):
    return F.silu(gate) * up


def _silu_mul_fused(gate, up):
    g = gate.to(_F32)
    return (_rne(g / (1 + torch.exp(-g))) * up.to(_F32)).to(_BF16)


def _square_fused(x):
    return x.to(_F32).pow(2)


def _rms_tail_fused(x, inv_rms, weight):
    return (_rne(x.to(_F32) * inv_rms) * weight.to(_F32)).to(_BF16)


def _scale_add_ref(x, y, scale):
    return x + y * scale


def _scale_add_fused(x, y, scale):
    return (x.to(_F32) + _rne(y.to(_F32) * scale.to(_F32))).to(_BF16)


class _FastLinear:
    """nn.Linear through a pre-transposed, contiguous copy of its weight."""

    def __init__(self, linear: torch.nn.Linear):
        self.weight_t = linear.weight.t().contiguous()
        self.bias = linear.bias
        self.out_features = linear.out_features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        y = torch.mm(flat, self.weight_t) if self.bias is None else \
            torch.addmm(self.bias, flat, self.weight_t)
        return y.view(*x.shape[:-1], self.out_features)


class _TrtVit:
    """A TensorRT engine for extract_feature (recogdrive_env/opt/trt_vit.py):
    fp32 (9, 3, 448, 448) in, fp32 (9, 256, hidden) out.  Opt-in and NOT
    bit-identical to the reference: fp16 moves the plan by ~0.08 m on average,
    the rebalanced partial INT8 by ~0.13 m (reference sampling spread 0.20 m)."""

    def __init__(self, path: str, pixels: torch.Tensor):
        import tensorrt as trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        shape = tuple(self._engine.get_tensor_shape("pixels"))
        if shape != tuple(pixels.shape):
            raise ValueError(f"engine takes {shape}, the fast path feeds {tuple(pixels.shape)}")
        self.inp = torch.zeros(shape, device=pixels.device, dtype=torch.float32)
        self.out = torch.zeros(tuple(self._engine.get_tensor_shape("features")),
                               device=pixels.device, dtype=torch.float32)
        self._context.set_tensor_address("pixels", self.inp.data_ptr())
        self._context.set_tensor_address("features", self.out.data_ptr())
        self._stream = torch.cuda.Stream()

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        current = torch.cuda.current_stream()
        self.inp.copy_(pixels)
        self._stream.wait_stream(current)
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        current.wait_stream(self._stream)
        return self.out


class _Stage:
    """A function of static tensors, run eagerly or replayed as a CUDA graph."""

    def __init__(self, fn: Callable[[], torch.Tensor], capture: bool, warmup: int = 2):
        self._fn = fn
        self._graph = None
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            if capture:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.out = fn()
                self._graph = graph
            else:
                self.out = fn()

    def __call__(self) -> torch.Tensor:
        if self._graph is not None:
            self._graph.replay()
        else:
            with torch.no_grad():
                self.out = self._fn()
        return self.out


class FastReCogDrive:
    """compute_trajectory for a built ReCogDriveAgent, single frame, batch 1.

    agent      the ReCogDriveAgent after initialize() (InternVL backbone,
               DDIM planner, no-cache mode)
    n_tiles    tiles the camera frame is cut into; 9 for 16:9 (4x2 + thumbnail).
               Frames that tile differently go to the reference path.
    capture    False runs the same code without CUDA graphs (debugging).
    fuse       False keeps the stock linears and elementwise ops.
    vit_engine a TensorRT engine for the vision encoder (see _TrtVit); "" = the
               PyTorch encoder, which is the bit-identical one.
    """

    def __init__(self, agent, n_tiles: int = 9, capture: bool = True, fuse: bool = True,
                 log: Optional[Callable[[str], None]] = None, profile: bool = False,
                 vit_engine: str = ""):
        from navsim.agents.recogdrive import recogdrive_backbone as rb
        from navsim.agents.recogdrive.utils.conversation import get_conv_template
        from navsim.agents.recogdrive.utils.internvl_preprocess import (
            IMAGENET_MEAN, IMAGENET_STD, find_closest_aspect_ratio)
        from navsim.agents.recogdrive.utils.utils import format_number
        from navsim.common.dataclasses import Trajectory

        self._rb, self._get_conv_template = rb, get_conv_template
        self._find_ratio, self._format_number, self._Trajectory = (
            find_closest_aspect_ratio, format_number, Trajectory)
        self._log = log or (lambda msg: None)
        self.profile = profile
        self.timing: Dict[str, float] = {}
        self.fallbacks = 0

        self.agent = agent
        if agent.backbone is None or agent.vlm_type != "internvl":
            raise ValueError("FastReCogDrive needs the in-agent InternVL backbone (no-cache mode)")
        if agent.action_head.config.sampling_method != "ddim" or agent.grpo:
            raise ValueError("FastReCogDrive implements the DDIM inference sampler only")
        self._bb = agent.backbone
        self._vlm = self._bb.model
        self._lm = self._vlm.language_model
        self._tok = self._bb.tokenizer
        self._ah = agent.action_head
        self._dev = torch.device("cuda")
        self._vdtype = next(self._vlm.parameters()).dtype
        self._pdtype = next(self._ah.parameters()).dtype
        self._hidden = self._lm.config.hidden_size
        self._builder = agent.get_feature_builders()[0]
        self.n_tiles = n_tiles
        self._n_img = n_tiles * self._bb.num_image_token
        self._pool = ThreadPoolExecutor(RESIZE_STRIPS + THUMBNAIL_STRIPS, thread_name_prefix="recog_resize")
        self._ratio_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._mean = torch.tensor(IMAGENET_MEAN, device=self._dev).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self._dev).view(1, 3, 1, 1)

        self._pixels = torch.zeros(n_tiles, 3, IMAGE_SIZE, IMAGE_SIZE,
                                   device=self._dev, dtype=self._vdtype)
        self._trt_vit = _TrtVit(vit_engine, self._pixels) if vit_engine else None
        self.exact = self._trt_vit is None
        if self._trt_vit is not None:
            self._log(f"FastReCogDrive: vision encoder from TensorRT engine {vit_engine} "
                      f"(not bit-identical to the reference)")
        with torch.no_grad():
            self._init_prompt()
            self._init_llm()
            self._init_kernels(fuse and self._vdtype == _BF16)
            self._init_planner()
            self._init_step(fuse)
            t0 = time.time()
            self._vit = _Stage(self._vit_fn, capture and self._trt_vit is None)
            self._llm = _Stage(self._llm_fn, capture)
            self._planner = _Stage(self._planner_fn, capture)
            torch.cuda.synchronize()
        self._log(f"FastReCogDrive ready: {self._n_prefix} prefix tokens cached, window "
                  f"{self._win.shape[1]} tokens, planner context {self._ctx.shape[1]}, "
                  f"{'CUDA graphs captured' if capture else 'eager'} in {time.time() - t0:.1f} s")

    def _query(self, question: str) -> str:
        """RecogDriveBackbone.forward's prompt, before the image tokens go in."""
        template = self._get_conv_template("internvl2_5")
        template.system_message = self._rb.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def _question(self, history: torch.Tensor, command_one_hot: torch.Tensor) -> str:
        """ReCogDriveAgent.forward's question for one sample."""
        fmt = self._format_number
        command = ['turn left', 'go straight', 'turn right'][int(torch.argmax(command_one_hot))]
        history_str = ' '.join([
            f'   - t-{3-j}: ({fmt(history[j, 0].item())}, '
            f'{fmt(history[j, 1].item())}, '
            f'{fmt(history[j, 2].item())})'
            for j in range(history.shape[0])
        ])
        prompt = (
            "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{command.upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        return f"{prompt}{output_requirements}"

    def _init_prompt(self) -> None:
        rb, tok = self._rb, self._tok
        query = self._query(self._question(torch.zeros(4, 3), torch.tensor([0.0, 1.0, 0.0])))
        self._pre_text, _ = query.split('<image>', 1)
        self._prefix_ids = tok(self._pre_text + rb.IMG_START_TOKEN)["input_ids"]
        self._n_prefix = len(self._prefix_ids)
        self._img_end_id = tok.convert_tokens_to_ids(rb.IMG_END_TOKEN)
        self._tail_cache: Tuple[str, Optional[torch.Tensor]] = ("", None)
        full = query.replace('<image>', rb.IMG_START_TOKEN + rb.IMG_CONTEXT_TOKEN * self._n_img
                             + rb.IMG_END_TOKEN, 1)
        ids = tok(full)["input_ids"]
        tail = self._tail_ids(query)
        expect = self._prefix_ids + [self._bb.img_context_token_id] * self._n_img + tail.tolist()
        if ids != expect:
            raise RuntimeError("prefix / image / tail tokenisation does not match the full prompt")

    def _tail_ids(self, query: str) -> Optional[torch.Tensor]:
        """Token ids from ``</img>`` to the end of the prompt (CPU tensor), or
        None if the prompt does not start with the cached prefix."""
        pre, post = query.split('<image>', 1)
        if pre != self._pre_text:
            return None
        if self._tail_cache[0] != post:
            ids = [self._img_end_id] + self._tok(post)["input_ids"]
            self._tail_cache = (post, torch.tensor(ids, dtype=torch.long))
        return self._tail_cache[1]

    def _grid(self, width: int, height: int) -> Tuple[int, int]:
        """dynamic_preprocess's tile grid (min_num=1, max_num=12)."""
        key = (width, height)
        if key not in self._ratio_cache:
            ratios = sorted(
                {(i, j) for n in range(1, 13) for i in range(1, n + 1) for j in range(1, n + 1)
                 if 1 <= i * j <= 12}, key=lambda x: x[0] * x[1])
            self._ratio_cache[key] = self._find_ratio(width / height, ratios, width, height, IMAGE_SIZE)
        return self._ratio_cache[key]

    def _load_tiles(self, image: Image.Image) -> Optional[np.ndarray]:
        """load_image's tiles as uint8 (n, 448, 448, 3): the resized image cut
        into its grid, then the thumbnail.  Same pixels as dynamic_preprocess;
        the big resize is done in horizontal strips on a thread pool."""
        width, height = image.size
        cols, rows = self._grid(width, height)
        if cols * rows == 1 or cols * rows + 1 != self.n_tiles:
            return None
        big = self._resize(image, IMAGE_SIZE * cols, IMAGE_SIZE * rows, RESIZE_STRIPS)
        thumb = self._resize(image, IMAGE_SIZE, IMAGE_SIZE, THUMBNAIL_STRIPS)
        big = np.concatenate([np.asarray(s.result()) for s in big], axis=0)
        thumb = np.concatenate([np.asarray(s.result()) for s in thumb], axis=0)
        tiles = big.reshape(rows, IMAGE_SIZE, cols, IMAGE_SIZE, 3).transpose(0, 2, 1, 3, 4)
        tiles = tiles.reshape(rows * cols, IMAGE_SIZE, IMAGE_SIZE, 3)
        return np.concatenate([tiles, thumb[None]], axis=0)

    def _resize(self, image: Image.Image, width: int, height: int, strips: int) -> list:
        """image.resize((width, height)) as futures of horizontal strips.  A
        strip is the same resize restricted to its rows (`box`), so the rows
        come out as in the one-shot resize."""
        src_w, src_h = image.size
        bounds = [height * s // strips for s in range(strips + 1)]
        return [self._pool.submit(image.resize, (width, b - a),
                                  box=(0, a * src_h / height, src_w, b * src_h / height))
                for a, b in zip(bounds[:-1], bounds[1:])]

    def _set_pixels(self, tiles: np.ndarray) -> None:
        """ToTensor + Normalize (fp32, as torchvision does them), then the
        reference's cast to the VLM dtype."""
        x = torch.from_numpy(np.ascontiguousarray(tiles)).to(self._dev)
        x = x.permute(0, 3, 1, 2).to(torch.float32).div(255)
        x = x.sub_(self._mean).div_(self._std)
        self._pixels.copy_(x)

    def _vit_fn(self) -> torch.Tensor:
        """InternVLChatModel.extract_feature."""
        if self._trt_vit is not None:
            return self._trt_vit(self._pixels.to(torch.float32)).to(self._vdtype)
        if not self._vit_custom:
            return self._vlm.extract_feature(self._pixels)
        vlm = self._vlm
        vision = vlm.vision_model
        x = vision.embeddings(self._pixels)
        b, n, c = x.shape
        for layer, (qkv, fc1, fc2) in zip(vision.encoder.layers, self._vit_linears):
            attn = layer.attn
            packed = qkv(layer.norm1(x).to(x.dtype)).view(b, n, 3, attn.num_heads, c // attn.num_heads)
            context, _ = attn.inner_attn(packed, key_padding_mask=None, need_weights=False, causal=False)
            x = self._scale_add(x, attn.proj(context.reshape(b, n, c)), layer.ls1)
            x = self._scale_add(x, fc2(layer.mlp.act(fc1(layer.norm2(x).to(x.dtype)))), layer.ls2)
        x = x[:, 1:, :]
        side = int(x.shape[1] ** 0.5)
        x = vlm.pixel_shuffle(x.reshape(b, side, side, -1), scale_factor=vlm.downsample_ratio)
        return vlm.mlp1(x.reshape(b, -1, x.shape[-1]))

    def _init_kernels(self, fuse: bool) -> None:
        dev, dt = self._dev, self._vdtype
        layers = self._lm.model.layers
        vision = self._vlm.vision_model
        n_win = self._win.shape[1]
        used, dropped = [], []

        def keep(name: str, same: bool) -> bool:
            (used if same else dropped).append(name)
            return same

        def fused(name, fn, ref, *samples):
            """torch.compile(fn) if it reproduces ref on every sample, else ref."""
            if not fuse:
                return ref
            try:
                compiled = torch.compile(fn)
                same = all(torch.equal(compiled(*a), ref(*a)) for a in samples)
            except Exception as exc:
                self._log(f"FastReCogDrive: {name} not fused ({type(exc).__name__}: {str(exc)[:120]})")
                same = False
            return compiled if keep(name, same) else ref

        def linears(name, modules, rows):
            """_FastLinear for each module if the first one reproduces the stock
            module on an input of the real shape, else the modules themselves."""
            if not fuse:
                return list(modules)
            x = torch.randn(1, rows, modules[0].in_features, device=dev, dtype=dt)
            fast = [_FastLinear(m) for m in modules]
            return fast if keep(name, torch.equal(fast[0](x), modules[0](x))) else list(modules)

        g = torch.Generator(device=dev).manual_seed(0)
        rand = lambda *shape: torch.randn(*shape, device=dev, dtype=dt, generator=g)
        mlps = [layer.mlp for layer in layers]
        self._llm_linears = list(zip(
            linears("llm gate_proj", [m.gate_proj for m in mlps], n_win),
            linears("llm up_proj", [m.up_proj for m in mlps], n_win),
            linears("llm down_proj", [m.down_proj for m in mlps], n_win)))
        q = rand(1, n_win, self._heads, self._head_dim)
        k = rand(1, n_win, self._kv_heads, self._head_dim)
        self._rope = fused("rope", _rope_fused, _rope_ref, (q, self._cos, self._sin), (k, self._cos, self._sin))

        self._silu_mul = None
        if isinstance(mlps[0].act_fn, torch.nn.SiLU):
            inter = mlps[0].gate_proj.out_features
            every = torch.arange(1 << 16, device=dev, dtype=torch.int32).to(torch.int16).view(_BF16)
            every = torch.where(torch.isfinite(every), every, torch.zeros_like(every))
            gate = every.repeat(n_win * inter // every.numel() + 1)[:n_win * inter].view(1, n_win, inter)
            self._silu_mul = fused("silu-gate", _silu_mul_fused, _silu_mul_ref,
                                   (gate, torch.ones_like(gate)), (gate, rand(1, n_win, inter)))
            del gate

        norm0 = layers[0].input_layernorm
        x = rand(1, n_win, self._hidden) * 20
        self._rms = lambda norm, t: norm(t)
        if fuse:
            try:
                square, tail = torch.compile(_square_fused), torch.compile(_rms_tail_fused)

                def rms(norm, t):
                    variance = square(t).mean(-1, keepdim=True)
                    return tail(t, torch.rsqrt(variance + norm.variance_epsilon), norm.weight)
                if keep("rmsnorm", torch.equal(rms(norm0, x), norm0(x))):
                    self._rms = rms
            except Exception as exc:
                self._log(f"FastReCogDrive: rmsnorm not fused ({type(exc).__name__}: {str(exc)[:120]})")
                keep("rmsnorm", False)

        vlayers = vision.encoder.layers
        attn0 = vlayers[0].attn
        self._vit_custom = bool(
            self._vlm.select_layer == -1 and attn0.use_flash_attn and not attn0.qk_normalization)
        if self._vit_custom:
            tokens = vision.embeddings.num_positions
            self._vit_linears = list(zip(
                linears("vit qkv", [layer.attn.qkv for layer in vlayers], self.n_tiles * tokens),
                linears("vit fc1", [layer.mlp.fc1 for layer in vlayers], self.n_tiles * tokens),
                linears("vit fc2", [layer.mlp.fc2 for layer in vlayers], self.n_tiles * tokens)))
            width = attn0.embed_dim
            a, b = rand(self.n_tiles, tokens, width), rand(self.n_tiles, tokens, width)
            self._scale_add = fused("vit layer-scale", _scale_add_fused, _scale_add_ref, (a, b, vlayers[0].ls1))
            self._pixels.copy_(rand(*self._pixels.shape))
            self._vit_custom = keep("vit encoder loop",
                                    torch.equal(self._vit_fn(), self._vlm.extract_feature(self._pixels)))
            self._pixels.zero_()
        self._log("FastReCogDrive: bit-identical replacements in use: " + (", ".join(used) or "none")
                  + (f"; dropped (not bit-identical here): {', '.join(dropped)}" if dropped else ""))

    def _init_llm(self) -> None:
        """The per-frame constants of the LLM: keys/values and final hidden
        states of the prefix, and the final hidden state of a padding row.

        They are taken from one stock forward of the reference's own length
        (left-padded to 2800), not computed on their own.  On this GPU a bf16
        matmul rounds a row differently depending on how many rows the batch
        has (measured: one result for 291..2816 rows, another above and for
        tiny batches), and that last-bit difference grows to ~5 % of the final
        hidden state.  Rows from a 2800-row pass are the reference's rows."""
        lm, dev, tok = self._lm, self._dev, self._tok
        embed = lm.get_input_embeddings()
        self._embed = embed
        n_win = self._n_img + TAIL_CAPACITY
        self._win = embed(torch.full((1, n_win), tok.pad_token_id, device=dev)).clone()

        n_pad = 16
        n_fill = SOURCE_MAX_LENGTH - n_pad - self._n_prefix
        ids = [tok.pad_token_id] * n_pad + self._prefix_ids
        head = embed(torch.tensor(ids, device=dev))[None]
        mask = torch.ones(1, SOURCE_MAX_LENGTH, dtype=torch.long, device=dev)
        mask[:, :n_pad] = 0
        position_ids = mask.cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        out = lm.model(inputs_embeds=torch.cat((head, self._win[:, :n_fill]), dim=1),
                       attention_mask=mask, position_ids=position_ids, use_cache=True,
                       return_dict=True)
        rows = slice(n_pad, n_pad + self._n_prefix)
        self._h_pad = out.last_hidden_state[0, 0].to(self._pdtype)
        self._prefix_h = out.last_hidden_state[0, rows].to(self._pdtype)
        self._prefix_kv = [(k[:, :, rows].transpose(1, 2).contiguous(),
                            v[:, :, rows].transpose(1, 2).contiguous())
                           for k, v in out.past_key_values]
        del out

        attn0 = lm.model.layers[0].self_attn
        self._heads, self._kv_heads, self._head_dim = (
            attn0.num_heads, attn0.num_key_value_heads, attn0.head_dim)
        pos = torch.arange(self._n_prefix, self._n_prefix + n_win, device=dev)
        cos, sin = attn0.rotary_emb(self._win, seq_len=self._n_prefix + n_win)
        self._cos = cos[pos].to(self._vdtype)[None, :, None, :]
        self._sin = sin[pos].to(self._vdtype)[None, :, None, :]

    def _llm_fn(self) -> torch.Tensor:
        """Qwen2 decoder over the window, attending to the cached prefix.

        The window is [image tokens | tail | unused slots]; attention is
        causal, so the unused slots at the end cannot influence the rows that
        are read.  flash_attn_func aligns its causal mask bottom-right when
        there are more keys than queries, which is the KV-cache case."""
        from flash_attn import flash_attn_func
        x = self._win
        n = x.shape[1]
        cos, sin = self._cos, self._sin
        for layer, (pk, pv), (gate, up, down) in zip(
                self._lm.model.layers, self._prefix_kv, self._llm_linears):
            attn = layer.self_attn
            h = self._rms(layer.input_layernorm, x)
            q = attn.q_proj(h).view(1, n, self._heads, self._head_dim)
            k = attn.k_proj(h).view(1, n, self._kv_heads, self._head_dim)
            v = attn.v_proj(h).view(1, n, self._kv_heads, self._head_dim)
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
            a = flash_attn_func(q, torch.cat((pk, k), dim=1), torch.cat((pv, v), dim=1), causal=True)
            x = x + attn.o_proj(a.reshape(1, n, self._hidden))
            h = self._rms(layer.post_attention_layernorm, x)
            if self._silu_mul is None:
                x = x + layer.mlp(h)
            else:
                x = x + down(self._silu_mul(gate(h), up(h)))
        return self._rms(self._lm.model.norm, x)

    def _init_planner(self) -> None:
        ah, dev = self._ah, self._dev
        cfg, dit = ah.config, ah.model
        self._steps, self._horizon = ah.ddim_steps, cfg.action_horizon
        n_ctx = max(SOURCE_MAX_LENGTH, self._n_prefix + self._win.shape[1])
        self._ctx = torch.zeros(1, n_ctx, self._hidden, device=dev, dtype=self._pdtype)
        self._ctx_bias = torch.zeros(1, 1, 1, n_ctx, device=dev, dtype=self._pdtype)
        self._ctx_maskf = torch.zeros(1, n_ctx, 1, device=dev, dtype=self._pdtype)
        self._ctx_inv_len = torch.zeros((), device=dev, dtype=self._pdtype)
        self._ctx_layout: Tuple[int, int] = (-1, -1)
        self._his = torch.zeros(1, 12, device=dev, dtype=self._pdtype)
        self._status = torch.zeros(1, 8, device=dev, dtype=self._pdtype)
        self._noise = torch.zeros(self._steps + 1, 1, self._horizon, cfg.action_dim,
                                  device=dev, dtype=self._pdtype)
        self._set_context_layout(SOURCE_MAX_LENGTH - 1)

        self._t = [ah.make_timesteps(1, ah.ddim_t[i], dev) for i in range(self._steps)]
        self._index = [ah.make_timesteps(1, i, dev) for i in range(self._steps)]
        self._time_emb = [dit.timestep_encoder(t) for t in self._t]
        self._etas = ah.eta(self._noise[0]).unsqueeze(1)
        self._pos = ah.position_embedding(torch.arange(self._horizon, device=dev)) \
            if hasattr(ah, 'position_embedding') else None
        self._min_std = getattr(ah, 'eval_min_sampling_denoising_std', 0.0001)
        self._randn_clip = getattr(ah, 'eval_randn_clip_value', 1.0)
        self._denoised_clip = getattr(ah, 'denoised_clip_value', 1.0)
        self._eps_clip = getattr(ah, 'eps_clip_value', None)
        self._final_clip = getattr(ah, 'final_action_clip_value', 1.0)
        dummy = torch.zeros(1, self._horizon, dit.inner_dim, device=dev, dtype=self._pdtype)
        self._rcos, self._rsin = dit.rotary_embedder(
            dummy, torch.arange(self._horizon, device=dev).unsqueeze(0))

    def _set_context_layout(self, n_real: int) -> None:
        """Lay the planner's context out as the reference's left-padded
        sequence: [padding rows | prefix | image + tail]."""
        n_pad = max(0, SOURCE_MAX_LENGTH - n_real)
        if (n_pad, n_real) == self._ctx_layout:
            return
        if n_pad != self._ctx_layout[0]:
            self._ctx[0, :n_pad] = self._h_pad
            self._ctx[0, n_pad:n_pad + self._n_prefix] = self._prefix_h
        total = n_pad + n_real
        self._ctx_bias[..., :total] = 0.0
        self._ctx_bias[..., total:] = float('-inf')
        self._ctx_maskf[:, :total] = 1.0
        self._ctx_maskf[:, total:] = 0.0
        self._ctx_inv_len.fill_(1.0 / total)
        self._ctx_layout = (n_pad, n_real)

    def _dit_attention(self, attn, x: torch.Tensor, context_kv) -> torch.Tensor:
        """blocks.attention.Attention.forward with the context keys/values
        precomputed (cross-attention) and RoPE's cos/sin cached."""
        n = x.shape[1]
        q = attn.to_q(x).view(1, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        q = (q * self._rcos) + (_rotate_half(q) * self._rsin)
        if context_kv is None:
            k = attn.to_k(x).view(1, n, attn.num_heads, attn.head_dim).transpose(1, 2)
            v = attn.to_v(x).view(1, n, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_norm(k)
            k = (k * self._rcos) + (_rotate_half(k) * self._rsin)
            weights = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
        else:
            k, v = context_kv
            weights = torch.matmul(q, k.transpose(-2, -1)) * attn.scale + self._ctx_bias
        out = torch.matmul(weights.softmax(dim=-1), v)
        return attn.to_out(out.transpose(1, 2).reshape(1, n, -1))

    def _dit(self, hidden: torch.Tensor, context_kv: List, ego: torch.Tensor,
             time_emb: torch.Tensor) -> torch.Tensor:
        """LightningDiT.forward."""
        dit = self._ah.model
        conditioning = time_emb + ego
        cross = iter(context_kv)
        for idx, block in enumerate(dit.transformer_blocks):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
                block.adaLN_modulation(conditioning).chunk(6, dim=1)
            m = block.norm1(hidden) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
            use_cross = not (idx % 2 == 0 and dit.interleave_attention)
            a = self._dit_attention(block.attn, m, next(cross) if use_cross else None)
            hidden = hidden + gate_a.unsqueeze(1) * a
            m = block.norm2(hidden) * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
            hidden = hidden + gate_f.unsqueeze(1) * block.ffn(m)
        final = dit.final_layer
        shift, scale = final.modulation_proj(conditioning).chunk(2, dim=1)
        return final.linear(final.norm_final(hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))

    def _planner_fn(self) -> torch.Tensor:
        """ReCogDriveDiffusionPlanner.get_action, DDIM branch, batch 1, with
        the noise drawn by the caller (self._noise) in the reference's order."""
        ah = self._ah
        dit, horizon = ah.model, self._horizon
        vl = ah.feature_encoder(self._ctx)
        vl_mean = ((vl * self._ctx_maskf).sum(1) * self._ctx_inv_len).unsqueeze(1).repeat(1, horizon, 1)
        history = ah.his_traj_encoder(self._his.unsqueeze(1)).repeat(1, horizon, 1)
        ego = ah.ego_status_encoder(self._status)

        context_kv = []
        for idx, block in enumerate(dit.transformer_blocks):
            if idx % 2 == 0 and dit.interleave_attention:
                continue
            attn = block.attn
            k = attn.to_k(vl).view(1, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            v = attn.to_v(vl).view(1, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            context_kv.append((attn.k_norm(k), v))

        x = self._noise[0]
        for i in range(self._steps):
            x = self._step(x, self._t[i], self._index[i], self._time_emb[i], self._noise[i + 1],
                           history, vl_mean, ego, context_kv)
        if self._final_clip is not None:
            x = x.clamp(-self._final_clip, self._final_clip)
        return ah.denorm_odo(x)

    def _ddim_step(self, x, t, index, time_emb, noise, history, vl_mean, ego, context_kv):
        """One iteration of get_action's DDIM loop: p_mean_variance, then the
        sampling step."""
        ah = self._ah
        features = ah.action_encoder(x, t)
        if self._pos is not None:
            features = features + self._pos
        fused = ah.fusion_projector(torch.cat((history, vl_mean, features), dim=2))
        pred_noise = ah.action_decoder(self._dit(fused, context_kv, ego, time_emb))

        alpha_t = ah.extract(ah.ddim_alphas, index, x.shape)
        sqrt_one_minus_alpha_t = ah.extract(ah.ddim_sqrt_one_minus_alphas, index, x.shape)
        x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t ** 0.5)
        x_recon = x_recon.clamp(-self._denoised_clip, self._denoised_clip)
        alpha_prev = ah.extract(ah.ddim_alphas_prev, index, x.shape)
        pred_noise = (x - (alpha_t ** 0.5) * x_recon) / sqrt_one_minus_alpha_t
        if self._eps_clip is not None:
            pred_noise = pred_noise.clamp(-self._eps_clip, self._eps_clip)
        sigma = (
            self._etas
            * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
        ).clamp(min=1e-10)
        pred_dir_xt = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
        mean = (alpha_prev ** 0.5) * x_recon + pred_dir_xt
        logvar = torch.log(sigma ** 2 + 1e-20)
        std = torch.exp(0.5 * logvar).clamp(min=self._min_std)
        return mean + std * noise.clamp(-self._randn_clip, self._randn_clip)

    def _init_step(self, fuse: bool) -> None:
        """torch.compile the denoising step (fp32: fusing its ~800 tiny kernels
        is what the planner's time is), kept only if it agrees with eager."""
        self._step = self._ddim_step
        if not fuse:
            return
        dev, dt, ah = self._dev, self._pdtype, self._ah
        g = torch.Generator(device=dev).manual_seed(0)
        rand = lambda *shape: torch.randn(*shape, device=dev, dtype=dt, generator=g)
        dim, heads = ah.model.inner_dim, ah.model.num_heads
        n_ctx = self._ctx.shape[1]
        cross = sum(1 for i in range(len(ah.model.transformer_blocks))
                    if not (i % 2 == 0 and ah.model.interleave_attention))
        kv = [(rand(1, heads, n_ctx, dim // heads), rand(1, heads, n_ctx, dim // heads)) for _ in range(cross)]
        width = ah.config.input_embedding_dim
        args = [(rand(1, self._horizon, ah.config.action_dim), self._t[i], self._index[i], self._time_emb[i],
                 rand(1, self._horizon, ah.config.action_dim), rand(1, self._horizon, width),
                 rand(1, self._horizon, width), rand(1, dim), kv) for i in (0, self._steps - 1)]
        try:
            compiled = torch.compile(self._ddim_step)
            worst = max((compiled(*a) - self._ddim_step(*a)).abs().max().item() for a in args)
        except Exception as exc:
            self._log(f"FastReCogDrive: planner step not compiled ({type(exc).__name__}: {str(exc)[:160]})")
            return
        if worst < 1e-4:
            self._step = compiled
            self._log(f"FastReCogDrive: planner step compiled (max deviation from eager {worst:.1e}, "
                      f"normalised action units)")
        else:
            self._log(f"FastReCogDrive: compiled planner step deviates by {worst:.1e}; keeping eager")

    def _tick(self, name: str, t0: float) -> float:
        if not self.profile:
            return t0
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.timing[name] = (now - t0) * 1e3
        return now

    def hidden_state(self) -> torch.Tensor:
        """The planner's VLM input of the last frame, as the reference lays it
        out: (1, max(2800, real tokens), hidden)."""
        n_pad, n_real = self._ctx_layout
        return self._ctx[:, :n_pad + n_real]

    @torch.no_grad()
    def plan(self, agent_input, image: Optional[np.ndarray] = None):
        """ReCogDriveAgent.compute_trajectory on the fast path, or None if this
        input is outside it (tile grid other than n_tiles, or a prompt longer
        than the window): the caller then uses the reference.

        image: the front frame as an RGB uint8 array (H, W, 3); without it the
        file agent_input names is read, as the reference does."""
        t0 = time.perf_counter()
        self.timing = {}
        self.agent.eval()
        feats = self._builder.compute_features(agent_input)
        history = feats["history_trajectory"]
        tail = self._tail_ids(self._query(self._question(history, feats["high_command_one_hot"])))
        if tail is None or len(tail) > TAIL_CAPACITY:
            return None
        pil = Image.fromarray(image) if image is not None else \
            Image.open(str(agent_input.cameras[-1].cam_f0.image)).convert('RGB')
        tiles = self._load_tiles(pil)
        if tiles is None:
            return None
        self._set_pixels(tiles)
        t0 = self._tick("image", t0)

        vit = self._vit()
        t0 = self._tick("vit", t0)

        n_tail = len(tail)
        n_win = self._n_img + n_tail
        self._win[0, :self._n_img] = vit.reshape(-1, self._hidden)
        self._win[0, self._n_img:n_win] = self._embed(tail.to(self._dev))
        hidden = self._llm()
        t0 = self._tick("llm", t0)

        self._set_context_layout(self._n_prefix + n_win)
        n_pad = self._ctx_layout[0]
        start = n_pad + self._n_prefix
        self._ctx[0, start:start + n_win] = hidden[0, :n_win]
        self._his.copy_(history.reshape(1, -1))
        self._status.copy_(feats["status_feature"].reshape(1, -1))
        for i in range(self._steps + 1):
            self._noise[i] = torch.randn((1, self._horizon, self._noise.shape[-1]),
                                         device=self._dev, dtype=self._pdtype)
        poses = self._planner().float().cpu().squeeze(0)
        self._tick("planner", t0)
        return self._Trajectory(poses)

    def compute_trajectory(self, agent_input):
        """Drop-in for ReCogDriveAgent.compute_trajectory: the fast path where
        it applies, the agent itself otherwise."""
        trajectory = self.plan(agent_input)
        if trajectory is None:
            self.fallbacks += 1
            if self.fallbacks == 1:
                self._log("FastReCogDrive: input outside the fast path (tile grid or prompt "
                          "length); such frames use the reference path")
            trajectory = self.agent.compute_trajectory(agent_input)
        return trajectory

    def verify(self, agent_input, seed: int = 0) -> Dict[str, float]:
        """Plan from one input with the reference and with the fast path, same
        torch seed (so the same diffusion noise), and compare.

        hidden_max_abs    largest difference in the planner's VLM input
                          (0.0 = the VLM half is bit-identical)
        trajectory_max_abs  largest difference in the 8 poses (m / rad)

        The random generators are put back as they were found: an unseeded
        node stays unseeded."""
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        try:
            return self._verify(agent_input, seed)
        finally:
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])

    def _verify(self, agent_input, seed: int) -> Dict[str, float]:
        seen = {}
        head = self.agent.action_head
        original = head.get_action

        def spy(vl_features, *args, **kwargs):
            seen["vl"] = vl_features.detach().clone()
            return original(vl_features, *args, **kwargs)

        head.get_action = spy
        try:
            torch.manual_seed(seed)
            ref = self.agent.compute_trajectory(agent_input).poses
        finally:
            del head.get_action
        torch.manual_seed(seed)
        fast = self.plan(agent_input)
        if fast is None:
            return {"covered": 0.0}
        hidden = self.hidden_state()
        same_shape = hidden.shape == seen["vl"].shape
        return {
            "covered": 1.0,
            "hidden_max_abs": float((hidden - seen["vl"]).abs().max()) if same_shape else float("inf"),
            "trajectory_max_abs": float(np.abs(np.asarray(fast.poses, dtype=np.float64)
                                               - np.asarray(ref, dtype=np.float64)).max()),
        }
