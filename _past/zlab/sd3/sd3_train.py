"""written by @anonymous"""

import tokenize
from .stability.other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from .stability.sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
from .stability.sd3_infer import SHIFT, load_into
from .prompts import _encode_prompt_with_clip, _encode_prompt_with_t5

import os
import copy
import wandb
import torch
from safetensors import safe_open
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, T5TokenizerFast

# load weights for modules related to CLIP
CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

class ClipG:
    def __init__(self, clipg_weights="models/clip_g.safetensors"):
        with safe_open(clipg_weights, framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)
            
CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

class ClipL:
    def __init__(self, clipl_weights):
        with safe_open(clipl_weights, framework="pt", device="cpu") as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

class T5XXL:
    def __init__(self, t5_weights="models/t5xxl.safetensors"):
        with safe_open(t5_weights, framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)

SD3_CONFIG = {
    "shift": 3,
}

class SD3:
    def __init__(self, sd3_weights, shift=SD3_CONFIG["shift"], verbose=False):
        with safe_open(sd3_weights, framework="pt", device="cpu") as f:
            # implementation of diffusion transformer is wrapped in BaseModel
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cpu",
                dtype=torch.float16,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)

class VAE:
    def __init__(self, vae_weights):
        # NOTE: this is a bit confusing, but the VAE weights are loaded from the same file as SD3
        with safe_open(vae_weights, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=torch.float16).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float16)

# NOTE: main class in this file
class SD3Agent:
    def print(self, txt):
        if self.verbose:
            print(txt)
    
    def load(
        self, 
        clipl_weights,
        clipg_weights,
        t5_weights,
        sd3_weights,
        vae_weights,
        verbose=False,
    ):
        # NOTE: sd3_weights and vae_weights are the same file
        self.verbose = verbose
        self.print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        # self.tokenizer = SD3Tokenizer()
        self.tokenizer_clip_l = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer",
            revision=None,
        )
        self.tokenizer_clip_g = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer_2",
            revision=None,
        )
        self.tokenizer_t5 = T5TokenizerFast.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer_3",
            revision=None,
        )
        self.print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL(clipl_weights)
        self.print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG(clipg_weights)
        self.print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL(t5_weights)
        self.print("Loading VAE model...")
        self.vae = VAE(vae_weights)
        self.print(f"Loading SD3 model {os.path.basename(sd3_weights)}...")
        self.sd3 = SD3(sd3_weights, verbose)
        self.print("Models loaded.")
