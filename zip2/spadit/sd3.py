"""
TODO: Add a description here
"""

import argparse
import copy
import itertools
import logging
import math
import os
import random
from re import L
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from regex import F, P
from sympy import use
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from dataclasses import dataclass
from transformers import CLIPTextModelWithProjection
from transformers import T5EncoderModel
import bitsandbytes as bnb
from accelerate import init_empty_weights, infer_auto_device_map
from torch.distributed import init_process_group
from accelerate import load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download

from datas import DreamBoothDataModule


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


class SD3Model:
    def __init__(
        self, 
        pretrained_model_name_or_path,
        revision,
        variant,
        max_sequence_length=77,
        load=False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.revision = revision
        self.variant = variant
        self.max_sequence_length = max_sequence_length
        if load:
            # print("Loading pretrained model...")
            self.load_pretrained_tokenizers()
            self.load_pretrained_models()
            # print("Pretrained model loaded")
        
    def load_pretrained_tokenizers(self):        
        tokenizer_one = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision,
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.revision,
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=self.revision,
        )
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        self.tokenizers = tokenizers
        
    def load_pretrained_models(self):
        # Noise scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler",
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        # Text encoders
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, self.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, self.revision, subfolder="text_encoder_2"
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, self.revision, subfolder="text_encoder_3"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder", 
            revision=self.revision, variant=self.variant,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder_2", 
            revision=self.revision, variant=self.variant,
        )
        text_encoder_three = text_encoder_cls_three.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder_3", 
            revision=self.revision, variant=self.variant,
        )
        
        # VAE and Transformer
        vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision,
            variant=self.variant,
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.revision,
            variant=self.variant,
        )
        
        # Freeze the models by default
        vae.requires_grad_(False)
        transformer.requires_grad_(True)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
        
        # Log the models as attributes
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_copy = noise_scheduler_copy
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.text_encoder_three = text_encoder_three
        self.vae = vae
        self.transformer = transformer
        
        # print(f"Device of text encoder one: {self.text_encoder_one.device}")
        # print(f"Device of text encoder two: {self.text_encoder_two.device}")
        # print(f"Device of text encoder three: {self.text_encoder_three.device}")
        # print(f"Device of VAE: {self.vae.device}")
        # print(f"Device of transformer: {self.transformer.device}")
        # print("Models loaded")

    def set_text_encoder_training(self, activate=True):
        if activate:
            self.text_encoder_one.requires_grad_(True)
            self.text_encoder_two.requires_grad_(True)
            self.text_encoder_three.requires_grad_(True)
        else:
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            self.text_encoder_three.requires_grad_(False)

class AcceleratorConfig:
    def __init__(
        self,
        *,
        report_to=None,
        mixed_precision=None,
        output_dir=None,
        logging_dir=None,
        gradient_accumulation_steps=None,
        push_to_hub=False,
        hub_model_id=None,
    ):
        logging_dir = Path(output_dir, logging_dir)
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir, 
            logging_dir=self.logging_dir
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.report_to = report_to
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        
@dataclass
class TrainConfig:
    seed: int
    train_text_encoder: bool
    train_transformer: bool
    gradient_checkpointing: bool
    allow_tf32: bool = False
    scale_lr: bool = True
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    adam_weight_decay_text_encoder: float = 0.0
    text_encoder_lr: float = None
    optimizer: str = "adamw"
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    max_train_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    gradient_accumulation_steps: int = 1
    overrode_max_train_steps: bool = False
    num_train_epochs: int = 1
    weighting_scheme: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    precondition_outputs: int = 1
    max_grad_norm: float = 1.0
    
class SD3Trainer:
    def __init__(
        self,
        sd3: SD3Model,
        accelerator_config: AcceleratorConfig,
        train_config: TrainConfig,
        logger = None,
        optimizer = None,
    ):
        self.sd3 = sd3
        self.accelerator_config = accelerator_config
        self.setup_accelerator()
        self.train_config = train_config
    
    def setup_accelerator(self):
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.accelerator_config.gradient_accumulation_steps,
            mixed_precision=self.accelerator_config.mixed_precision,
            log_with=self.accelerator_config.report_to,
            project_config=self.accelerator_config.accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        # print(f"Accelerator: {accelerator}")
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.accelerator_config.output_dir is not None:
                os.makedirs(self.accelerator_config.output_dir, exist_ok=True)
            
            if self.accelerator_config.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.accelerator_config.hub_model_id or Path(self.accelerator_config.output_dir).name,
                    exist_ok=True,
                ).repo_id
        
        # Log the accelerator as an attribute
        self.accelerator = accelerator
        # print(f"Device of accelerator: {self.accelerator.device}")
    
    def train(self, data_module):        
        # Set seed
        if self.train_config.seed is not None:
            set_seed(self.train_config.seed)
        
        # Whether to train the text encoders and transformer
        self.set_trainable_sd3_modules()
        
        # Move to device and set dtype and enable gradient checkpointing
        self.set_sd3_modules_device_and_dtype()
        
        # Enable TF32 for faster training on Ampere GPUs
        self.enable_tf32()
        
        # Scale learning rate
        self.scale_learning_rate()
        
        # Create an optimizer
        self.create_optimizer()
    
        # Log data
        self.prepare_data(data_module)  # Get `self.train_loader`
        
        # If text encoders are not trained
        if not self.train_config.train_text_encoder:
            self.handle_not_train_text_encoder()
        
        # Get scheduler
        self.lr_scheduler = self.get_scheduler_from_optimizer(self.optimizer)
        
        # Prepare everything with the accelerator
        self.accelerator_prepare()
        
        # Calculate number of training epochs
        self.calculate_num_training_epochs()
        
        # Fit the model
        self.fit()
    
    def fit(self):
        train_batch_size = self.train_config.train_batch_size
        num_processes = self.accelerator.num_processes
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps
        global_step = 0
        first_epoch = 0
        
        initial_global_step = 0
        for epoch in range(first_epoch, self.train_config.num_train_epochs):
            if self.train_config.train_transformer:
                self.sd3.transformer.train()
            if self.train_config.train_text_encoder:
                self.sd3.text_encoder_one.train()
                self.sd3.text_encoder_two.train()
                self.sd3.text_encoder_three.train()

            for step, batch in enumerate(self.train_dataloader):
                if self.train_config.train_transformer:
                    models_to_accumulate = [self.sd3.transformer]
                if self.train_config.train_text_encoder:
                    models_to_accumulate.extend([self.sd3.text_encoder_one, self.sd3.text_encoder_two, self.sd3.text_encoder_three])
                with self.accelerator.accumulate(models_to_accumulate):
                    pixel_values = batch["pixel_values"].to(dtype=self.sd3.vae.dtype)
                    prompts = batch["prompts"]
    
                    # Convert images to latent space
                    model_input = self.sd3.vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - self.sd3.vae.config.shift_factor) * self.sd3.vae.config.scaling_factor
                    model_input = model_input.to(dtype=self.weight_dtype)
                        
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    
                    # Sample a random timestep for each image
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=self.train_config.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=self.train_config.logit_mean,
                        logit_std=self.train_config.logit_std,
                        mode_scale=self.train_config.mode_scale,
                    )
                    indices = (u * self.sd3.noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = self.sd3.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching
                    sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    # Encode batch prompts when custom prompts are provided for each image
                    if not self.train_config.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(
                            prompts, 
                            [self.sd3.text_encoder_one, self.sd3.text_encoder_two, self.sd3.text_encoder_three],
                            self.sd3.tokenizers,
                        )
                    else:
                        return 0
                    
                    # Predict the noise residual
                    print(
                        f"Data type of `noisy_model_input`: {noisy_model_input.dtype}",
                        f"Data type of `timesteps`: {timesteps.dtype}",
                        f"Data type of `prompt_embeds`: {prompt_embeds.dtype}",
                        f"Data type of `pooled_prompt_embeds`: {pooled_prompt_embeds.dtype}",
                        sep="\n",
                    )
                    model_pred = self.sd3.transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    print(f"Data type of `model_pred`: {model_pred.dtype}")
                    
                    # Follow section 5 of https://arxiv.org/abs/2206.00364
                    # Preconditioning of the model outputs
                    if self.train_config.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_model_input
                    
                    # Compute loss weightting for SD3
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.train_config.weighting_scheme, sigmas=sigmas)
                    
                    # Flow matching loss
                    if self.train_config.precondition_outputs:
                        target = model_input
                    else:
                        target = noise - model_input

                    # Compute regular loss
                    print(
                        f"Data type of `weighting`: {weighting.dtype}",
                        f"Data type `model_pred`: {model_pred.dtype}",
                        f"Data type `target`: {target.dtype}",
                        sep="\n",
                    )
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()
                    print(f"Dtype of `loss`: {loss.dtype}")
                    print(f"Loss: {loss}")
                    
                    # Backpropagate
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.sd3.transformer.parameters(),
                                self.sd3.text_encoder_one.parameters(),
                                self.sd3.text_encoder_two.parameters(),
                                self.sd3.text_encoder_three.parameters(),
                            )
                            if self.train_config.train_text_encoder else self.sd3.transformer.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.train_config.max_grad_norm)
                    
                    # Update the model
                    assert torch.isfinite(loss).all(), "Loss contains NaN or Inf values."
                    assert hasattr(self.accelerator, "scaler"), "GradScaler is not initialized."
                    for name, param in self.sd3.transformer.named_parameters():
                        if param.grad is None:
                            print(f"Gradient for {name} is None.")
                            break
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                break
            break
        
    def compute_text_embeddings(self, prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, self.sd3.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.sd3.noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.sd3.noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    def calculate_num_training_epochs(self):
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.train_config.gradient_accumulation_steps)
        if self.train_config.overrode_max_train_steps:
            self.train_config.max_train_steps = self.train_config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.train_config.num_train_epochs = math.ceil(self.train_config.max_train_steps / num_update_steps_per_epoch)
    
    def accelerator_prepare(self):
        if self.train_config.train_text_encoder and self.train_config.train_transformer:
            (
                self.sd3.transformer,
                self.sd3.text_encoder_one,
                self.sd3.text_encoder_two,
                self.sd3.text_encoder_three,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.sd3.transformer,
                self.sd3.text_encoder_one,
                self.sd3.text_encoder_two,
                self.sd3.text_encoder_three,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
        elif not self.train_config.train_text_encoder and self.train_config.train_transformer:
            (
                self.sd3.transformer,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.sd3.transformer, 
                self.optimizer, 
                self.train_dataloader, 
                self.lr_scheduler
            )
        elif self.train_config.train_text_encoder and not self.train_config.train_transformer:
            (
                self.sd3.text_encoder_one,
                self.sd3.text_encoder_two,
                self.sd3.text_encoder_three,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.sd3.text_encoder_one,
                self.sd3.text_encoder_two,
                self.sd3.text_encoder_three,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
        else:
            raise ValueError("Both text encoders and transformer cannot be frozen.")
    
    def get_scheduler_from_optimizer(self, optimizer):        
        lr_scheduler = get_scheduler(
            self.train_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.train_config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.train_config.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.train_config.lr_num_cycles,
            power=self.train_config.lr_power,
        )
        return lr_scheduler
    
    # TODO: Implement this method
    def handle_not_train_text_encoder(self):
        # Line 1370 to 1428
        pass
    
    def prepare_data(self, data_module: DreamBoothDataModule):
        self.data_module = data_module
        self.train_dataloader = data_module.get_train_dataloader()
        
    def create_optimizer(self):
        # Get optimizer parameters
        params_to_optimize = []
        
        # If transformer is trainable
        if self.train_config.train_transformer:
            # print("Transformer is going to be trained")
            transformer_parameters_with_lr = {
                "params": self.sd3.transformer.parameters(), 
                "lr": self.train_config.learning_rate,
            }
            params_to_optimize.append(transformer_parameters_with_lr)
        
        # If text encoders are trainable
        if self.train_config.train_text_encoder:
            # print("Text encoders are going to be trained")
            weight_decay_text_encoder = self.train_config.adam_weight_decay_text_encoder
            lr_text_encoder = self.train_config.text_encoder_lr if self.train_config.text_encoder_lr else self.train_config.learning_rate
            text_parameters_one_with_lr = {
                "params": self.sd3.text_encoder_one.parameters(),
                "weight_decay": weight_decay_text_encoder,
                "lr": lr_text_encoder,
            }
            text_parameters_two_with_lr = {
                "params": self.sd3.text_encoder_two.parameters(),
                "weight_decay": weight_decay_text_encoder,
                "lr": lr_text_encoder,
            }
            text_parameters_three_with_lr = {
                "params": self.sd3.text_encoder_three.parameters(),
                "weight_decay": weight_decay_text_encoder,
                "lr": lr_text_encoder,
            }
            params_to_optimize.append(text_parameters_one_with_lr)
            params_to_optimize.append(text_parameters_two_with_lr)
            params_to_optimize.append(text_parameters_three_with_lr)
        
        num_params = 0
        for param_group in params_to_optimize:
            params = param_group["params"]
            num_params += sum(p.numel() for p in params)
        print(f"Total number of parameters: {num_params}")
        
        if len(params_to_optimize) == 0:
            print("No parameters to optimize")
            return None
        
        if self.train_config.optimizer == "adamw":
            if self.train_config.use_8bit_adam:
                optimizer_class = bnb.optim.AdamW8bit
            else:
                optimizer_class = torch.optim.AdamW
            
        self.optimizer = optimizer_class(
            params_to_optimize,
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
            weight_decay=self.train_config.adam_weight_decay,
            eps=self.train_config.adam_epsilon,
        )
    
    def enable_tf32(self):
        if self.train_config.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def scale_learning_rate(self):
        if self.train_config.scale_lr:
            learning_rate = self.train_config.learning_rate
            gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
            train_batch_size = self.train_config.train_batch_size
            num_processes = self.accelerator.num_processes
            self.train_config.learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * num_processes
            )
            # print(f"Learning rate scaled to {self.train_config.learning_rate}")
            
    def set_sd3_modules_device_and_dtype(self):
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        self.sd3.vae.to(self.accelerator.device, dtype=torch.float32)
        self.sd3.transformer.to(self.accelerator.device, dtype=torch.float32)
        if not self.train_config.train_text_encoder:
            self.sd3.text_encoder_one.to(self.accelerator.device, dtype=weight_dtype)
            self.sd3.text_encoder_two.to(self.accelerator.device, dtype=weight_dtype)
            self.sd3.text_encoder_three.to(self.accelerator.device, dtype=weight_dtype)

        if self.train_config.gradient_checkpointing:
            if self.train_config.train_transformer:
                self.sd3.transformer.enable_gradient_checkpointing()
            if self.train_config.train_text_encoder:
                self.sd3.text_encoder_one.gradient_checkpointing_enable()
                self.sd3.text_encoder_two.gradient_checkpointing_enable()
                self.sd3.text_encoder_three.gradient_checkpointing_enable()
    
    def set_trainable_sd3_modules(self):
        if self.train_config.train_text_encoder:
            self.sd3.set_text_encoder_training(activate=True)
        else:
            self.sd3.set_text_encoder_training(activate=False)

    # TODO: Implement this method
    def with_prior_preservation(self):
        # Line 1040 to 1084
        pass

    # TODO: Implement this method
    def register_hooks(self):
        # Line 1181 to 1241
        pass


class SD3Evaluator:
    pass


def _test():
    # Model
    args = argparse.Namespace(
        # model
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium",
        revision=None,
        variant=None,
        load=True,
    )
    sd3 = SD3Model(
        args.pretrained_model_name_or_path, args.revision, args.variant,
        load=args.load
    )
    
    # Accelerator
    args = argparse.Namespace(
        # training
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        report_to="wandb",
        logging_dir="logs",
        output_dir="output",
    )
    accelerator_config = AcceleratorConfig(
        **vars(args)
    )
    
    # Trainer
    args = argparse.Namespace(
        seed=1999, train_text_encoder=False, train_transformer=True,
        gradient_checkpointing=False,
    )
    train_config = TrainConfig(
        **vars(args)
    )
    trainer = SD3Trainer(
        sd3,
        accelerator_config,
        train_config,
    )
    
    # Data and training
    args = argparse.Namespace(
        # dataset
        instance_data_dir="/scratch0/zy45/work/code/diffusers/zip2/data/dog",
        instance_prompt="a photo of a dog",
        class_prompt=None,
        train_batch_size=1,
        dataloader_num_workers=0,
        with_prior_preservation=False,
    )
    data_module = DreamBoothDataModule(
        **vars(args)
    )
    trainer.train(data_module)

if __name__ == "__main__":
    print("Running test...")
    _test()