# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import logging
import math
import os
import shutil
import torchvision.transforms.functional as TF
import sys
import numpy as np
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Union, Optional

import diffusers
import torch
import transformers
import copy
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer, T5EncoderModel

import decord  # isort:skip
decord.bridge.set_bridge("torch")

from args import get_args  # isort:skip
from dataset import BucketSampler, SimpleBatchSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop, VideoDatasetWithResizingTracking  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import get_gradient_norm, get_optimizer, prepare_rotary_positional_embeddings, print_memory, reset_memory  # isort:skip

from diffusers.utils import load_image, load_video
from omegaconf import OmegaConf
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from diffusers import WanPipeline
from models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel, CLIPModel, AutoencoderKLWan
from models.wan_transformer3d import WanTransformer3DModel
from models.illumicraft import WanImageToVideoPipelineTracking, WanTransformer3DModelTracking
from models.discrete_sampler import DiscreteSampling
from models.pipeline_wan_fun_control import WanFunControlPipeline
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    WanPipeline)
from diffusers.training_utils import compute_loss_weighting_for_sd3
import pdb
import torch.nn.functional as F
logger = get_logger(__name__)
import os
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["TORCH_LOGS"] = ""

def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def log_validation(
    accelerator: Accelerator,
    pipe: Union[WanPipeline, WanImageToVideoPipelineTracking],
    vae: Union[AutoencoderKLWan, None],
    dataset: Union[VideoDatasetWithResizingTracking, None],
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    epoch,
    is_final_validation: bool = False,
    random_flip: Optional[float] = None,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    foreground_map_path = pipeline_args.pop("foreground_map_path", None)
    background_map_path = pipeline_args.pop("background_map_path", None)
    hdr_map_path = pipeline_args.pop("hdr_map_path", None)
    tracking_map_path = pipeline_args.pop("tracking_map_path", None)

    foreground_frames = None
    background_frames = None
    hdr_maps = None
    tracking_maps = None

    if args.load_tensors is False:
        from torchvision import transforms
        from torchvision.transforms.functional import resize

        video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(dataset.identity_transform),
                transforms.Lambda(dataset.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        if hdr_map_path:
            hdr_map_path = Path(hdr_map_path)
            hdr_reader = decord.VideoReader(uri=hdr_map_path.as_posix())
            frame_indices = list(range(0, len(hdr_reader)))
            hdr_frames = hdr_reader.get_batch(frame_indices)
            nearest_res = dataset._find_nearest_resolution(hdr_frames.shape[2], hdr_frames.shape[3])
            hdr_frames = hdr_frames.permute(0, 3, 1, 2).contiguous()
            hdr_frames_resized = torch.stack([resize(hdr_frame, nearest_res) for hdr_frame in hdr_frames], dim=0)
            hdr_maps = torch.stack([video_transforms(hdr_frame) for hdr_frame in hdr_frames_resized], dim=0)

        if tracking_map_path:
            tracking_map_path = Path(tracking_map_path)
            tracking_reader = decord.VideoReader(uri=tracking_map_path.as_posix())
            frame_indices = list(range(0, len(tracking_reader)))
            tracking_frames = tracking_reader.get_batch(frame_indices)
            nearest_res = dataset._find_nearest_resolution(tracking_frames.shape[2], tracking_frames.shape[3])
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_maps = torch.stack([video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

        foreground_map_path = Path(foreground_map_path)
        foreground_reader = decord.VideoReader(uri=foreground_map_path.as_posix())
        frame_indices = list(range(0, len(foreground_reader)))
        foreground_frames = foreground_reader.get_batch(frame_indices)
        nearest_res = dataset._find_nearest_resolution(foreground_frames.shape[2], foreground_frames.shape[3])
        foreground_frames = foreground_frames.permute(0, 3, 1, 2).contiguous()
        foreground_frames_resized = torch.stack([resize(foreground_frame, nearest_res) for foreground_frame in foreground_frames], dim=0)
        foreground_frames = torch.stack([video_transforms(foreground_frame) for foreground_frame in foreground_frames_resized], dim=0)

        if background_map_path is not None:
            background_map_path = Path(background_map_path)
            background_reader = decord.VideoReader(uri=background_map_path.as_posix())
            tmp_num_frames = len(background_reader)
            frame_indices = [0]
            first_frame_batch = background_reader.get_batch(frame_indices)
            _, H, W, C = first_frame_batch.shape
            zeros = torch.zeros((tmp_num_frames - 1, H, W, C), dtype=first_frame_batch.dtype, device=first_frame_batch.device)
            background_frames = torch.cat([first_frame_batch, zeros], dim=0)
            nearest_res = dataset._find_nearest_resolution(background_frames.shape[2], background_frames.shape[3])
            background_frames = background_frames.permute(0, 3, 1, 2).contiguous()
            background_frames_resized = torch.stack([resize(background_frame, nearest_res) for background_frame in background_frames], dim=0)
            background_frames = torch.stack([video_transforms(background_frame) for background_frame in background_frames_resized], dim=0)

        with torch.no_grad():
            if hdr_map_path:
                hdr_maps = hdr_maps.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
                hdr_maps = hdr_maps.permute(0, 2, 1, 3, 4)
                n, c, d, h, w = hdr_maps.shape
                hdr_maps_flat = hdr_maps.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
                hdr_maps_flat = F.interpolate(hdr_maps_flat, size=(32, 32), mode="bilinear", align_corners=False)
                hdr_maps = hdr_maps_flat.view(n, d, c, 32, 32).permute(0, 2, 1, 3, 4)

            if tracking_map_path:
                tracking_maps = tracking_maps.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
                tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
                tracking_latent_dist = vae.encode(tracking_maps).latent_dist
                tracking_maps = tracking_latent_dist.sample().to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)

            foreground_frames = foreground_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
            foreground_frames = foreground_frames.permute(0, 2, 1, 3, 4)

            if background_map_path is not None:
                background_frames = background_frames.unsqueeze(0).to(device=accelerator.device, dtype=accelerator.unwrap_model(vae).dtype)
                background_frames = background_frames.permute(0, 2, 1, 3, 4)

    pipe = pipe.to(accelerator.device)

    pipeline_args["hdr_maps"] = hdr_maps
    pipeline_args["tracking_maps"] = tracking_maps
    pipeline_args["control_video"] = foreground_frames
    pipeline_args["ref_image"] = background_frames

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    if args.num_validation_videos > 2:
        args.num_validation_videos = 2

    videos = []
    for _ in range(args.num_validation_videos):
        with torch.no_grad():
            video = pipe(**pipeline_args, generator=generator, output_type="np").videos.numpy()[0]
        videos.append(video)

    for _tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        for i, video in enumerate(videos):
            prompt = (
                pipeline_args["prompt"][:25]
                .replace(" ", "_")
                .replace("'", "_")
                .replace('"', "_")
                .replace("/", "_")
            )
            filename = os.path.join(args.output_dir, f"{phase_name}_ep{epoch}_{i}th_{prompt}.mp4")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            export_to_video(video, filename, fps=24)

    torch.cuda.empty_cache()
    return videos


class CollateFunctionImageTracking:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [sample["prompt"] for sample in data]
        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [sample["image"] for sample in data]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [sample["video"] for sample in data]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        foreground_maps = [sample["foreground_map"] for sample in data]
        foreground_maps = torch.stack(foreground_maps).to(dtype=self.weight_dtype, non_blocking=True)

        background_maps = [sample["background_map"] for sample in data]
        background_maps = torch.stack(background_maps).to(dtype=self.weight_dtype, non_blocking=True)

        hdr_maps = [sample["hdr_map"] for sample in data]
        hdr_maps = torch.stack(hdr_maps).to(dtype=self.weight_dtype, non_blocking=True)

        tracking_maps = [sample["tracking_map"] for sample in data]
        tracking_maps = torch.stack(tracking_maps).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "prompts": prompts,
            "images": images,
            "videos": videos,
            "foreground_maps": foreground_maps,
            "background_maps": background_maps,
            "hdr_maps": hdr_maps,
            "tracking_maps": tracking_maps,
        }


def move_model(model, device, dtype):
    if any(param.device.type == "meta" for param in model.parameters()):
        model = model.to_empty(device=device)
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model


def main(args):
    if args.hub_token is not None:
        raise ValueError("This training script does not support hub push. Please unset --hub_token.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    if args.config_path:
        config = OmegaConf.load(args.config_path)
    else:
        raise ValueError("config_path is required")

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    val_scheduler = copy.deepcopy(scheduler)

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=load_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        ),
    )

    transformer = WanTransformer3DModelTracking.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    ).to(load_dtype)

    clip_image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)

    weight_dtype = torch.bfloat16
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer = move_model(transformer, accelerator.device, weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = WanTransformer3DModelTracking.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        load_model = WanTransformer3DModelTracking.from_pretrained(
            input_dir,
            subfolder="transformer",
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        ).to(load_dtype)
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    def load_model_hook_tracking(models, input_dir):
        transformer_ = None
        init_under_meta = False

        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = WanTransformer3DModelTracking.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                )
                init_under_meta = True

        load_model = WanTransformer3DModelTracking.from_pretrained(
            input_dir,
            subfolder="transformer",
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        ).to(load_dtype)
        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook if args.resume_from_checkpoint is None else load_model_hook_tracking)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = [p for p in transformer.parameters() if p.requires_grad]
    trainable_param_names = {name for name, param in transformer.named_parameters() if param.requires_grad}
    print(len(trainable_param_names), trainable_param_names)

    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "foreground_column": args.foreground_column,
        "background_column": args.background_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": args.load_tensors,
        "random_flip": args.random_flip,
        "image_to_video": True,
    }
    if args.hdr_column is not None:
        dataset_init_kwargs["hdr_column"] = args.hdr_column
        dataset_init_kwargs["tracking_column"] = args.tracking_column
        train_dataset = VideoDatasetWithResizingTracking(**dataset_init_kwargs)
    elif args.video_reshape_mode is None:
        train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        train_dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode,
            **dataset_init_kwargs,
        )

    collate_fn_image_tracking = CollateFunctionImageTracking(weight_dtype, args.load_tensors)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=SimpleBatchSampler(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
        ),
        collate_fn=collate_fn_image_tracking,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "Wan-sft"
        accelerator.init_trackers(tracker_name, config=vars(args))
        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if args.load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    idx_sampling = DiscreteSampling(args.max_train_steps, uniform_sampling=False)
    rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
    torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)

    clip_index = 0
    max_sequence_length = 512
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            gradient_norm_before_clip = None
            gradient_norm_after_clip = None

            with accelerator.accumulate(models_to_accumulate):
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]

                if args.hdr_column is not None:
                    hdr_maps = batch["hdr_maps"].to(accelerator.device, non_blocking=True)

                if args.tracking_column is not None:
                    tracking_maps = batch["tracking_maps"].to(accelerator.device, non_blocking=True)

                if args.foreground_column is not None:
                    foreground_maps = batch["foreground_maps"].to(accelerator.device, non_blocking=True)

                if args.background_column is not None:
                    background_maps = batch["background_maps"].to(accelerator.device, non_blocking=True)

                with torch.no_grad():
                    videos = videos.permute(0, 2, 1, 3, 4)
                    latent_dist = vae.encode(videos).latent_dist

                    if args.hdr_column is not None:
                        hdr_maps = hdr_maps.permute(0, 2, 1, 3, 4)
                        n, c, d, h, w = hdr_maps.shape
                        hdr_maps_flat = hdr_maps.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
                        hdr_maps_flat = F.interpolate(hdr_maps_flat, size=(32, 32), mode="bilinear", align_corners=False)
                        hdr_maps = hdr_maps_flat.view(n, d, c, 32, 32).permute(0, 2, 1, 3, 4)

                    if args.tracking_column is not None:
                        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
                        tracking_latent_dist = vae.encode(tracking_maps).latent_dist
                        tracking_maps = tracking_latent_dist.sample().to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    if args.foreground_column is not None:
                        foreground_maps = foreground_maps.permute(0, 2, 1, 3, 4)
                        foreground_latent_dist = vae.encode(foreground_maps).latent_dist
                        control_latents = foreground_latent_dist.sample().to(
                            memory_format=torch.contiguous_format,
                            dtype=weight_dtype,
                        )

                    if args.background_column is not None:
                        background_maps = background_maps.permute(0, 2, 1, 3, 4)
                        background_maps[:, :, 1:, :, :] *= 0
                        background_latent_dist = vae.encode(background_maps).latent_dist
                        background_control_latents = background_latent_dist.sample().to(
                            memory_format=torch.contiguous_format,
                            dtype=weight_dtype,
                        )

                    clip_context = []
                    ref_pixel_values = []
                    for i in range(videos.shape[0]):
                        ref_pixel_values.append(background_control_latents[i, :, clip_index, :, :].contiguous().unsqueeze(0))
                        frame = videos[i, :, clip_index, :, :].permute(1, 2, 0).contiguous()
                        frame = ((frame * 0.5 + 0.5) * 255)
                        clip_image = Image.fromarray(np.uint8(frame.float().cpu().numpy()))
                        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(accelerator.device, weight_dtype)
                        _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                        zero_init_clip_in = rng.choice([True, False], p=[0.1, 0.9])
                        clip_context.append(_clip_context if not zero_init_clip_in else torch.zeros_like(_clip_context))

                    clip_context = torch.cat(clip_context)
                    video_latents = latent_dist.sample().to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    for i in range(len(ref_pixel_values)):
                        ref_pixel_values[i] = ref_pixel_values[i].unsqueeze(0)

                    ref_pixel_values = torch.cat(ref_pixel_values).permute(0, 2, 1, 3, 4).to(accelerator.device, weight_dtype)
                    ref_latents_conv_in = torch.zeros_like(video_latents).to(video_latents.device, video_latents.dtype)
                    ref_latents_conv_in[:, :, :1] = ref_pixel_values
                    for bs_index in range(ref_latents_conv_in.size()[0]):
                        zero_init_ref_latents_conv_in = rng.choice([0, 1], p=[0.90, 0.10])
                        if zero_init_ref_latents_conv_in and control_latents.size()[1] != 1:
                            ref_latents_conv_in[bs_index, :, :1] = ref_latents_conv_in[bs_index, :, :1] * 0
                    control_latents = torch.cat([control_latents, ref_latents_conv_in], dim=1)

                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        max_sequence_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                indices = idx_sampling(batch_size, generator=torch_rng, device=accelerator.device).long().cpu()
                timesteps = scheduler.timesteps[indices].to(device=accelerator.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    matches = schedule_timesteps[:, None] == timesteps[None, :]
                    step_indices = matches.float().argmax(dim=0).long()
                    sigma = sigmas[step_indices]
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                try:
                    sigmas = get_sigmas(timesteps, n_dim=video_latents.ndim, dtype=video_latents.dtype)
                except Exception as e:
                    print("get_sigmas failed:", repr(e))
                    print("indices min/max:", indices.min().item(), indices.max().item())
                    print("len timesteps:", len(scheduler.timesteps), "len sigmas:", len(scheduler.sigmas))
                    raise

                noisy_latents = (1.0 - sigmas) * video_latents + sigmas * noise
                target = noise - video_latents

                target_shape = (vae.config.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3])
                    / (
                        accelerator.unwrap_model(transformer).config.patch_size[1]
                        * accelerator.unwrap_model(transformer).config.patch_size[2]
                    )
                    * target_shape[1]
                )

                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction="none")
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    return masked_loss.mean()

                model_output = transformer(
                    x=noisy_latents,
                    context=prompt_embeds,
                    hdr_maps=hdr_maps,
                    tracking_maps=tracking_maps,
                    t=timesteps,
                    seq_len=seq_len,
                    y=control_latents,
                    clip_fea=clip_context,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
                loss = custom_mse_loss(model_output.float(), target.float(), weighting.float())
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()

                del videos, latent_dist, noise, timesteps, noisy_latents, prompt_embeds, model_output, target

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for removing_checkpoint in checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs = {"loss": loss.detach().item(), "lr": last_lr}
            if accelerator.distributed_type != DistributedType.DEEPSPEED:
                logs.update(
                    {
                        "gradient_norm_before_clip": gradient_norm_before_clip,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                    }
                )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and global_step % args.checkpointing_steps == 0:
                accelerator.print("===== Memory before validation =====")
                print_memory(accelerator.device)
                torch.cuda.synchronize(accelerator.device)
                transformer.eval()

                if args.hdr_column is None:
                    pipe = WanImageToVideoPipelineTracking.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrap_model(transformer),
                        scheduler=scheduler,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                else:
                    pipe = WanImageToVideoPipelineTracking(
                        vae=accelerator.unwrap_model(vae).to(weight_dtype),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        transformer=accelerator.unwrap_model(transformer),
                        scheduler=val_scheduler,
                        clip_image_encoder=clip_image_encoder,
                    )

                if args.enable_model_cpu_offload:
                    pipe.enable_model_cpu_offload()

                validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                validation_videos = args.validation_images.split(args.validation_prompt_separator)
                validation_backgrounds = args.validation_backgrounds.split(args.validation_prompt_separator)

                for validation_video, validation_background, validation_prompt in zip(
                    validation_videos, validation_backgrounds, validation_prompts
                ):
                    pipeline_args = {
                        "foreground_map_path": validation_video,
                        "background_map_path": validation_background,
                        "prompt": validation_prompt,
                        "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                        "guidance_scale": args.guidance_scale,
                        "height": args.height,
                        "width": args.width,
                        "max_sequence_length": 512,
                    }

                    if args.hdr_column is not None:
                        pipeline_args["hdr_map_path"] = args.hdr_map_path

                    if args.tracking_column is not None:
                        pipeline_args["tracking_map_path"] = args.tracking_map_path

                    log_validation(
                        accelerator=accelerator,
                        pipe=pipe,
                        vae=vae,
                        dataset=train_dataset,
                        args=args,
                        pipeline_args=pipeline_args,
                        epoch=(epoch + 1),
                        is_final_validation=False,
                    )

                transformer.train()
                accelerator.print("===== Memory after validation =====")
                print_memory(accelerator.device)
                reset_memory(accelerator.device)

                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(accelerator.device)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(transformer).to(
            dtype=(
                torch.float16
                if args.mixed_precision == "fp16"
                else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
            )
        )

        if args.hdr_column is None:
            pipe = WanImageToVideoPipelineTracking.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        else:
            pipe = WanImageToVideoPipelineTracking(
                vae=accelerator.unwrap_model(vae).to(weight_dtype),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=accelerator.unwrap_model(transformer),
                scheduler=val_scheduler,
                clip_image_encoder=clip_image_encoder,
            )

        pipe.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )

        if args.load_tensors:
            del pipe
        else:
            del text_encoder, vae, pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
