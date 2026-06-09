import gc
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import diffusers
import numpy as np
import torch
from accelerate import DistributedType, init_empty_weights
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
from models.illumicraft import WanImageToVideoPipelineTracking, WanTransformer3DModelTracking

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

logger = get_logger(__name__)

def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    """Build the optimizer selected by the training arguments."""
    optimizer_name = optimizer_name.lower()

    # DeepSpeed provides its own optimizer wrapper, so we return a dummy optimizer here.
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    # torchao is required for low-bit CPU offload and torchao-backed 4-bit optimizers.
    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Normalize unsupported optimizer names to AdamW so training can continue.
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    # Select the optimizer class and its constructor kwargs.
    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {"betas": (beta1, beta2), "eps": epsilon, "weight_decay": weight_decay}

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {"betas": (beta1, beta2), "eps": epsilon, "weight_decay": weight_decay}

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME
        init_kwargs = {"lr": learning_rate, "eps": (1e-30, 1e-16), "betas": (beta1, beta2, beta3), "weight_decay": weight_decay}

    # CPU offload wraps the underlying optimizer so gradients and states can live on host memory.
    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize,
            optimizer_class=optimizer_class,
            offload_gradients=offload_gradients,
            **init_kwargs,
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer

def get_gradient_norm(parameters):
    """Compute the global L2 norm of all available gradients."""
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    return norm**0.5

# This follows the same crop logic used by Diffusers-style video grid preparation.
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    """Return the centered crop region after resizing a source grid to the target aspect ratio."""
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create 3D rotary embeddings for the current video resolution and frame count."""
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    return freqs_cos.to(device=device), freqs_sin.to(device=device)

def reset_memory(device: Union[str, torch.device]) -> None:
    """Release cached GPU memory and reset CUDA peak statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

def print_memory(device: Union[str, torch.device]) -> None:
    """Print current and peak CUDA memory usage in gigabytes."""
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")

def _to_uint8_video(frames):
    """Convert a tensor or NumPy array into uint8 video frames with layout [F, H, W, C]."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu()
        if frames.ndim == 5:
            frames = frames[0]  # [C, F, H, W]
        if frames.ndim == 4 and frames.shape[0] in (1, 3):
            frames = frames.permute(1, 2, 3, 0)  # [F, H, W, C]
        frames = frames.float().numpy()
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[0] in (1, 3):
            frames = np.transpose(frames, (1, 2, 3, 0))
    else:
        raise TypeError(f"Unsupported frames type: {type(frames)}")

    # Normalize to [0, 255] uint8 when the input is floating-point video data.
    if frames.dtype != np.uint8:
        if frames.min() < 0.0 or frames.max() > 1.0:
            frames = frames * 0.5 + 0.5
        frames = np.clip(frames, 0.0, 1.0)
        frames = (frames * 255.0).round().astype(np.uint8)

    return frames

def _resize_frames_to_match(seq, height, width):
    """Resize each frame in a sequence to the requested spatial size."""
    resized = []
    for frame in seq:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        resized.append(frame)
    return np.stack(resized, axis=0)

def save_side_by_side_video(
    foreground_frames,
    background_image,
    generated_video,
    output_path,
    fps=24,
):
    """Save a three-panel comparison video: foreground, background, and generated output."""
    fg = _to_uint8_video(foreground_frames)
    bg = _to_uint8_video(background_image)
    out = _to_uint8_video(generated_video)

    num_frames = out.shape[0]

    # Repeat the background or foreground so every panel has the same frame count.
    if bg.shape[0] == 1:
        bg = np.repeat(bg, num_frames, axis=0)
    elif bg.shape[0] != num_frames:
        bg = np.repeat(bg[:1], num_frames, axis=0)

    if fg.shape[0] != num_frames:
        fg = np.repeat(fg[:1], num_frames, axis=0)

    # Resize everything to the output resolution if needed.
    h, w = out.shape[1], out.shape[2]
    fg = _resize_frames_to_match(fg, h, w)
    bg = _resize_frames_to_match(bg, h, w)
    out = _resize_frames_to_match(out, h, w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))

    for f, b, o in zip(fg, bg, out):
        concat = np.concatenate([f, b, o], axis=1)  # horizontal concat
        writer.write(cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

    writer.release()

def save_side_by_side_foreground_generated_video(
    foreground_frames,
    generated_video,
    output_path,
    fps=24,
):
    """Save a two-panel comparison video: foreground input and generated output."""
    fg = _to_uint8_video(foreground_frames)
    out = _to_uint8_video(generated_video)

    num_frames = min(fg.shape[0], out.shape[0])
    if num_frames == 0:
        raise ValueError("Cannot save an empty comparison video.")

    fg = fg[:num_frames]
    out = out[:num_frames]

    h, w = out.shape[1], out.shape[2]
    fg = _resize_frames_to_match(fg, h, w)
    out = _resize_frames_to_match(out, h, w)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    for f, o in zip(fg, out):
        writer.write(cv2.cvtColor(np.concatenate([f, o], axis=1), cv2.COLOR_RGB2BGR))

    writer.release()

def move_model(model, device, dtype):
    """Move a model to the target device, handling meta-initialized modules safely."""
    if any(param.device.type == "meta" for param in model.parameters()):
        model = model.to_empty(device=device)
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model

def unwrap_model(accelerator, model):
    """Return the underlying model, unwrapping Accelerate and compiled-module wrappers."""
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def _strip_prefix(state_dict, prefix="model."):
    """Remove a common prefix from all keys in a state dict when one is present."""
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def save_model_hook(
    accelerator,
    transformer,
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    models,
    weights,
    output_dir,
):
    """Save the training checkpoint in the layout expected by the downstream pipeline."""
    if not accelerator.is_main_process:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap all distributed/compiled wrappers before saving.
    transformer = unwrap_model(accelerator, transformer)
    vae = unwrap_model(accelerator, vae)
    text_encoder = unwrap_model(accelerator, text_encoder)
    clip_image_encoder = unwrap_model(accelerator, clip_image_encoder)

    transformer.save_pretrained(output_dir / "transformer", safe_serialization=True, max_shard_size="5GB")
    shutil.copyfile(output_dir / "transformer" / "config.json", output_dir / "config.json")

    # Save tokenizers under the same folder names expected by the model loader.
    tokenizer.save_pretrained(output_dir / "google" / "umt5-xxl")
    tokenizer.save_pretrained(output_dir / "xlm-roberta-large")

    # Save raw weights in the filenames expected by the existing checkpoint format.
    torch.save(text_encoder.state_dict(), output_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    
    vae_state = _strip_prefix(vae.state_dict(), prefix="model.")
    torch.save(vae_state, output_dir / "Wan2.1_VAE.pth")

    clip_state = _strip_prefix(clip_image_encoder.state_dict(), prefix="model.")
    torch.save(clip_state, output_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

    # Write small metadata files so the checkpoint can be reconstructed later.
    configuration_json = {
        "_class_name": "IllumiCraftCheckpoint",
        "_diffusers_version": diffusers.__version__,
    }
    with open(output_dir / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(configuration_json, f, indent=2)

    model_index_json = {
        "_class_name": "WanImageToVideoPipelineTracking",
        "_diffusers_version": diffusers.__version__,
        "transformer": ["transformer", "WanTransformer3DModelTracking"],
        "tokenizer": ["google/umt5-xxl", "AutoTokenizer"],
        "tokenizer_2": ["xlm-roberta-large", "AutoTokenizer"],
        "text_encoder": ["models_t5_umt5-xxl-enc-bf16.pth", "WanT5EncoderModel"],
        "vae": ["Wan2.1_VAE.pth", "AutoencoderKLWan"],
        "image_encoder": ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "CLIPModel"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index_json, f, indent=2)

    # Consume the list so Accelerate's hook contract is satisfied.
    while weights:
        weights.pop()

def load_model_hook(accelerator, transformer, args, config, load_dtype, models, input_dir):
    """Restore the transformer checkpoint and reattach its config."""
    transformer_ = None
    init_under_meta = False

    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                transformer_ = unwrap_model(accelerator, model)
            else:
                raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
    else:
        # DeepSpeed restores the transformer under init_empty_weights so the module can be materialized later.
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

def build_pipeline(
    args,
    accelerator,
    transformer,
    scheduler,
    val_scheduler,
    tokenizer,
    clip_image_encoder,
    weight_dtype,
    vae=None,
    text_encoder=None,
):
    """Build the inference pipeline used for validation and checkpoint sampling."""
    return WanImageToVideoPipelineTracking(
        vae=accelerator.unwrap_model(vae).to(weight_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
        scheduler=val_scheduler,
        clip_image_encoder=clip_image_encoder,
    )
