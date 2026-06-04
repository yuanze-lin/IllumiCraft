import gc
import cv2
import numpy as np 
import inspect
from typing import Optional, Tuple, Union

import torch
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from pathlib import Path


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
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
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

    # Optimizer creation
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

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

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

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
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

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")


def _to_uint8_video(frames):
    """
    Convert video data into uint8 frames with shape [F, H, W, C].
    Accepts torch.Tensor / np.ndarray in layouts:
      - [1, C, F, H, W]
      - [C, F, H, W]
      - [F, C, H, W]
      - [F, H, W, C]
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()
    elif not isinstance(frames, np.ndarray):
        raise TypeError(f"Unsupported frames type: {type(frames)}")

    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]  # [C, F, H, W]

    if frames.ndim != 4:
        raise ValueError(f"Expected 4D or 5D video input, got shape {frames.shape}")

    # [C, F, H, W] -> [F, H, W, C]
    if frames.shape[0] in (1, 3):
        frames = np.transpose(frames, (1, 2, 3, 0))
    # [F, C, H, W] -> [F, H, W, C]
    elif frames.shape[1] in (1, 3):
        frames = np.transpose(frames, (0, 2, 3, 1))

    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = np.clip(frames, 0.0, 1.0)
            frames = (frames * 255.0).round().astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

    return frames


def save_side_by_side_video(
    foreground_frames,
    background_image,
    generated_video,
    output_path,
    fps=24,
):
    """
    Save a video concatenating:
      foreground | repeated background | output
    """
    fg = _to_uint8_video(foreground_frames)
    bg = _to_uint8_video(background_image)
    out = _to_uint8_video(generated_video)

    num_frames = out.shape[0]

    if bg.shape[0] == 1:
        bg = np.repeat(bg, num_frames, axis=0)
    elif bg.shape[0] != num_frames:
        bg = np.repeat(bg[:1], num_frames, axis=0)

    if fg.shape[0] != num_frames:
        fg = np.repeat(fg[:1], num_frames, axis=0)

    # Resize everything to the output resolution if needed.
    h, w = out.shape[1], out.shape[2]

    def _resize_seq(seq):
        resized = []
        for frame in seq:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        return np.stack(resized, axis=0)

    fg = _resize_seq(fg)
    bg = _resize_seq(bg)
    out = _resize_seq(out)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))

    for f, b, o in zip(fg, bg, out):
        concat = np.concatenate([f, b, o], axis=1)  # horizontal concat
        concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
        writer.write(concat_bgr)

    writer.release() 


def save_side_by_side_foreground_generated_video(
    foreground_frames,
    generated_video,
    output_path,
    fps=24
):
    fg = _to_uint8_video(foreground_frames)
    out = _to_uint8_video(generated_video)

    num_frames = min(fg.shape[0], out.shape[0])
    if num_frames == 0:
        raise ValueError("Cannot save an empty comparison video.")

    fg = fg[:num_frames]
    out = out[:num_frames]

    h, w = out.shape[1], out.shape[2]

    def _resize_seq(seq):
        resized = []
        for frame in seq:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        return np.stack(resized, axis=0)

    fg = _resize_seq(fg)
    out = _resize_seq(out)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    for f, o in zip(fg, out):
        concat = np.concatenate([f, o], axis=1)
        writer.write(cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

    writer.release()
