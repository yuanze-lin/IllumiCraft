import argparse
import os
import sys
import torch
import numpy as np
import random
import tempfile
import cv2
import decord  # isort:skip

from pathlib import Path
from typing import Any, Dict, Union, Optional

import torch.nn.functional as F

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
)
from diffusers.utils import export_to_video

from transformers import AutoTokenizer
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import resize

from omegaconf import OmegaConf
from PIL import Image

decord.bridge.set_bridge("torch")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
from models import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
)

from models.illumicraft import (
    WanImageToVideoPipelineTracking,
    WanTransformer3DModelTracking,
)


from training.utils import save_side_by_side_video, save_side_by_side_foreground_generated_video

def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def find_nearest_resolution(height_buckets, width_buckets, frame_buckets, height, width):
    resolutions = [(f, h, w) for h in height_buckets for w in width_buckets for f in frame_buckets]
    nearest_res = min(resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
    return nearest_res[1], nearest_res[2]


def scale_transform(x):
    return x / 255.0


def move_model(model, device, dtype):
    if any(param.device.type == "meta" for param in model.parameters()):
        model = model.to_empty(device=device)
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model


def prepare_frames(
    path,
    height_buckets,
    width_buckets,
    frame_buckets,
    image_transforms,
    num_frames=49,
    width=720,
    height=480,
):
    """Load a conditioning source (image or video) and return frames as [F, C, H, W]."""
    path = str(path)
    if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img = Image.open(path).convert("RGB")
        img = img.resize((width, height))
        first_frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()  # [C,H,W]
        zeros = torch.zeros((num_frames - 1, *first_frame.shape), dtype=first_frame.dtype)
        frames = torch.cat([first_frame.unsqueeze(0), zeros], dim=0)  # [F,C,H,W]
    else:
        reader = decord.VideoReader(uri=path)
        frame_indices = list(range(len(reader)))
        frames = reader.get_batch(frame_indices)  # [F,H,W,C]
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [F,C,H,W]

    nearest_res = find_nearest_resolution(
        height_buckets,
        width_buckets,
        frame_buckets,
        frames.shape[2],
        frames.shape[3],
    )
    frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
    frames = torch.stack([image_transforms(frame) for frame in frames_resized], dim=0)
    return frames

def generate_video(
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    pipe: Union[WanPipeline, WanImageToVideoPipelineTracking],
    dtype: torch.dtype = torch.bfloat16,
    fps: int = 24,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_transforms = transforms.Compose(
        [
            transforms.Lambda(scale_transform),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    pipe.transformer.eval()
    pipe.clip_image_encoder.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    for module in (pipe.transformer, pipe.clip_image_encoder, pipe.text_encoder, pipe.vae):
        for param in module.parameters():
            param.requires_grad = False

    pipe.transformer.gradient_checkpointing = False

    foreground_frames = prepare_frames(
        args.foreground_video_path,
        args.height_buckets,
        args.width_buckets,
        args.frame_buckets,
        image_transforms,
        num_frames=args.frame_buckets[0],
        width=args.width,
        height=args.height,
    )

    hdr_maps = None
    tracking_maps = None
    ref_image = None

    if args.hdr_path:
        hdr_maps = prepare_frames(
            args.hdr_path,
            args.height_buckets,
            args.width_buckets,
            args.frame_buckets,
            image_transforms,
            num_frames=args.frame_buckets[0],
            width=args.width,
            height=args.height,
        )

    if args.tracking_path:
        tracking_maps = prepare_frames(
            args.tracking_path,
            args.height_buckets,
            args.width_buckets,
            args.frame_buckets,
            image_transforms,
            num_frames=args.frame_buckets[0],
            width=args.width,
            height=args.height,
        )

    with torch.no_grad():
        foreground_frames = foreground_frames.unsqueeze(0).to(device=device, dtype=dtype)
        foreground_frames = foreground_frames.permute(0, 2, 1, 3, 4)  # [B,C,F,H,W]

        if hdr_maps is not None:
            hdr_maps = hdr_maps.unsqueeze(0).to(device=device, dtype=dtype)
            hdr_maps = hdr_maps.permute(0, 2, 1, 3, 4)
            n, c, d, h, w = hdr_maps.shape
            hdr_maps_flat = hdr_maps.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
            hdr_maps_flat = F.interpolate(hdr_maps_flat, size=(32, 32), mode="bilinear", align_corners=False)
            hdr_maps = hdr_maps_flat.view(n, d, c, 32, 32).permute(0, 2, 1, 3, 4)

        if tracking_maps is not None:
            tracking_maps = tracking_maps.unsqueeze(0).to(device=device, dtype=dtype)
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
            tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
            tracking_maps = tracking_latent_dist.sample().to(device=device, dtype=dtype)

    base_prompt = args.base_prompt.strip()
    prompt = None
    if args.lighting_prompt and args.lighting_prompt.strip():
        prompt = f"{base_prompt}, {args.lighting_prompt.strip()}"

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(args.foreground_video_path).stem

    # No-background generation.
    nobg_args = dict(pipeline_args)
    nobg_args["control_video"] = foreground_frames
    nobg_args["prompt"] = base_prompt
    if hdr_maps is not None:
        nobg_args["hdr_maps"] = hdr_maps
    if tracking_maps is not None:
        nobg_args["tracking_maps"] = tracking_maps

    nobg_path = output_root / f"{stem}_nobg.mp4"
    generated_video_nobg = pipe(
        **nobg_args,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="np",
    ).videos.numpy()[0]
    export_to_video(generated_video_nobg, str(nobg_path), fps=fps)
    save_side_by_side_foreground_generated_video(foreground_frames, generated_video_nobg, str(nobg_path).replace(".mp4", "_concat.mp4"), fps=fps)

    # Background-conditioned generation, only if background_path is provided.
    if args.background_path:
        ref_frames = prepare_frames(
            args.background_path,
            args.height_buckets,
            args.width_buckets,
            args.frame_buckets,
            image_transforms,
            num_frames=args.frame_buckets[0],
            width=args.width,
            height=args.height,
        )
        with torch.no_grad():
            ref_frames = ref_frames.unsqueeze(0).to(device=device, dtype=dtype)
            ref_frames = ref_frames.permute(0, 2, 1, 3, 4)
            ref_image = ref_frames[:, :, 0:1, :, :]

        bg_args = dict(pipeline_args)
        bg_args["control_video"] = foreground_frames
        bg_args["ref_image"] = ref_image
        bg_args["prompt"] = prompt if prompt is not None else base_prompt
        if hdr_maps is not None:
            bg_args["hdr_maps"] = hdr_maps
        if tracking_maps is not None:
            bg_args["tracking_maps"] = tracking_maps

        bg_path = output_root / f"{stem}_bg.mp4"
        generated_video_bg = pipe(
            **bg_args,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="np",
        ).videos.numpy()[0]
        export_to_video(generated_video_bg, str(bg_path), fps=fps)
        save_side_by_side_video(
            foreground_frames,
            ref_image,
            generated_video_bg,
            str(bg_path).replace(".mp4", "_concat.mp4"),
            fps=fps,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate a video using the Illumicraft pipeline")
    parser.add_argument("--config_path", type=str, required=True, help="The config of the model in training.")
    parser.add_argument(
        "--foreground_video_path",
        type=str,
        required=True,
        help="Path to the foreground video (or image).",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        required=True,
        help="Foreground prompt text.",
    )
    parser.add_argument(
        "--lighting_prompt",
        type=str,
        default=None,
        help="Optional lighting prompt for background-conditioned generation.",
    )
    parser.add_argument(
        "--background_path",
        type=str,
        default=None,
        help="Optional background image/video path for background-conditioned generation.",
    )
    parser.add_argument(
        "--tracking_path",
        type=str,
        default=None,
        help="Optional tracking map video path.",
    )
    parser.add_argument(
        "--hdr_path",
        type=str,
        default=None,
        help="Optional HDR/lighting map video path.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to Wan2.1-Fun-1.3B-Control.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to IllumiCraft checkpoint directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=int,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=int,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument("--frame_buckets", nargs="+", type=int, default=[49])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    config = OmegaConf.load(args.config_path)

    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    )

    transformer = WanTransformer3DModelTracking.from_pretrained(
        os.path.join(args.model_path, "transformer"),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    ).to(dtype)

    text_encoder.to(device, dtype=dtype)
    transformer = move_model(transformer, device, dtype)
    vae.to(device, dtype=dtype)
    clip_image_encoder.to(device, dtype=dtype)

    pipe = WanImageToVideoPipelineTracking(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipe.to(device, dtype=dtype)

    pipeline_args = {
        "negative_prompt": (
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion."
        ),
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "max_sequence_length": 512,
    }

    generate_video(
        args=args,
        pipeline_args=pipeline_args,
        pipe=pipe,
        dtype=dtype,
        fps=24,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
