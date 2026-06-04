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
    """Keep only keyword arguments accepted by a class constructor."""
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def find_nearest_resolution(height_buckets, width_buckets, frame_buckets, height, width):
    """Pick the closest bucketed resolution for an input frame size."""
    resolutions = [(f, h, w) for h in height_buckets for w in width_buckets for f in frame_buckets]
    nearest_res = min(resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
    return nearest_res[1], nearest_res[2]


def scale_transform(x):
    """Scale an image tensor from [0, 255] to [0, 1]."""
    return x / 255.0


def file_lines_to_list(file_path):
    """Read a newline-separated text file into a list of strings."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def move_model(model, device, dtype):
    """Move a model to the target device, including meta-initialized modules."""
    if any(param.device.type == "meta" for param in model.parameters()):
        model = model.to_empty(device=device)
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model


def _load_video_frames(path, height_buckets, width_buckets, frame_buckets, image_transforms):
    """Load, resize, and normalize all frames from a video file."""
    video_reader = decord.VideoReader(uri=Path(path).as_posix())
    frame_indices = list(range(0, len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    nearest_res = find_nearest_resolution(
        height_buckets,
        width_buckets,
        frame_buckets,
        frames.shape[1],
        frames.shape[2],
    )
    frames = frames.permute(0, 3, 1, 2).contiguous()
    frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
    frames = torch.stack([image_transforms(frame) for frame in frames_resized], dim=0)
    return frames

def prepare_frames(path, height_buckets, width_buckets, frame_buckets, image_transforms, num_frames=49, width=720, height=480):
    """
    Load a conditioning source (image or video) and return frames as:
        [F, C, H, W]
    """
    path = str(path)
    if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img = Image.open(path).convert("RGB")
        img = img.resize((width, height))

        first_frame = torch.from_numpy(np.array(img))  # [H,W,C]
        zeros = torch.zeros((num_frames - 1, *first_frame.shape),dtype=first_frame.dtype)
        frames = torch.cat([first_frame.unsqueeze(0), zeros],dim=0)  # [F,H,W,C]
    else:
        reader = decord.VideoReader(uri=path)
        frame_indices = list(range(len(reader)))
        frames = reader.get_batch(frame_indices)  # [F,H,W,C]

    nearest_res = find_nearest_resolution(
        height_buckets,
        width_buckets,
        frame_buckets,
        frames.shape[1],
        frames.shape[2],
    )

    frames = frames.permute(0, 3, 1, 2).contiguous()  # [F,C,H,W]
    frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
    frames = torch.stack([image_transforms(frame) for frame in frames_resized],dim=0)

    return frames

def _video_to_uint8_array(video):
    """Convert a video tensor/array to uint8 NHWC format."""
    if torch.is_tensor(video):
        arr = video.detach().cpu().float().numpy()
    else:
        arr = np.asarray(video).astype(np.float32)

    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor/array, got shape {arr.shape}")

    # Accept either [F, C, H, W] or [F, H, W, C]
    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))

    # Normalize common ranges to [0, 255]
    if arr.min() >= -1.5 and arr.max() <= 1.5:
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def generate_video(
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    pipe: Union[WanPipeline, WanImageToVideoPipelineTracking],
    dtype: torch.dtype = torch.bfloat16,
    fps: int = 24,
    seed: int = 42,
):
    """Run inference and export generated videos."""
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

    prompts = file_lines_to_list(os.path.join(args.data_root, args.caption_column))
    foreground_videos = file_lines_to_list(os.path.join(args.data_root, args.foreground_column))

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    bg_reference_imgs = None
    light_prompts = None

    bg_reference_imgs_path = os.path.join(args.data_root, args.background_column)
    if os.path.exists(bg_reference_imgs_path):
        bg_reference_imgs = file_lines_to_list(bg_reference_imgs_path)

    light_prompt_path = os.path.join(args.data_root, args.lighting_caption_column)
    if os.path.exists(light_prompt_path):
        light_prompts = file_lines_to_list(light_prompt_path)

    use_background = (
        bg_reference_imgs is not None
        and light_prompts is not None
        and len(bg_reference_imgs) > 0
        and len(light_prompts) > 0
    )

    tracking_videos = (
        file_lines_to_list(os.path.join(args.data_root, args.tracking_column))
        if args.tracking_column is not None
        else None
    )
    hdr_videos = (
        file_lines_to_list(os.path.join(args.data_root, args.hdr_column))
        if args.hdr_column is not None
        else None
    )

    foreground_list = []
    for i in tqdm(range(len(foreground_videos))):
        print("Processing video index: "  + str(i) + ", | Video name: " + str(foreground_videos[i]) + "| Processed number: " + str(len(foreground_list)))

        foreground_map_path = Path(os.path.join(args.data_root, foreground_videos[i]))
        foreground_frames = prepare_frames(
            foreground_map_path,
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

        if args.hdr_column is not None and hdr_videos is not None:
            hdr_map_path = Path(os.path.join(args.data_root, hdr_videos[i]))
            hdr_maps = _load_video_frames(
                hdr_map_path.as_posix(),
                args.height_buckets,
                args.width_buckets,
                args.frame_buckets,
                image_transforms,
            )

        if args.tracking_column is not None and tracking_videos is not None:
            tracking_map_path = Path(os.path.join(args.data_root, tracking_videos[i]))
            tracking_maps = _load_video_frames(
                tracking_map_path.as_posix(),
                args.height_buckets,
                args.width_buckets,
                args.frame_buckets,
                image_transforms,
            )

        with torch.no_grad():
            foreground_frames = foreground_frames.unsqueeze(0).to(device=device, dtype=dtype)
            foreground_frames = foreground_frames.permute(0, 2, 1, 3, 4)

            if args.hdr_column is not None and hdr_maps is not None:
                hdr_maps = hdr_maps.unsqueeze(0).to(device=device, dtype=dtype)
                hdr_maps = hdr_maps.permute(0, 2, 1, 3, 4)
                n, c, d, h, w = hdr_maps.shape
                hdr_maps_flat = hdr_maps.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
                hdr_maps_flat = F.interpolate(hdr_maps_flat, size=(32, 32), mode="bilinear", align_corners=False)
                hdr_maps = hdr_maps_flat.view(n, d, c, 32, 32).permute(0, 2, 1, 3, 4)

            if args.tracking_column is not None and tracking_maps is not None:
                tracking_maps = tracking_maps.unsqueeze(0).to(device=device, dtype=dtype)
                tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
                tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
                tracking_maps = tracking_latent_dist.sample().to(device=device, dtype=dtype)

        base_prompt = prompts[i].split(".")[0]
        random_idx = random.randrange(len(light_prompts))
        light_prompt = light_prompts[random_idx]
        prompt = base_prompt + ", " + light_prompts[random_idx].lower()

        # -----------------------
        # 1) Background-conditioned generation (optional)
        # -----------------------
        if use_background:
            ref_map_path = Path(os.path.join(args.data_root, bg_reference_imgs[random_idx]))
            ref_frames = prepare_frames(
                ref_map_path,
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

            bg_pipeline_args = dict(pipeline_args)
            bg_pipeline_args["control_video"] = foreground_frames
            bg_pipeline_args["ref_image"] = ref_image
            if args.hdr_column is not None and hdr_maps is not None:
                bg_pipeline_args["hdr_maps"] = hdr_maps
            if args.tracking_column is not None and tracking_maps is not None:
                bg_pipeline_args["tracking_maps"] = tracking_maps
            bg_pipeline_args["prompt"] = prompt

            bg_video_output_path = os.path.join(
                output_root.as_posix(),
                Path(foreground_videos[i]).stem + "_bg.mp4",
            )
            print("[BG]", bg_video_output_path)
            print("[BG prompt]", bg_pipeline_args["prompt"])

            generated_video_bg = pipe(
                **bg_pipeline_args,
                generator=torch.Generator(device=device).manual_seed(seed),
                output_type="np",
            ).videos.numpy()[0]
            export_to_video(generated_video_bg, bg_video_output_path, fps=fps)

            bg_concat_path = bg_video_output_path.replace(".mp4", "_concat.mp4")
            save_side_by_side_video(
                foreground_frames=foreground_frames[0].permute(1, 2, 3, 0),  # [F, H, W, C]
                background_image=ref_image,
                generated_video=generated_video_bg,
                output_path=bg_concat_path,
                fps=fps,
            )

        # -----------------------
        # 2) Generation without background
        # -----------------------
        nobg_pipeline_args = dict(pipeline_args)
        nobg_pipeline_args["control_video"] = foreground_frames
        if args.hdr_column is not None and hdr_maps is not None:
            nobg_pipeline_args["hdr_maps"] = hdr_maps
        if args.tracking_column is not None and tracking_maps is not None:
            nobg_pipeline_args["tracking_maps"] = tracking_maps
        nobg_pipeline_args["prompt"] = prompt

        nobg_video_output_path = os.path.join(
            output_root.as_posix(),
            Path(foreground_videos[i]).stem + "_nobg.mp4",
        )
        print("[NoBG]", nobg_video_output_path)
        print("[NoBG prompt]", nobg_pipeline_args["prompt"])

        generated_video_nobg = pipe(
            **nobg_pipeline_args,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="np",
        ).videos.numpy()[0]
        export_to_video(generated_video_nobg, nobg_video_output_path, fps=fps)

        nobg_concat_path = nobg_video_output_path.replace(".mp4", "_concat.mp4")
        save_side_by_side_foreground_generated_video(
            foreground_frames=foreground_frames,
            generated_video=generated_video_nobg,
            output_path=nobg_concat_path,
            fps=fps,
        )

def main():
    """Parse arguments, build the pipeline, and launch inference."""
    parser = argparse.ArgumentParser(description="Generate a video using the Illumicraft pipeline")
    parser.add_argument("--data_root", type=str, default=None, help="A folder containing the data.")
    parser.add_argument("--config_path", type=str, default=None, help="The config of the model in training.")
    parser.add_argument(
        "--foreground_column",
        type=str,
        default="video",
        help="The column of the dataset containing foreground videos.",
    )
    parser.add_argument(
        "--tracking_column",
        type=str,
        default=None,
        help="The column of the dataset containing the tracking map for each sample.",
    )
    parser.add_argument(
        "--hdr_column",
        type=str,
        default=None,
        help="The column of the dataset containing the HDR map for each sample.",
    )
    parser.add_argument(
        "--background_column",
        type=str,
        default=None,
        help="The column of the dataset containing the background image paths.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each sample.",
    )
    parser.add_argument(
        "--lighting_caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each sample.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan2.1-Fun-1.3B-Control",
        help="The path of the pre-trained model to be used",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model",
        help="The path of the checkpoint to be used for the transformer.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="The directory where generated videos will be saved.",
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--height", type=int, default=480, help="All input videos are resized to this height.")
    parser.add_argument("--width", type=int, default=720, help="All input videos are resized to this width.")
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
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

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
        "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
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
