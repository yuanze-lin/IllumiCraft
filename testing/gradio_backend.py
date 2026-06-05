import tempfile
from pathlib import Path
from types import SimpleNamespace

import decord
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

import testing.inference_single_sample as infer


def normalize_upload(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x.get("name") or x.get("path") or x.get("file")
    if isinstance(x, (list, tuple)):
        return normalize_upload(x[0]) if x else None
    return str(x)


def prepare_frames_fixed(
    path,
    height_buckets,
    width_buckets,
    frame_buckets,
    image_transforms,
    num_frames=49,
    width=720,
    height=480,
):
    """Same behavior as the original, but always converts video batches to torch.Tensor."""
    path = str(path)

    if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img = Image.open(path).convert("RGB")
        img = img.resize((width, height))
        first_frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()
        zeros = torch.zeros((num_frames - 1, *first_frame.shape), dtype=first_frame.dtype)
        frames = torch.cat([first_frame.unsqueeze(0), zeros], dim=0)
    else:
        reader = decord.VideoReader(uri=path)
        frame_indices = list(range(len(reader)))
        frames = reader.get_batch(frame_indices)

        if hasattr(frames, "asnumpy"):
            frames = torch.from_numpy(frames.asnumpy())
        elif not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)

        frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = infer.find_nearest_resolution(
        height_buckets,
        width_buckets,
        frame_buckets,
        frames.shape[2],
        frames.shape[3],
    )
    frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
    frames = torch.stack([image_transforms(frame) for frame in frames_resized], dim=0)
    return frames


def patch_inference_module():
    """Replace the original prepare_frames with the fixed one."""
    infer.prepare_frames = prepare_frames_fixed


def run_gradio_inference(
    pipe,
    wan_model_path,
    illumicraft_ckpt_path,
    config_path,
    output_dir,
    foreground_video,
    foreground_prompt,
    lighting_prompt,
    background_file,
    seed,
    guidance_scale=6.0,
    height=480,
    width=720,
    height_buckets=None,
    width_buckets=None,
    frame_buckets=None,
    dtype_str="bfloat16",
):
    patch_inference_module()

    foreground_video = normalize_upload(foreground_video)
    background_file = normalize_upload(background_file)

    if not foreground_video:
        raise ValueError("Foreground video is required.")
    if not foreground_prompt or not foreground_prompt.strip():
        raise ValueError("Foreground prompt is required.")

    height_buckets = height_buckets or [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
    width_buckets = width_buckets or [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
    frame_buckets = frame_buckets or [49]

    run_dir = tempfile.mkdtemp(dir=output_dir, prefix="illumicraft_")

    args = SimpleNamespace(
        config_path=config_path,
        foreground_video_path=foreground_video,
        base_prompt=foreground_prompt.strip(),
        lighting_prompt=(lighting_prompt or "").strip() or None,
        background_path=background_file,
        tracking_path=None,
        hdr_path=None,
        pretrained_model_name_or_path=wan_model_path,
        model_path=illumicraft_ckpt_path,
        output_path=run_dir,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        height_buckets=height_buckets,
        width_buckets=width_buckets,
        frame_buckets=frame_buckets,
        dtype=dtype_str,
        seed=int(seed),
    )

    infer.generate_video(
        args=args,
        pipeline_args={
            "negative_prompt": (
                "The video is not of a high quality, it has a low resolution. "
                "Watermark present in each frame. The background is solid. "
                "Strange body and strange trajectory. Distortion."
            ),
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "max_sequence_length": 512,
        },
        pipe=pipe,
        dtype=torch.bfloat16 if dtype_str == "bfloat16" else torch.float16,
        fps=24,
        seed=int(seed),
    )

    outputs = sorted(
        p for p in Path(run_dir).glob("*.mp4")
        if "_concat" not in p.name
    )
    if not outputs:
        raise RuntimeError("No output video was generated.")
    return str(outputs[0])
