"""Foreground Video Preparation for IllumiCraft.

Given a raw RGB input video (real background) and a short text description of
the subject to keep, this reproduces the paper's foreground-video-preparation
pipeline:

  1. SAM3 (text-prompted concept segmentation) on the first frame -> seed mask.
  2. MatAnyone (video matting) propagates that seed mask across the whole clip
     and returns a per-frame alpha matte.
  3. The original video is composited against the alpha matte onto the fixed
     gray background (136, 139, 136) used by IllumiCraft's own reference
     script (utils/generate_foreground_video_example.py).

Usable both as a CLI and as a library: `prepare_foreground_video(...)` runs the
whole pipeline in-process, which is how the inference scripts auto-generate a
foreground video from `--input_video_path`.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

GRAY_BACKGROUND = (136, 139, 136)  # #888b88, matches generate_foreground_video_example.py


def read_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame of {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def compute_seed_mask(frame_rgb, text_prompt, sam3_model_path, score_threshold, device):
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained(sam3_model_path).to(device)
    processor = Sam3Processor.from_pretrained(sam3_model_path)

    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        raise RuntimeError(
            f"SAM3 found no object matching text_prompt={text_prompt!r} "
            f"(score_threshold={score_threshold}). Try a shorter noun phrase via --text_prompt."
        )

    best_idx = int(torch.as_tensor(scores).argmax())
    mask = masks[best_idx]
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    # free SAM3 before MatAnyone loads onto the same GPU
    del model, processor, outputs
    torch.cuda.empty_cache()
    return mask


def run_matanyone(video_path, seed_mask_path, work_dir, matanyone_model):
    from matanyone import InferenceCore

    processor = InferenceCore(matanyone_model)
    foreground_path, alpha_path = processor.process_video(
        input_path=str(video_path),
        mask_path=str(seed_mask_path),
        output_path=str(work_dir),
    )
    return foreground_path, alpha_path


def composite_gray_background(video_path, alpha_video_path, out_path):
    cap_v = cv2.VideoCapture(str(video_path))
    cap_m = cv2.VideoCapture(str(alpha_video_path))

    fps = cap_v.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    background = np.full((h, w, 3), GRAY_BACKGROUND, dtype=np.uint8)

    n_written = 0
    while True:
        ret_v, frame = cap_v.read()
        ret_m, mask_frame = cap_m.read()
        if not ret_v or not ret_m:
            break

        if mask_frame.shape[:2] != (h, w):
            mask_frame = cv2.resize(mask_frame, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
        mask = mask[..., None]

        output = frame * mask + background * (1 - mask)  # frame is BGR (cv2 convention)
        output_rgb = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB)
        writer.append_data(output_rgb)
        n_written += 1

    cap_v.release()
    cap_m.release()
    writer.close()

    if n_written == 0:
        raise RuntimeError("Compositing produced zero frames -- check the input/alpha videos.")
    return out_path


def prepare_foreground_video(
    video_path,
    text_prompt,
    output_path,
    sam3_model_path="checkpoints/sam3",
    matanyone_model="PeiqingYang/MatAnyone",
    score_threshold=0.4,
    work_dir=None,
    device=None,
):
    """Run the full SAM3 seed mask + MatAnyone matting + gray composite pipeline.

    Returns the path to the written gray-background foreground video.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cleanup_work_dir = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="illumicraft_fgprep_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[prepare_foreground_video] Reading first frame of {video_path}")
        frame_rgb = read_first_frame(video_path)

        print(f"[prepare_foreground_video] Running SAM3 with text_prompt={text_prompt!r}")
        seed_mask = compute_seed_mask(frame_rgb, text_prompt, sam3_model_path, score_threshold, device)
        seed_mask_path = work_dir / "seed_mask.png"
        cv2.imwrite(str(seed_mask_path), seed_mask)

        print("[prepare_foreground_video] Running MatAnyone video matting")
        _, alpha_path = run_matanyone(video_path, seed_mask_path, work_dir, matanyone_model)

        print(f"[prepare_foreground_video] Compositing onto gray background -> {output_path}")
        composite_gray_background(video_path, alpha_path, output_path)

        print(f"[prepare_foreground_video] Done: {output_path}")
        return str(output_path)
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare a foreground video (SAM3 seed mask + MatAnyone matting).")
    parser.add_argument("--video_path", type=str, required=True, help="Raw input video with a real background.")
    parser.add_argument("--text_prompt", type=str, required=True, help="Short description of the subject to keep (SAM3 concept prompt).")
    parser.add_argument("--output_path", type=str, required=True, help="Where to write the final gray-background foreground video.")
    parser.add_argument("--sam3_model_path", type=str, default=os.environ.get("SAM3_MODEL_PATH", "checkpoints/sam3"))
    parser.add_argument("--matanyone_model", type=str, default="PeiqingYang/MatAnyone")
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--work_dir", type=str, default=None, help="Scratch dir for intermediate mask/matte files (defaults to a temp dir).")
    args = parser.parse_args()

    prepare_foreground_video(
        video_path=args.video_path,
        text_prompt=args.text_prompt,
        output_path=args.output_path,
        sam3_model_path=args.sam3_model_path,
        matanyone_model=args.matanyone_model,
        score_threshold=args.score_threshold,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
