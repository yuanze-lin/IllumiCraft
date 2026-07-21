#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"
OUTPUT_PATH="demo/single_sample_from_video_outputs"

# Raw input video with a real (non-gray) background -- the foreground video is
# auto-extracted from this via SAM3 (text-prompted segmentation) + MatAnyone
# (video matting), instead of requiring an already-prepared foreground video.
INPUT_VIDEO_PATH="demo/eval/00000.mp4"
FOREGROUND_PROMPT="A majestic waterfall cascades down a rugged cliff into a serene pool."
LIGHTING_PROMPT="Cool-blue spotlights beam through mist onto a central pool of light, creating high-contrast cinematic depth and a moody, immersive atmosphere."

# Optional background-conditioned generation
BACKGROUND_PATH="demo/eval/custom_background.jpg"

# fgprep env (SAM3 + MatAnyone) -- auto-detected as the named `fgprep` conda
# env under your conda installation (see README's Foreground Video
# Preparation section). Uncomment and set explicitly if your conda envs live
# somewhere non-standard (e.g. created with `conda create --prefix`).
# FGPREP_PYTHON="/path/to/conda/envs/fgprep/bin/python"

python testing/inference_single_sample.py \
    --config_path config/wan.yaml \
    --model_path "$ILLUMICRAFT_CKPT_PATH" \
    --input_video_path "$INPUT_VIDEO_PATH" \
    --foreground_prompt "$FOREGROUND_PROMPT" \
    --lighting_prompt "$LIGHTING_PROMPT" \
    --output_path "$OUTPUT_PATH" \
    ${FGPREP_PYTHON:+--fgprep_python "$FGPREP_PYTHON"} \
    ${BACKGROUND_PATH:+--background_path "$BACKGROUND_PATH"}
