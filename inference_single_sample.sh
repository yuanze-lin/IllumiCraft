#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"
OUTPUT_PATH="demo/single_sample_outputs"

FOREGROUND_VIDEO_PATH="demo/eval/foreground_videos_00000.mp4"
FOREGROUND_PROMPT="A majestic waterfall cascades down a rugged cliff into a serene pool."
LIGHTING_PROMPT="Cool-blue spotlights beam through mist onto a central pool of light, creating high-contrast cinematic depth and a moody, immersive atmosphere."

# Optional background-conditioned generation
BACKGROUND_PATH=""

python testing/inference_single_sample.py \
    --config_path config/wan.yaml \
    --model_path "$ILLUMICRAFT_CKPT_PATH" \
    --foreground_video_path "$FOREGROUND_VIDEO_PATH" \
    --foreground_prompt "$FOREGROUND_PROMPT" \
    --lighting_prompt "$LIGHTING_PROMPT" \
    --output_path "$OUTPUT_PATH" \
    ${BACKGROUND_PATH:+--background_path "$BACKGROUND_PATH"}
