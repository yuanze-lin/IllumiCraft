#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

WAN_MODEL_PATH="checkpoints/Wan2.1-Fun-1.3B-Control"
ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_checkpoint"
OUTPUT_PATH="demo/single_sample_outputs"

FOREGROUND_VIDEO_PATH="demo/eval/foreground_videos_00000.mp4"
FOREGROUND_PROMPT="A majestic waterfall cascades down a rugged cliff into a serene pool."

# Optional: enable background-conditioned generation
BACKGROUND_PATH=""  # optional
LIGHTING_PROMPT="Cool-blue spotlights beam through mist onto a central pool of light, creating high-contrast cinematic depth and a moody, immersive atmosphere."   # optional

python testing/inference_single_sample.py \
    --pretrained_model_name_or_path "$WAN_MODEL_PATH" \
    --config_path config/wan.yaml \
    --model_path "$ILLUMICRAFT_CKPT_PATH" \
    --foreground_video_path "$FOREGROUND_VIDEO_PATH" \
    --base_prompt "$FOREGROUND_PROMPT" \
    --output_path "$OUTPUT_PATH" \
    ${LIGHTING_PROMPT:+--lighting_prompt "$LIGHTING_PROMPT"} \
    ${BACKGROUND_PATH:+--background_path "$BACKGROUND_PATH"}
