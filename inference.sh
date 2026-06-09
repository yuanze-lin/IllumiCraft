#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"
OUTPUT_PATH="demo/outputs"

DATA_ROOT="dataset/demo_examples"
CAPTION_COLUMN="foreground_prompt.txt"
FOREGROUND_COLUMN="foreground_videos.txt"

# Optional: enable background-conditioned generation
BACKGROUND_COLUMN="background_images.txt"
LIGHT_CAPTION_COLUMN="lighting_prompt.txt"

python testing/inference.py \
    --data_root $DATA_ROOT \
    --config_path config/wan.yaml \
    --model_path $ILLUMICRAFT_CKPT_PATH \
    --caption_column $CAPTION_COLUMN \
    --lighting_caption_column $LIGHT_CAPTION_COLUMN \
    --foreground_column $FOREGROUND_COLUMN \
    --background_column $BACKGROUND_COLUMN \
    --output_path $OUTPUT_PATH
