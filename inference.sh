#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

WAN_MODEL_PATH="/mnt/data0/yuanze/ckpt/Wan2.1-Fun-1.3B-Control"
ILLUMICRAFT_CKPT_PATH="/mnt/data0/yuanze/illumicraft_checkpoint"
OUTPUT_PATH="demo/outputs"

DATA_ROOT="/mnt/data0/yuanze/dataset/demo_examples"
CAPTION_COLUMN="foreground_prompt.txt"
FOREGROUND_COLUMN="foreground_videos.txt"

# Optional: enable background-conditioned generation
BACKGROUND_COLUMN=background.txt
LIGHT_CAPTION_COLUMN="lighting_prompt.txt"

python testing/inference.py \
    --pretrained_model_name_or_path $WAN_MODEL_PATH \
    --data_root $DATA_ROOT \
    --config_path config/wan.yaml \
    --model_path $ILLUMICRAFT_CKPT_PATH \
    --caption_column $CAPTION_COLUMN \
    --lighting_caption_column $LIGHT_CAPTION_COLUMN \
    --foreground_column $FOREGROUND_COLUMN \
    --background_column $BACKGROUND_COLUMN \
    --output_path $OUTPUT_PATH
