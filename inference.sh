#!/bin/bash
# YOU MUST SET THE CUDA_HOME AND PATH AND LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Absolute paths
PRETRAINED_MODEL_PATH="/path/to/Wan2.1-Fun-1.3B-Control"
MODEL_PATH="checkpoints/illumicraft"
OUTPUT_PATH="demo/outputs"
DATA_ROOT="examples"
CAPTION_COLUMN="prompt.txt"
FOREGROUND_COLUMN="foreground_videos.txt"

python testing/inference.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --data_root $DATA_ROOT \
    --config_path "config/wan.yaml" \
    --model_path $MODEL_PATH \
    --caption_column $CAPTION_COLUMN \
    --foreground_column $FOREGROUND_COLUMN \
    --output_path $OUTPUT_PATH
