#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

WAN_MODEL_PATH="checkpoints/Wan2.1-Fun-1.3B-Control"
ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"

python app.py \
  --wan_model_path "$WAN_MODEL_PATH" \
  --illumicraft_ckpt_path "$ILLUMICRAFT_CKPT_PATH" \
  --config_path config/wan.yaml \
  --output_dir ./gradio_outputs \
  --host 0.0.0.0 \
  --port 7860
