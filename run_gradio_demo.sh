#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH

ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"

python utils/app.py \
  --illumicraft_ckpt_path "$ILLUMICRAFT_CKPT_PATH" \
  --config_path config/wan.yaml \
  --output_dir ./gradio_outputs \
  --host 0.0.0.0 \
  --port 7860
