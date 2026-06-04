#!/bin/bash
# Environment variables
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"            # No torch logs
export TORCHDYNAMO_VERBOSE=0
export WANDB_MODE=${WANDB_MODE:-"online"}
export WANDB_PROJECT=${WANDB_PROJECT:-"IllumiCraft"}
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=30

# Use only 4 GPUs (GPU index 0, 1, 2, 3) and 4 processes.
GPU_IDS="0,1,2,3"
NUM_PROCESSES=4
PORT=28800

# Training Configurations
LEARNING_RATES=("4e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("3000")
WARMUP_STEPS=100
CHECKPOINT_STEPS=100
TRAIN_BATCH_SIZE=2

# Multi-GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"

# Use paths that are set locally by the user
DATA_ROOT="/path/to/train_dataset"
WAN_MODEL_PATH="/path/to/Wan2.1-Fun-1.3B-Control"
OUTPUT_PATH="checkpoints/illumicraft_weights"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
FOREGROUND_COLUMN="foreground_videos.txt"
HDR_COLUMN="lighting_videos.txt"
TRACKING_COLUMN="tracking_videos.txt"
BACKGROUND_COLUMN="background_videos.txt"

# Validation parameters
HDR_MAP_PATH="demo/eval/lighting_videos_00000.mp4"
TRACKING_MAP_PATH="demo/eval/tracking_videos_00000.mp4"
VALIDATION_IMAGES="demo/eval/foreground_videos_00000.mp4"
VALIDATION_BACKGROUNDS="demo/eval/background_videos_00000.mp4"
VALIDATION_PROMPT="A majestic waterfall cascades down a rugged cliff into a serene pool,
surrounded by lush greenery and ancient rock formations. The soft lighting enhances the
tranquil atmosphere, with a subtle rainbow arc adding a touch of magic. As the scene continues,
the waterfall remains a consistent, serene, and untouched natural spectacle, with the surrounding
foliage and the soft lighting emphasizing the peacefulness and the beauty of the landscape"

# Launch experiments with different hyperparameters.
# Note: The command uses "-m pdb" to launch the Python debugger.
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_PATH}/illumicraft_steps_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"
        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
          --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --main_process_port $PORT \
          training/train.py \
          --config_path="config/wan.yaml" \
          --pretrained_model_name_or_path $WAN_MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --foreground_column $FOREGROUND_COLUMN \
          --background_column $BACKGROUND_COLUMN \
          --tracking_column $TRACKING_COLUMN \
	        --hdr_column $HDR_COLUMN \
          --hdr_map_path $HDR_MAP_PATH \
	        --tracking_map_path $TRACKING_MAP_PATH \
          --height_buckets 480 \
          --width_buckets 720 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt \"$VALIDATION_PROMPT\" \
          --validation_images \"$VALIDATION_IMAGES\" \
          --validation_backgrounds \"$VALIDATION_BACKGROUNDS\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size $TRAIN_BATCH_SIZE \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINT_STEPS \
          --gradient_accumulation_steps 4 \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
	        --gradient_checkpointing \
          --lr_warmup_steps $WARMUP_STEPS \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --noised_image_dropout 0.05 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
	        --resume_from_checkpoint \"latest\" \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done

