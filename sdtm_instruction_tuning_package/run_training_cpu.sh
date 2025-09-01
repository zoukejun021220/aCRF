#!/bin/bash
# Training script for CPU-only environments (no GPU/CUDA)

# Set environment variables
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export WANDB_PROJECT="sdtm-mapper-finetuning"
export WANDB_MODE="offline"  # Offline mode for CPU training

# Model configuration - using smaller model for CPU
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for CPU
USE_MODELSCOPE=false
DATA_PATH="./data"
OUTPUT_DIR="./output"

# CPU-friendly hyperparameters
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
LEARNING_RATE=2e-4
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Running CPU-only training (no 4-bit quantization)..."

# Run training without 4-bit quantization
python train_model.py \
  --model_name_or_path $MODEL_NAME \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 2 \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --do_eval \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --use_4bit False \
  --max_seq_length $MAX_SEQ_LENGTH \
  --preprocessing_num_workers 2 \
  --dataloader_num_workers 0 \
  --report_to "none" \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --bf16 False \
  --fp16 False \
  --gradient_checkpointing True
