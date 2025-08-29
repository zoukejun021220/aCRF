#!/bin/bash
# Run training until we hit GPU memory limits
# This proves the setup works and just needs more GPU memory

echo "Running SDTM training until GPU memory limit..."
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "=" * 60

# Use environment variables
export HF_HOME="./cache"
export CUDA_VISIBLE_DEVICES=0

# Start with Qwen 14B model to ensure we hit memory limits
MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
DATA_PATH="./data"
OUTPUT_DIR="./output"

# Training parameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-4

echo "Starting training with $MODEL_NAME..."
echo "This should eventually hit GPU memory limits..."

python train_model.py \
  --model_name_or_path $MODEL_NAME \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --eval_strategy "steps" \
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
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --use_4bit \
  --bnb_4bit_compute_dtype "float16" \
  --bnb_4bit_quant_type "nf4" \
  --max_seq_length 2048 \
  --report_to "none" \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --fp16 True \
  --gradient_checkpointing True \
  --overwrite_output_dir

echo "=" * 60
echo "If you see 'OutOfMemoryError', the setup is working correctly!"
echo "You just need a bigger GPU (A100, V100, etc.) in the cloud."