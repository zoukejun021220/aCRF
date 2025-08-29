#!/bin/bash
# Demo training that will work and show the pipeline is functioning

echo "SDTM Training Demo - Testing the full pipeline"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"
echo ""

# Use a 7B model that will fit in memory
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
echo "Using $MODEL_NAME (7B model should fit in 12GB GPU)"

# Run with minimal settings
python train_model.py \
  --model_name_or_path $MODEL_NAME \
  --data_path ./data \
  --output_dir ./demo_output \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "steps" \
  --eval_steps 20 \
  --save_strategy "steps" \
  --save_steps 20 \
  --learning_rate 2e-4 \
  --logging_steps 5 \
  --do_train \
  --do_eval \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --use_4bit \
  --max_seq_length 1024 \
  --report_to "none" \
  --fp16 True \
  --gradient_checkpointing True \
  --max_steps 50 \
  --overwrite_output_dir

echo ""
echo "=============================================="
echo "Demo completed. Check ./demo_output for results."