#!/bin/bash
# Training script for SDTM mapper instruction tuning

#!/usr/bin/env bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-"sdtm-mapper-finetuning"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"./cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"./cache"}

# Resolve paths relative to this script to avoid hardcoded host paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
USE_MODELSCOPE=${USE_MODELSCOPE:-false}  # Set to true to use ModelScope
MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-}"  # Optional: specify different ModelScope ID
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"

# Ensure dataset exists; if missing, build from default external reference
if [ ! -f "$DATA_PATH/alpaca_format.json" ]; then
  echo "Dataset not found at $DATA_PATH. Generating from repository reference_with_sections..."
  REF_DIR_DEFAULT="${REFERENCE_DIR:-$REPO_ROOT/reference_with_sections}"
  CRF_DIR_DEFAULT="${CRF_DIR:-$REPO_ROOT/crf_json}"
  python "$SCRIPT_DIR/create_instruction_dataset.py" \
    --reference-dir "$REF_DIR_DEFAULT" \
    --crf-dir "$CRF_DIR_DEFAULT" \
    --output-dir "$DATA_PATH"
fi

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}

# LoRA configuration
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build ModelScope arguments
MODELSCOPE_ARGS=""
if [ "$USE_MODELSCOPE" = true ]; then
  MODELSCOPE_ARGS="--use_modelscope True"
  if [ ! -z "$MODELSCOPE_MODEL_ID" ]; then
    MODELSCOPE_ARGS="$MODELSCOPE_ARGS --modelscope_model_id $MODELSCOPE_MODEL_ID"
  fi
fi

# Decide reporting backend (default to none if WANDB not configured)
REPORT_TO="${REPORT_TO:-}"
if [ -z "$REPORT_TO" ]; then
  if [ -n "$WANDB_API_KEY" ]; then
    REPORT_TO="wandb"
  else
    REPORT_TO="none"
  fi
fi

# Run training
python "$SCRIPT_DIR/train_model.py" \
  --model_name_or_path $MODEL_NAME \
  $MODELSCOPE_ARGS \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --do_eval \
  --use_lora \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --use_4bit \
  --bnb_4bit_compute_dtype "float16" \
  --bnb_4bit_quant_type "nf4" \
  --max_seq_length $MAX_SEQ_LENGTH \
  --preprocessing_num_workers 4 \
  --report_to "$REPORT_TO" \
  --run_name "sdtm-mapper-qlora" \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --ddp_find_unused_parameters False \
  --group_by_length \
  --bf16 False \
  --fp16 True \
  --gradient_checkpointing True
