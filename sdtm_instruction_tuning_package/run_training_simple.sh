#!/bin/bash
# Simple training script - ready to run

echo "SDTM Instruction Tuning - Simple Start"
echo "======================================"

# Step 1: Check environment
echo "1. Checking environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
    echo "PyTorch not properly installed. Running setup..."
    ./setup_environment.sh
}

# Step 2: Check if data exists
echo -e "\n2. Checking data..."
if [ ! -f "./data/alpaca_format.json" ]; then
    echo "Training data not found. Generating..."
    
    # First, let me create a simple test dataset
    python -c "
import json
test_data = [
    {
        'instruction': 'You are helping select the appropriate SDTM domain for CRF fields. Available domains: AE, CM, DM, DS, EG, IE, LB, MH, PE, QS, RS, SV, VS',
        'input': 'Select the SDTM domain for: Field Label: Age ≥ 18 years (INC1), Form: Baseline, Section: Inclusion Criteria',
        'output': 'IE'
    },
    {
        'instruction': 'You are selecting the annotation pattern for domain IE. Available patterns: plain, findings, conditional, supplemental, not_submitted',
        'input': 'Select pattern for: Domain: IE, Field Label: Age ≥ 18 years (INC1), Has Options: True',
        'output': 'conditional'
    },
    {
        'instruction': 'You are helping select the appropriate SDTM domain for CRF fields. Available domains: AE, CM, DM, DS, EG, IE, LB, MH, PE, QS, RS, SV, VS',
        'input': 'Select the SDTM domain for: Field Label: Systolic Blood Pressure, Form: Vital Signs, Section: Measurements',
        'output': 'VS'
    }
]
with open('./data/alpaca_format.json', 'w') as f:
    json.dump(test_data, f, indent=2)
print(f'Created test dataset with {len(test_data)} examples')
    "
fi

# Step 3: Run training with minimal settings
echo -e "\n3. Starting training..."
echo "Using ModelScope for model download (China-friendly)..."

python train_model.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --use_modelscope \
    --data_path ./data \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 16 \
    --use_4bit \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 5 \
    --do_train \
    --fp16 \
    --max_seq_length 512 \
    --warmup_steps 10