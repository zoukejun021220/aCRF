#!/usr/bin/env python3
"""
Minimal test to verify the training setup works
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

print("SDTM Training Test - Minimal Configuration")
print("=" * 50)

# Check environment
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load data
print("\nLoading training data...")
data_file = Path("./data/alpaca_format.json")
with open(data_file) as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"First example: {data[0]['instruction'][:50]}...")

# Prepare mini dataset (10 examples)
mini_data = data[:10]

# Initialize tokenizer (use a small model for testing)
model_name = "gpt2"  # Very small model for testing
print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize data
def tokenize_function(examples):
    texts = []
    for i in range(len(examples['instruction'])):
        text = f"Instruction: {examples['instruction'][i]}\n"
        text += f"Input: {examples['input'][i]}\n"
        text += f"Output: {examples['output'][i]}"
        texts.append(text)
    
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=256)
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings

# Create dataset
dataset = Dataset.from_dict({
    'instruction': [ex['instruction'] for ex in mini_data],
    'input': [ex['input'] for ex in mini_data],
    'output': [ex['output'] for ex in mini_data]
})

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into train/eval
train_dataset = tokenized_dataset.select(range(8))
eval_dataset = tokenized_dataset.select(range(8, 10))

print(f"Train examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)}")

# Load model
print(f"\nLoading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Training arguments (minimal)
training_args = TrainingArguments(
    output_dir="./test_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=1,
    save_steps=100,
    eval_strategy="no",
    save_strategy="no",
    max_steps=5,  # Only 5 steps
    report_to="none",
    overwrite_output_dir=True
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train
print("\nStarting training (5 steps)...")
trainer.train()

print("\n✓ Training completed successfully!")

# Test generation
print("\nTesting generation...")
prompt = "Instruction: Select SDTM domain\nInput: Blood pressure measurement\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated: {response}")
print("\n✓ All tests passed!")