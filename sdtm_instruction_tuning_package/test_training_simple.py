#!/usr/bin/env python3
"""
Simple test of the training pipeline
"""

import subprocess
import sys
import os

# Set minimal environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = './cache'
os.environ['TRANSFORMERS_CACHE'] = './cache'

print("Testing SDTM Instruction Tuning Training Pipeline")
print("=" * 50)

# First, let's just test if we can load the model
print("\n1. Testing model loading...")

test_code = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try loading a small model
model_name = "microsoft/phi-2"
print(f"Testing with model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("✓ Tokenizer loaded")

# Try without quantization first
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("✓ Model loaded successfully")
    
    # Test generation
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0])
    print(f"✓ Generation test: {result[:50]}...")
    
except Exception as e:
    print(f"✗ Error: {e}")
"""

# Run the test
result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Warnings/Errors:", result.stderr[:500])

# Now test with our actual training script but minimal settings
print("\n2. Testing training script with minimal data...")

# Create a minimal training command
cmd = [
    sys.executable, "train_model.py",
    "--model_name_or_path", "microsoft/phi-2",
    "--data_path", "./data",
    "--output_dir", "./test_output",
    "--max_steps", "5",  # Only 5 steps
    "--per_device_train_batch_size", "1",
    "--learning_rate", "2e-4",
    "--do_train",
    "--no-use_4bit",  # Disable 4-bit for testing
    "--use_lora",
    "--lora_r", "8",
    "--lora_alpha", "16",
    "--save_steps", "1000",  # Don't save during test
    "--logging_steps", "1",
    "--report_to", "none",
    "--overwrite_output_dir"
]

print("Running command:")
print(" ".join(cmd))
print("\nTraining output:")
print("-" * 50)

# Run training
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Show last 2000 chars
if result.stderr:
    print("\nErrors/Warnings:")
    print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)  # Show last 1000 chars

print("\n" + "=" * 50)
print("Test completed!")

# Check if output was created
if os.path.exists("./test_output"):
    print("✓ Output directory created")
    files = os.listdir("./test_output")
    print(f"  Files created: {files[:5]}")  # Show first 5 files
else:
    print("✗ No output directory created")