#!/usr/bin/env python3
"""
Verify the training setup is working correctly
"""

import json
import torch
import sys
from pathlib import Path

print("SDTM Instruction Tuning - Setup Verification")
print("=" * 60)

# 1. Check Python and PyTorch
print("1. Environment Check:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. Check data
print("\n2. Data Check:")
data_file = Path("./data/alpaca_format.json")
if data_file.exists():
    with open(data_file) as f:
        data = json.load(f)
    print(f"   ✓ Training data found: {len(data)} examples")
    print(f"   ✓ First example type: {data[0].keys()}")
else:
    print("   ✗ Training data not found!")

# 3. Check KB files
print("\n3. Knowledge Base Check:")
kb_files = [
    "kb/sdtmig_v3_4_complete/proto_define.json",
    "kb/sdtmig_v3_4_complete/domains_by_class.json",
    "kb/sdtmig_v3_4_complete/class_definitions.json"
]
for kb_file in kb_files:
    if Path(kb_file).exists():
        print(f"   ✓ {kb_file}")
    else:
        print(f"   ✗ {kb_file}")

# 4. Check required packages
print("\n4. Package Check:")
packages = [
    "transformers",
    "datasets", 
    "peft",
    "accelerate",
    "bitsandbytes"
]
for package in packages:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - Please install with: pip install {package}")

# 5. Model loading test
print("\n5. Model Loading Test:")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("   ✓ Can load models from Hugging Face")
except Exception as e:
    print(f"   ✗ Error loading from HF: {e}")

try:
    from modelscope import snapshot_download
    print("   ✓ ModelScope available")
except ImportError:
    print("   ℹ ModelScope not installed (optional)")

# 6. Training readiness
print("\n6. Training Readiness:")
issues = []

if not torch.cuda.is_available():
    issues.append("No GPU detected - training will be slow on CPU")

if not data_file.exists():
    issues.append("Training data not generated - run create_instruction_dataset.py")

try:
    import bitsandbytes as bnb
    if hasattr(bnb, 'nn'):
        print("   ✓ Bitsandbytes working")
    else:
        issues.append("Bitsandbytes may not support quantization")
except Exception as e:
    issues.append(f"Bitsandbytes issue: {e}")

if issues:
    print("   Issues found:")
    for issue in issues:
        print(f"     - {issue}")
else:
    print("   ✓ All checks passed - ready to train!")

print("\n" + "=" * 60)
print("\nTo start training, run one of:")
print("  ./run_training.sh          # Standard training") 
print("  ./run_training_modelscope.sh  # Using ModelScope")
print("  ./run_training_cpu.sh      # CPU-only mode")
print("\nFor help with issues, see TROUBLESHOOTING.md")