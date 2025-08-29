# SDTM Instruction Tuning - Complete Training Guide

## 📦 Package Information
- **Package**: `sdtm_instruction_tuning_final.tar.gz` (6.1MB)
- **Training Data**: 658 examples (already included)
- **Models Supported**: Qwen 0.5B to 14B

## 🚀 Quick Start (3 Steps)

### 1. Extract Package
```bash
tar -xzf sdtm_instruction_tuning_final.tar.gz
cd sdtm_instruction_tuning_package
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
```bash
# For GPU
./run_training.sh

# For CPU
./run_training_cpu.sh

# For China/Asia (uses ModelScope)
./run_training_modelscope.sh
```

## 💻 Detailed Instructions by Environment

### For Cloud GPU (Recommended)

#### AWS EC2
```bash
# Launch instance: p3.2xlarge (V100) or g5.2xlarge (A10G)
# SSH into instance
wget <your_package_url>/sdtm_instruction_tuning_final.tar.gz
tar -xzf sdtm_instruction_tuning_final.tar.gz
cd sdtm_instruction_tuning_package
./setup_environment.sh
./run_training.sh
```

#### Google Colab
```python
# Upload package to Google Drive first
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/sdtm_instruction_tuning_final.tar.gz .
!tar -xzf sdtm_instruction_tuning_final.tar.gz
!cd sdtm_instruction_tuning_package && ./run_training.sh
```

### For CPU-Only (Your Current Setup)

Since you don't have GPU, use the CPU script:

```bash
cd sdtm_instruction_tuning_package

# Fix bitsandbytes issue
pip uninstall -y bitsandbytes
pip install bitsandbytes-cpu

# Run CPU training with smaller model
python train_model.py \
  --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
  --data_path ./data \
  --output_dir ./output \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --do_train \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --no-use_4bit \
  --max_seq_length 512 \
  --logging_steps 10 \
  --save_steps 100 \
  --fp16 False
```

## 🎯 Training Parameters Explained

| Parameter | GPU Value | CPU Value | Purpose |
|-----------|-----------|-----------|---------|
| model_name_or_path | Qwen/Qwen2.5-14B | Qwen/Qwen2.5-0.5B | Model size |
| batch_size | 4 | 1 | Samples per step |
| gradient_accumulation | 8 | 16 | Effective batch = batch_size × accumulation |
| use_4bit | True | False | 4-bit quantization (GPU only) |
| max_seq_length | 2048 | 512 | Max token length |
| fp16 | True | False | Half precision (GPU only) |

## 📊 Monitor Training Progress

### Real-time Monitoring
```bash
# Watch training loss
tail -f output/trainer_state.json | grep loss

# For GPU: Monitor memory
watch -n1 nvidia-smi
```

### Expected Training Times
- **GPU (A100)**: ~2-3 hours for 3 epochs
- **GPU (T4)**: ~4-6 hours for 3 epochs  
- **CPU**: ~24-48 hours for 1 epoch (not recommended)

## 🔧 Troubleshooting

### "No GPU detected"
You're on CPU. Use smaller model and CPU settings above.

### "Bitsandbytes issue"
```bash
# For CPU
pip uninstall -y bitsandbytes
pip install bitsandbytes-cpu

# Or disable quantization
# Add --no-use_4bit to training command
```

### "Out of Memory"
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Reduce sequence length
--max_seq_length 256

# Use smaller model
--model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct"
```

### "Model download too slow"
Use ModelScope (faster in Asia):
```bash
./run_training_modelscope.sh
```

## ✅ Verify Training Success

After training completes:

```bash
# Check if model saved
ls -la output/

# Test the model
python inference_test.py

# Expected output:
# Domain: VS (for blood pressure)
# Pattern: findings
# Annotation: VSORRES / VSORRESU when VSTESTCD = SYSBP
```

## 🌐 Alternative: Use Cloud Services

If CPU training is too slow:

### 1. Google Colab (Free GPU)
- Upload package to Google Drive
- Open Colab notebook
- Select Runtime > Change runtime type > GPU
- Run commands above

### 2. Kaggle Notebooks (Free GPU)
- Upload package as dataset
- Create new notebook with GPU
- Run training

### 3. Paperspace Gradient (Low cost)
- $8/month for basic GPU
- Pre-configured ML environment
- Run training directly

## 📝 Final Notes

- **CPU Training**: Possible but very slow. Use smallest model (0.5B)
- **Recommended**: Use any cloud GPU service for faster training
- **Data is Ready**: All 658 training examples are included
- **Support**: Check TROUBLESHOOTING.md for more issues

Ready to train? Just run the appropriate script for your environment!