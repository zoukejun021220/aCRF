# Troubleshooting Guide

## Common Issues and Solutions

### 1. PyTorch/Torchvision Compatibility Error

**Error**: `RuntimeError: operator torchvision::nms does not exist`

**Solution**:
```bash
# Run the setup script
./setup_environment.sh

# Or manually fix:
pip uninstall -y torch torchvision transformers
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2
```

### 2. Hugging Face Not Available

**Error**: `ModuleNotFoundError: Could not import module 'Trainer'`

**Solutions**:

#### Option A: Use ModelScope Only
```bash
# Install ModelScope
pip install modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# Use the ModelScope-only training script
python train_model_modelscope_only.py \
    --model_id qwen/Qwen2.5-14B-Instruct \
    --data_path ./data \
    --output_dir ./output
```

#### Option B: Fix Transformers Installation
```bash
# Clean install
pip uninstall -y transformers accelerate peft
pip install transformers==4.36.2 accelerate==0.25.0 peft==0.7.1
```

### 3. CUDA Version Mismatch

**Error**: CUDA runtime error

**Solution**:
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Out of Memory (OOM)

**Error**: `torch.cuda.OutOfMemoryError`

**Solutions**:
```bash
# Reduce batch size
--per_device_train_batch_size 2

# Increase gradient accumulation
--gradient_accumulation_steps 16

# Use smaller model
--model_name_or_path Qwen/Qwen2.5-7B-Instruct

# Enable gradient checkpointing
--gradient_checkpointing True

# Use CPU offloading
--optim adamw_bnb_8bit
```

### 5. ModelScope Connection Issues

**Error**: Connection timeout to ModelScope

**Solutions**:
```bash
# Use mirror
export MODELSCOPE_CACHE=./models
export MODELSCOPE_HUB_URL=https://modelscope.cn/api/v1/models

# Or download manually first
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-14B-Instruct', cache_dir='./models')
```

### 6. Missing Dependencies

**Error**: ImportError for various packages

**Solution**:
```bash
# Use the fixed requirements
pip install -r requirements_fixed.txt

# Or install minimal set
pip install torch==2.1.0
pip install transformers==4.36.2
pip install datasets accelerate peft
pip install bitsandbytes sentencepiece
```

### 7. Environment Variables

Set these before training:
```bash
# For Hugging Face issues
export HF_HOME=./cache
export TRANSFORMERS_OFFLINE=1

# For ModelScope
export MODELSCOPE_CACHE=./models

# For CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 8. Quick Test

Test your environment:
```python
# test_env.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    print("Transformers not available")

try:
    from modelscope import snapshot_download
    print("ModelScope: Available")
except:
    print("ModelScope: Not available")

try:
    import peft
    print(f"PEFT: {peft.__version__}")
except:
    print("PEFT not available")
```

### 9. Alternative: Docker Setup

If all else fails, use Docker:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements_fixed.txt .
RUN pip install -r requirements_fixed.txt

COPY . .

CMD ["./run_training.sh"]
```

Build and run:
```bash
docker build -t sdtm-training .
docker run --gpus all -v $(pwd)/data:/app/data sdtm-training
```