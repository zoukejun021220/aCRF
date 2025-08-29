#!/bin/bash
# Fix bitsandbytes GPU support issues

echo "Fixing bitsandbytes installation..."

# First, check if we have CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected"
    
    # Uninstall CPU-only version
    pip uninstall -y bitsandbytes
    
    # Install GPU version with proper CUDA support
    # Try multiple approaches
    
    # Method 1: Install pre-built wheel
    pip install bitsandbytes --upgrade --index-url https://pypi.org/simple/
    
    # If that fails, try building from source
    if [ $? -ne 0 ]; then
        echo "Trying alternative installation..."
        pip install bitsandbytes-cuda117  # For CUDA 11.7
    fi
    
    # Install triton for GPU support
    pip install triton
    
else
    echo "No CUDA detected - using CPU-only mode"
    
    # For CPU-only, we need to disable 4-bit quantization
    echo "WARNING: Running on CPU - 4-bit quantization will be disabled"
    
    # Create a modified training script for CPU
    cp train_model.py train_model_cpu.py
    
    # Modify to disable 4-bit by default
    sed -i 's/use_4bit: bool = field(default=True/use_4bit: bool = field(default=False/g' train_model_cpu.py
fi

echo "Testing imports..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import bitsandbytes; print('Bitsandbytes: OK')" || echo "Bitsandbytes: Failed"

echo "Fix complete!"