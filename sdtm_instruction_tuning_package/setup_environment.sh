#!/bin/bash
# Setup script to install dependencies with proper versions

echo "Setting up SDTM Instruction Tuning Environment..."

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1-2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "No CUDA detected, installing CPU version"
    CUDA_VERSION="cpu"
fi

# Uninstall conflicting packages first
echo "Cleaning up conflicting packages..."
pip uninstall -y torch torchvision torchaudio transformers

# Install PyTorch based on CUDA version
echo "Installing PyTorch..."
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.7" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu117
else
    # CPU or unknown CUDA version
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements_fixed.txt

# Install ModelScope (optional, for China users)
echo "Installing ModelScope..."
pip install modelscope -U

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "try: from modelscope import snapshot_download; print('ModelScope: Available'); except: print('ModelScope: Not available')"

echo "Setup complete!"