# SDTM Mapper Instruction Tuning Package

This package contains everything needed to fine-tune a language model for SDTM annotation using instruction tuning.

## Overview

The SDTM mapper uses a step-by-step annotation process:
1. **Domain Selection** - Select the appropriate SDTM domain (AE, VS, IE, etc.)
2. **Pattern Selection** - Choose the mapping pattern (direct, findings, conditional, supplemental)
3. **Variable/Value Selection** - Map to specific SDTM variables based on the pattern

This instruction tuning approach trains the model to follow this exact process.

## Package Structure

```
sdtm_instruction_tuning_package/
├── data/                      # Training data directory
├── kb/                        # Knowledge base files
│   └── sdtmig_v3_4_complete/  # SDTM KB resources
├── scripts/                   # Helper scripts
├── output/                    # Model checkpoints
├── cache/                     # Model cache
├── logs/                      # Training logs
├── create_instruction_dataset.py  # Dataset creation script
├── train_model.py             # Training script
├── inference_test.py          # Inference testing
├── run_training.sh            # Training launcher
├── requirements.txt           # Python dependencies
├── training_config.json       # Training configuration
└── README.md                  # This file
```

## Setup Instructions

### 1. Environment Setup (Cloud GPU)

```bash
# Create conda environment
conda create -n sdtm-tuning python=3.10
conda activate sdtm-tuning

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data

First, run the package preparation script:
```bash
python prepare_package.py
```

Then copy your data:
```bash
# Copy reference annotations
cp -r /path/to/reference/all_results ./data/reference/

# Copy CRF JSON files
cp -r /path/to/crf_json/blankCRF_corrected ./data/crf_json/
```

### 3. Generate Instruction Dataset

```bash
python create_instruction_dataset.py
```

This will create:
- `data/instruction_dataset.json` - Full dataset in JSON format
- `data/instruction_dataset.jsonl` - JSONL format for streaming
- `data/alpaca_format.json` - Alpaca-style format for training
- `data/dataset_metadata.json` - Dataset statistics

### 4. Configure Training

Edit `training_config.json` to adjust:
- Model selection (default: Qwen/Qwen2.5-14B-Instruct)
- Training hyperparameters
- LoRA configuration
- Hardware settings

### 5. Run Training

For single GPU with Hugging Face:
```bash
./run_training.sh
```

For single GPU with ModelScope (recommended for users in China):
```bash
./run_training_modelscope.sh
```

For multi-GPU (adjust CUDA_VISIBLE_DEVICES):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train_model.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --use_modelscope \  # Add this flag for ModelScope
    --data_path ./data \
    --output_dir ./output \
    [other arguments from run_training.sh]
```

#### Using ModelScope

ModelScope provides faster model downloads for users in regions with limited Hugging Face access:

1. Install ModelScope:
```bash
pip install modelscope
```

2. Use the ModelScope training script:
```bash
./run_training_modelscope.sh
```

3. Or manually specify ModelScope in training:
```bash
python train_model.py \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
    --use_modelscope \
    --modelscope_model_id "qwen/Qwen2.5-14B-Instruct" \  # Optional
    --data_path ./data \
    --output_dir ./output
```

Supported models on ModelScope:
- `qwen/Qwen2.5-14B-Instruct`
- `qwen/Qwen2.5-7B-Instruct`
- `qwen/Qwen2.5-3B-Instruct`
- `qwen/Qwen2.5-1.5B-Instruct`
- `qwen/Qwen2.5-0.5B-Instruct`

## Cloud GPU Recommendations

### AWS EC2
- Instance: p3.8xlarge (4x V100 16GB) or g5.12xlarge (4x A10G 24GB)
- Storage: 200GB+ EBS volume
- AMI: Deep Learning AMI with PyTorch

### Google Cloud
- Instance: n1-standard-8 with 4x T4 or A100
- Storage: 200GB+ persistent disk
- Image: Deep Learning VM with PyTorch

### Azure
- Instance: Standard_NC6s_v3 (V100) or Standard_NC24ads_A100_v4
- Storage: 200GB+ managed disk

### Memory Optimization
The training uses QLoRA (4-bit quantization) to reduce memory usage:
- 14B model fits in ~16GB VRAM with 4-bit quantization
- Effective batch size: 32 (4 * 8 gradient accumulation)
- Max sequence length: 2048 tokens

## Training Monitoring

### Weights & Biases
Training logs to W&B by default. Set up:
```bash
wandb login
export WANDB_PROJECT="sdtm-mapper-finetuning"
```

### TensorBoard
```bash
tensorboard --logdir ./output
```

## Testing the Model

After training, test with:
```bash
python inference_test.py
```

This will:
1. Load the fine-tuned model
2. Test on sample CRF fields
3. Show step-by-step annotation results

## Integration with SDTM Mapper

To use the fine-tuned model in the main mapper:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./output/checkpoint-final"
)

# Use in UnifiedSDTMMapper
mapper = UnifiedSDTMMapper(
    model=model,  # Pass fine-tuned model
    kb_path="./kb/sdtmig_v3_4_complete"
)
```

## Dataset Format

The instruction dataset follows this format:

```json
{
  "instruction": "System prompt describing the task",
  "input": "User prompt with field information",
  "output": "Expected model response",
  "metadata": {
    "step": "domain_selection|pattern_selection|variable_selection",
    "domain": "AE|VS|IE|...",
    "pattern": "direct|findings|conditional|supplemental",
    "field_label": "Original field label"
  }
}
```

## Troubleshooting

### Out of Memory
- Reduce batch_size in run_training.sh
- Increase gradient_accumulation_steps
- Use smaller model (7B instead of 14B)
- Enable gradient checkpointing

### Slow Training
- Use mixed precision (fp16=True)
- Reduce max_seq_length if possible
- Use DeepSpeed for multi-GPU

### Poor Results
- Increase num_epochs
- Adjust learning rate
- Add more training examples
- Check data quality

## Additional Resources

- [SDTM Implementation Guide](https://www.cdisc.org/standards/foundational/sdtmig)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Support

For issues or questions:
1. Check the logs in `./logs/`
2. Review dataset statistics in `data/dataset_metadata.json`
3. Ensure KB files are complete in `kb/sdtmig_v3_4_complete/`