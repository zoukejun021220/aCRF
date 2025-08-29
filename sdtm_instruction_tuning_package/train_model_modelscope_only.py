#!/usr/bin/env python3
"""
Training script using ONLY ModelScope - no Hugging Face dependencies
For environments where Hugging Face is blocked or unavailable
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Set environment to use ModelScope mirror
os.environ['MODELSCOPE_CACHE'] = './models'

# Import ModelScope components
try:
    from modelscope import (
        Model, 
        Trainer as MSTrainer,
        TrainingArgs as MSTrainingArgs,
        snapshot_download,
        EpochBasedTrainer
    )
    from modelscope.models import Model
    from modelscope.trainers import build_trainer
    from modelscope.utils.config import Config
    from modelscope.msdatasets import MsDataset
except ImportError as e:
    print(f"Error importing ModelScope: {e}")
    print("\nPlease install ModelScope:")
    print("pip install modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_id: str = field(
        default="qwen/Qwen2.5-14B-Instruct",
        metadata={"help": "ModelScope model ID"}
    )
    model_revision: str = field(
        default="master",
        metadata={"help": "Model revision"}
    )
    cache_dir: str = field(
        default="./models",
        metadata={"help": "Model cache directory"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_path: str = field(
        metadata={"help": "Path to the training data"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


def download_model_from_modelscope(model_id: str, cache_dir: str = "./models") -> str:
    """Download model from ModelScope"""
    logger.info(f"Downloading model from ModelScope: {model_id}")
    
    local_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision='master'
    )
    
    logger.info(f"Model downloaded to: {local_path}")
    return local_path


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data"""
    data_file = Path(data_path) / "alpaca_format.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file) as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_data_for_modelscope(examples: List[Dict]) -> List[Dict]:
    """Format data for ModelScope training"""
    formatted_data = []
    
    for ex in examples:
        # Format as conversation
        text = f"### Instruction:\n{ex['instruction']}\n\n"
        text += f"### Input:\n{ex['input']}\n\n"
        text += f"### Response:\n{ex['output']}"
        
        formatted_data.append({
            "text": text,
            "instruction": ex['instruction'],
            "input": ex['input'],
            "output": ex['output']
        })
    
    return formatted_data


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_id", default="qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--cache_dir", default="./models")
    
    # Data arguments
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # Training arguments
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download model
    logger.info("Downloading model from ModelScope...")
    model_path = download_model_from_modelscope(args.model_id, args.cache_dir)
    
    # Load and format data
    logger.info("Loading training data...")
    raw_data = load_training_data(args.data_path)
    train_data = format_data_for_modelscope(raw_data)
    
    # Split data
    train_size = int(0.9 * len(train_data))
    train_dataset = train_data[:train_size]
    eval_dataset = train_data[train_size:]
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Save formatted data
    train_file = Path(args.output_dir) / "train.jsonl"
    eval_file = Path(args.output_dir) / "eval.jsonl"
    
    with open(train_file, 'w') as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(eval_file, 'w') as f:
        for item in eval_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Configure training
    cfg = Config({
        'framework': 'pytorch',
        'task': 'text-generation',
        'model': {
            'type': model_path,
        },
        'dataset': {
            'train': {
                'type': 'json',
                'files': [str(train_file)],
            },
            'val': {
                'type': 'json', 
                'files': [str(eval_file)],
            }
        },
        'preprocessor': {
            'type': 'text-gen-preprocessor',
            'max_length': args.max_seq_length,
        },
        'train': {
            'max_epochs': args.num_epochs,
            'batch_size_per_gpu': args.batch_size,
            'lr': args.learning_rate,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'save_strategy': 'steps',
            'save_steps': args.save_steps,
            'eval_strategy': 'steps',
            'eval_steps': args.eval_steps,
            'logging_steps': 10,
            'warmup_ratio': 0.03,
            'weight_decay': 0.01,
            'lr_scheduler_type': 'cosine',
            'work_dir': args.output_dir,
        }
    })
    
    # Build trainer
    logger.info("Initializing trainer...")
    trainer = build_trainer(cfg, default_args='text-generation')
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.output_dir}")
    
    # Instructions for using the model
    print("\nTo use the fine-tuned model:")
    print(f"model = Model.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()