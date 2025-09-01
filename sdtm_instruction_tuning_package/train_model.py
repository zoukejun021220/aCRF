#!/usr/bin/env python3
"""
Training script for SDTM mapper instruction tuning
Optimized for cloud GPU training with QLoRA
"""

import os
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ModelScope for alternative model downloading
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
    logger.info("ModelScope is available for model downloading")
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logger.info("ModelScope not available, will use Hugging Face")


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-14B-Instruct",
        metadata={"help": "Model name or path"}
    )
    use_modelscope: bool = field(
        default=False,
        metadata={"help": "Download model from ModelScope instead of Hugging Face"}
    )
    modelscope_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "ModelScope model ID (if different from HF name)"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization"}
    )
    local_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Filesystem path to a pre-downloaded model (overrides model_name_or_path)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Use nested quantization for 4-bit models"}
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
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )
    use_metadata: bool = field(
        default=True,
        metadata={"help": "Use instruction_dataset.json with step metadata if available"}
    )
    step_weights: Optional[str] = field(
        default=None,
        metadata={"help": "JSON mapping of step->weight, e.g. {'domain_selection':1.0}"}
    )
    eval_stepwise: bool = field(
        default=True,
        metadata={"help": "Run evaluation per step and report losses"}
    )


@dataclass
class TrainingArgs(TrainingArguments):
    """Extended training arguments"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="paged_adamw_32bit")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for training"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )


class SDTMDataset:
    """Dataset class for SDTM instruction tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 use_metadata: bool = True, step_weights: Optional[Dict[str, float]] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.step_weights = step_weights or {}
        self.examples = self.load_data(data_path, use_metadata)
        # Build step id mapping
        steps = sorted({ex.get("step", "unknown") for ex in self.examples})
        self.step_to_id = {s: i for i, s in enumerate(steps)}
        
    def load_data(self, data_path: str, use_metadata: bool) -> List[Dict]:
        """Load instruction tuning data, prefer metadata-rich file when available"""
        base = Path(data_path)
        rich = base / "instruction_dataset.json"
        simple = base / "alpaca_format.json"
        if use_metadata and rich.exists():
            with open(rich) as f:
                data = json.load(f)
            # Normalize to include step
            normalized = []
            for ex in data:
                norm = {
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                    "output": ex.get("output", ""),
                    "step": (ex.get("metadata") or {}).get("step", "unknown")
                }
                normalized.append(norm)
            return normalized
        elif simple.exists():
            with open(simple) as f:
                data = json.load(f)
            # No step metadata available
            for ex in data:
                ex["step"] = "unknown"
            return data
        else:
            raise FileNotFoundError(f"Data file not found: {rich if use_metadata else simple}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as conversation
        if self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": example["instruction"]},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback format
            text = f"### Instruction:\n{example['instruction']}\n\n"
            text += f"### Input:\n{example['input']}\n\n"
            text += f"### Response:\n{example['output']}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Set labels (same as input_ids for causal LM)
        encodings["labels"] = encodings["input_ids"].clone()
        # Attach step info and precomputed sample weight
        step = example.get("step", "unknown")
        step_id = self.step_to_id.get(step, 0)
        weight = float(self.step_weights.get(step, 1.0))
        item = {k: v.squeeze() for k, v in encodings.items()}
        item["step_id"] = torch.tensor(step_id, dtype=torch.long)
        item["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
        return item


def download_model_from_modelscope(model_id: str, cache_dir: str = "./models") -> str:
    """Download model from ModelScope and return local path"""
    logger.info(f"Downloading model from ModelScope: {model_id}")
    
    # ModelScope model ID mapping for common models
    modelscope_mapping = {
        "Qwen/Qwen2.5-14B-Instruct": "qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct": "qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen/Qwen2.5-0.5B-Instruct",
    }
    
    # Use mapping or provided ModelScope ID
    ms_model_id = modelscope_mapping.get(model_id, model_id)
    
    # Download from ModelScope
    local_path = snapshot_download(
        ms_model_id,
        cache_dir=cache_dir,
        revision='master'
    )
    
    logger.info(f"Model downloaded to: {local_path}")
    return local_path


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer with quantization"""
    
    # Determine model path
    model_path = model_args.local_model_path or model_args.model_name_or_path
    
    # Download from ModelScope if requested
    if (not model_args.local_model_path) and model_args.use_modelscope and MODELSCOPE_AVAILABLE:
        modelscope_id = model_args.modelscope_model_id or model_args.model_name_or_path
        model_path = download_model_from_modelscope(modelscope_id)
    elif (not model_args.local_model_path) and model_args.use_modelscope and not MODELSCOPE_AVAILABLE:
        logger.warning("ModelScope requested but not available. Install with: pip install modelscope")
        logger.info("Falling back to Hugging Face")
    
    # Quantization config
    bnb_config = None
    if model_args.use_4bit:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    # Load config explicitly with trust_remote_code to handle newer model types
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load config with trust_remote_code: {e}. Proceeding without explicit config.")
        config = None

    def _load_from(source: str):
        return AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            config=config,
        )

    # Primary attempt
    try:
        model = _load_from(model_path)
    except Exception as e_first:
        logger.warning(f"Primary model load failed from {model_path}: {e_first}")
        # Fallback 1: if local path provided, try the HF repo id
        fallback_id = None
        if model_args.local_model_path:
            fallback_id = model_args.model_name_or_path
            logger.info(f"Falling back to repo id: {fallback_id}")
            try:
                model = _load_from(fallback_id)
            except Exception as e_second:
                logger.warning(f"Repo id load failed: {e_second}")
                # Fallback 2: snapshot download then load
                try:
                    from huggingface_hub import snapshot_download
                    logger.info("Downloading snapshot via Hugging Face Hub...")
                    local_snap = snapshot_download(repo_id=fallback_id, resume_download=True)
                    model_path = local_snap
                    model = _load_from(model_path)
                except Exception as e_third:
                    logger.warning(f"HF snapshot download failed: {e_third}")
                    if MODELSCOPE_AVAILABLE:
                        try:
                            logger.info("Falling back to ModelScope snapshot download...")
                            ms_id = model_args.modelscope_model_id or model_args.model_name_or_path
                            model_path = download_model_from_modelscope(ms_id)
                            model = _load_from(model_path)
                        except Exception as e_fourth:
                            raise RuntimeError(f"All model load fallbacks failed: {e_fourth}") from e_fourth
                    else:
                        raise RuntimeError(f"All model load fallbacks failed: {e_third}") from e_third
        else:
            # No local path; try ModelScope then raise
            if MODELSCOPE_AVAILABLE:
                try:
                    logger.info("Attempting ModelScope snapshot download...")
                    ms_id = model_args.modelscope_model_id or model_args.model_name_or_path
                    model_path = download_model_from_modelscope(ms_id)
                    model = _load_from(model_path)
                except Exception as e_ms:
                    raise RuntimeError(f"ModelScope load failed after primary: {e_ms}") from e_ms
            else:
                raise
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_lora(model, training_args: TrainingArgs):
    """Setup LoRA for efficient fine-tuning"""
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=training_args.lora_target_modules,
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


class SDTMTrainer(Trainer):
    """Custom trainer with SDTM-specific logging"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        """Compute loss with proper label masking"""
        labels = inputs.pop("labels")
        # Extract and remove custom fields before forward
        sample_weight = inputs.pop("sample_weight", None)
        _ = inputs.pop("step_id", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute token-level loss (HF convention uses -100 as ignore index)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # Reshape to [batch, seq]
        bsz = shift_labels.size(0)
        seqlen = shift_labels.size(1)
        token_loss = token_loss.view(bsz, seqlen)
        # Average per-sample over valid tokens
        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        per_sample = (token_loss * mask).sum(dim=1) / denom
        
        # Apply sample weights only during training
        if sample_weight is not None and model.training:
            # Broadcast/cast to correct device
            sample_weight = sample_weight.to(per_sample.device)
            per_sample = per_sample * sample_weight
        
        loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Log basic info
    logger.info(f"Training model: {model_args.model_name_or_path}")
    logger.info(f"Data path: {data_args.data_path}")
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup LoRA if enabled
    if training_args.use_lora:
        logger.info("Setting up LoRA...")
        model = setup_lora(model, training_args)
    
    # Parse step weights if provided
    step_weights: Dict[str, float] = {}
    if data_args.step_weights:
        try:
            step_weights = json.loads(data_args.step_weights)
        except Exception as e:
            logger.warning(f"Failed to parse step_weights JSON: {e}")
            step_weights = {}
    if not step_weights:
        # Defaults (can be overridden via --step_weights)
        step_weights = {
            "domain_selection": 1.0,
            "pattern_selection": 1.0,
            "testcode_selection": 1.2,
            "variable_selection": 1.2,
            "criterion_code_extraction": 1.1,
            "qnam_selection": 0.9,
            "unknown": 1.0,
        }
    logger.info(f"Step weights: {step_weights}")

    # Load dataset (with step metadata if available)
    logger.info("Loading dataset...")
    train_dataset = SDTMDataset(
        data_args.data_path,
        tokenizer,
        max_length=data_args.max_seq_length,
        use_metadata=data_args.use_metadata,
        step_weights=step_weights,
    )
    
    # Split into train/eval (90/10)
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, eval_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = SDTMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")

    # Optional: Stepwise evaluation on eval split (unweighted)
    if data_args.eval_stepwise and isinstance(eval_dataset, torch.utils.data.Subset):
        base_ds: SDTMDataset = eval_dataset.dataset  # type: ignore
        # Group indices by step
        step_to_indices: Dict[str, List[int]] = {}
        for idx in eval_dataset.indices:  # type: ignore[attr-defined]
            step = base_ds.examples[idx].get("step", "unknown")
            step_to_indices.setdefault(step, []).append(idx)
        logger.info("\nStepwise evaluation (unweighted):")
        for step, idxs in sorted(step_to_indices.items(), key=lambda x: x[0]):
            if not idxs:
                continue
            subset = torch.utils.data.Subset(base_ds, idxs)
            metrics = trainer.evaluate(eval_dataset=subset)
            logger.info(f"  {step}: eval_loss={metrics.get('eval_loss'):.4f}")


if __name__ == "__main__":
    main()
