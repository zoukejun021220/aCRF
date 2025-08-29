#!/usr/bin/env python3
"""
Test training without downloading - use GPT2 which is small and often cached
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing training pipeline without downloading models...")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Use GPT2 - it's small and often already cached
    model_name = "gpt2"
    
    # Load our training data
    with open("./data/alpaca_format.json") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    
    # Use subset for testing
    train_data = data[:100]
    eval_data = data[100:110]
    
    # Load tokenizer and model
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    logger.info(f"Model loaded: {model.num_parameters()/1e6:.1f}M parameters")
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # GPT2 specific
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    def preprocess(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"{examples['instruction'][i]}\n{examples['input'][i]}\n{examples['output'][i]}"
            texts.append(text)
        
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings
    
    train_dataset = Dataset.from_dict({
        'instruction': [ex['instruction'] for ex in train_data],
        'input': [ex['input'] for ex in train_data], 
        'output': [ex['output'] for ex in train_data]
    })
    
    eval_dataset = Dataset.from_dict({
        'instruction': [ex['instruction'] for ex in eval_data],
        'input': [ex['input'] for ex in eval_data],
        'output': [ex['output'] for ex in eval_data]
    })
    
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        logging_steps=5,
        learning_rate=5e-4,
        warmup_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        max_steps=30,  # Just 30 steps to test
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    
    # Test inference
    logger.info("\nTesting inference...")
    prompt = "You are helping select the appropriate SDTM domain. Select domain for: Blood pressure"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    
    logger.info("\n✅ Training pipeline is working correctly!")
    logger.info("Ready for cloud deployment with larger models.")

if __name__ == "__main__":
    main()