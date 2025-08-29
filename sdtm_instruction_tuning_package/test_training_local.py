#!/usr/bin/env python3
"""
Quick test of training locally with minimal settings
"""

import json
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training():
    # Use a very small model for testing
    model_name = "microsoft/phi-2"  # 2.7B model, much smaller than Qwen
    
    logger.info(f"Testing with model: {model_name}")
    
    # Load a few examples from our dataset
    data_path = Path("./data/alpaca_format.json")
    with open(data_path) as f:
        all_data = json.load(f)
    
    # Use only first 10 examples for quick test
    test_data = all_data[:10]
    
    logger.info(f"Using {len(test_data)} examples for test")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    def preprocess_function(examples):
        texts = []
        for ex in examples["data"if "data" in examples else "text"]:
            if isinstance(ex, dict):
                text = f"### Instruction:\n{ex['instruction']}\n\n"
                text += f"### Input:\n{ex['input']}\n\n"
                text += f"### Response:\n{ex['output']}"
            else:
                text = ex
            texts.append(text)
        
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Create dataset
    dataset = Dataset.from_dict({"data": test_data})
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split for train/eval
    train_dataset = tokenized_dataset.select(range(8))
    eval_dataset = tokenized_dataset.select(range(8, 10))
    
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["Wqkv", "fc1", "fc2"],  # phi-2 specific modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments - very minimal
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        logging_steps=1,
        save_steps=5,
        eval_steps=5,
        evaluation_strategy="steps",
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        max_steps=10,  # Only 10 steps for testing
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training test...")
    
    # Train for just a few steps
    trainer.train()
    
    logger.info("Test completed successfully!")
    
    # Test inference
    model.eval()
    test_text = "### Instruction:\nYou are helping select SDTM domain.\n\n### Input:\nSelect domain for: Blood Pressure\n\n### Response:\n"
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation: {response}")

if __name__ == "__main__":
    test_training()