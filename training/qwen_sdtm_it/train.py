#!/usr/bin/env python3
"""
LoRA Supervised Fine-Tuning for Qwen2.5-14B-Instruct on SDTM mapping.

Usage (single GPU):
  accelerate launch train.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --output-dir outputs/qwen2_5_14b_sdtm_lora \
    --batch-size 2 --grad-accum 8 --epochs 2 --lr 2e-4 --bf16
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_messages_dataset(path: str):
    # JSONL with a key `messages` (list of {role, content}) per line
    return load_dataset("json", data_files=path, split="train")


def to_chat_tokens(tokenizer, example, add_generation_prompt=False):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return {"text": text}


def tokenize(tokenizer, example, max_length=2048):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    out["labels"] = out["input_ids"].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.set_defaults(load_in_4bit=True)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model…")
    # Be robust in cloud: if bitsandbytes missing, disable 4-bit
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); disabling 4-bit load.")
            args.load_in_4bit = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch.float16,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    print("Loading datasets…")
    train_ds = load_messages_dataset(args.train_file)
    val_ds = load_messages_dataset(args.val_file)

    train_ds = train_ds.map(lambda e: to_chat_tokens(tokenizer, e), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda e: to_chat_tokens(tokenizer, e), remove_columns=val_ds.column_names)

    train_ds = train_ds.map(lambda e: tokenize(tokenizer, e, args.max_length))
    val_ds = val_ds.map(lambda e: tokenize(tokenizer, e, args.max_length))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
