#!/usr/bin/env python3
"""
Minimal smoke test for the fine-tune loop using a tiny model.

- Uses `sshleifer/tiny-gpt2` (very small) by default to avoid heavy downloads.
- Trains for a handful of steps on a tiny subset of the prepared dataset.
- Does not require bitsandbytes or wandb.

Usage: python test_finetune_miniloop.py [--model sshleifer/tiny-gpt2]
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_subset(n: int = 32) -> list[dict]:
    data_file = Path("sdtm_instruction_tuning_package/data/alpaca_format.json")
    data = json.loads(data_file.read_text())
    return data[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2", help="HF model id to use")
    ap.add_argument("--steps", type=int, default=8, help="Max training steps")
    ap.add_argument("--out", default="sdtm_instruction_tuning_package/test_output_smoke", help="Output dir")
    args = ap.parse_args()

    print("Loading tiny dataset subset...")
    subset = load_subset(32)

    print(f"Loading model/tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    def preprocess(batch):
        texts = [
            f"{ins}\n{inp}\n{out}"
            for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]) 
        ]
        enc = tok(texts, truncation=True, padding="max_length", max_length=256)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = Dataset.from_list(subset)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)

    args_training = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        num_train_epochs=1,
        logging_steps=2,
        max_steps=args.steps,
        report_to="none",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )

    print("Starting tiny training...")
    out = trainer.train()
    print(f"Done. final loss={out.training_loss:.4f}")

    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"✓ Output created: {out_dir}")
    else:
        print("✗ Output dir missing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
