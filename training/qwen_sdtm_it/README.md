# Qwen2.5-14B Instruction Tuning for SDTM Mapping

This folder contains a minimal, cloud-friendly pipeline to fine‑tune Qwen2.5‑14B‑Instruct on your CRF → SDTM mapping task using the ground‑truth in `reference_with_sections`.

## Overview
- Source: `reference_with_sections/page_XXX_result.json` (augmented with `form` and `section`).
- Input (user): Form, Section, and Question text.
- Target (assistant): The `annotation` string from the reference (kept as‑is).
- Format: Chat messages (system + user + assistant) with Qwen chat template.
- Training: PEFT LoRA on top of Qwen2.5‑14B‑Instruct, 4‑bit quantization for memory efficiency.

## Quick Start

1) Prepare a Python 3.10+ env and install deps:
- pip install -U transformers datasets peft bitsandbytes accelerate

2) Build the training/validation dataset:
- python prepare_dataset.py \
  --reference-dir ../reference_with_sections \
  --output-dir data \
  --val-ratio 0.05

This writes `data/train.jsonl` and `data/val.jsonl` (each line is a `messages` array).

3) Launch training (LoRA + 4‑bit):
- accelerate launch train.py \
  --model "Qwen/Qwen2.5-14B-Instruct" \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --output-dir outputs/qwen2_5_14b_sdtm_lora \
  --batch-size 2 --grad-accum 8 \
  --lr 2e-4 --epochs 2 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --bf16 --save-steps 500

Notes:
- Use smaller batches/longer grad accumulation if RAM is tight.
- For inference, load base model + LoRA adapters saved in `output-dir`.

## Data Assumptions
- Each reference file has shape:
  - `annotations`: array of { text, annotation, form?, section? }.
- If `form`/`section` are absent, the pipeline leaves them blank.

## Inference Snippet
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-14B-Instruct"
out = "outputs/qwen2_5_14b_sdtm_lora"

tok = AutoTokenizer.from_pretrained(base)
mdl = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, out)

messages = [
  {"role": "system", "content": "You map CRF questions to SDTM annotations."},
  {"role": "user", "content": "Form: Baseline\nSection: Inclusion Criteria\nQuestion: Age ≥ 18 years (INC1)\nReturn SDTM annotation:"}
]
inputs = tok.apply_chat_template(messages, return_tensors="pt").to(mdl.device)
out_ids = mdl.generate(inputs, max_new_tokens=256)
print(tok.decode(out_ids[0], skip_special_tokens=True))
```

## Cloud Tips
- Set `HF_HOME`/`TRANSFORMERS_CACHE` to a persistent volume to avoid re‑downloads.
- Prefer BF16 on A100/H100; otherwise FP16.
- Use `accelerate config` to set multi‑GPU / DeepSpeed if needed.
