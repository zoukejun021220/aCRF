#!/usr/bin/env python3
"""
Sanity-check the fine-tuning pipeline environment and data.

Runs lightweight checks without downloading large models or training.
Usage: python test_finetune_setup.py
"""

import importlib
import json
from pathlib import Path


def check_import(mod: str) -> tuple[bool, str]:
    try:
        importlib.import_module(mod)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def main() -> int:
    print("Checking fine-tune dependencies...")
    mods = [
        "transformers",
        "datasets",
        "peft",
        "torch",
    ]

    results = {m: check_import(m) for m in mods}
    for m, (ok, msg) in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {m}: {msg if not ok else 'importable'}")

    # bitsandbytes and wandb are optional; warn if missing
    for opt in ("bitsandbytes", "wandb"):
        ok, msg = check_import(opt)
        status = "✓" if ok else "⚠"
        print(f"  {status} {opt}: {msg if not ok else 'importable'}")

    # Verify training data exists and is non-empty
    data_path = Path("sdtm_instruction_tuning_package/data/alpaca_format.json")
    if data_path.exists():
        try:
            data = json.loads(data_path.read_text())
            print(f"\n✓ Found training data: {data_path} ({len(data)} examples)")
            if not data:
                print("⚠ Dataset is empty — regenerate with create_instruction_dataset.py")
        except Exception as e:
            print(f"✗ Failed to read dataset: {e}")
    else:
        print(f"\n✗ Training data not found: {data_path}")
        print("  Generate it with: python sdtm_instruction_tuning_package/create_instruction_dataset.py")

    # Quick GPU/CPU note (no assertions)
    try:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nRuntime device: {device}")
        if device == "cuda":
            print(f"GPU count: {torch.cuda.device_count()}")
    except Exception:
        pass

    print("\nSetup check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

