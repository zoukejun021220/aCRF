#!/usr/bin/env python3
"""
Wrapper test to exercise train_model.py with a tiny model.

This avoids optional deps by stubbing missing modules (wandb, bitsandbytes)
and runs just a couple of steps using a very small model.

Usage: python test_finetune_train_model_wrapper.py
"""

import os
import runpy
import sys
import types
from pathlib import Path


def stub_optional_modules():
    # Provide lightweight stubs only if modules are missing
    import importlib.util
    if importlib.util.find_spec("wandb") is None:
        sys.modules["wandb"] = types.SimpleNamespace(
            init=lambda *a, **k: None,
            log=lambda *a, **k: None,
            finish=lambda *a, **k: None,
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        mod = types.ModuleType("bitsandbytes")
        # A minimal spec to satisfy find_spec checks in transformers
        import importlib.machinery as _mach
        mod.__spec__ = _mach.ModuleSpec("bitsandbytes", loader=None)
        mod.__version__ = "stub"
        sys.modules["bitsandbytes"] = mod


def main() -> int:
    # Use tiny model to keep runtime small
    model = "sshleifer/tiny-gpt2"
    data_path = "sdtm_instruction_tuning_package/data"
    out_dir = "sdtm_instruction_tuning_package/test_cli_output"

    # Prepare argv for train_model.py (disable 4bit; disable LoRA to avoid PEFT if missing)
    sys.argv = [
        sys.executable,
        "--model_name_or_path", model,
        "--data_path", data_path,
        "--output_dir", out_dir,
        "--use_4bit", "False",
        "--use_lora", "False",
        "--per_device_train_batch_size", "1",
        "--learning_rate", "5e-4",
        "--max_steps", "2",
        "--logging_steps", "1",
        "--report_to", "none",
        "--overwrite_output_dir",
    ]

    # Inject stubs and run the script as __main__
    stub_optional_modules()
    # Force CPU to avoid GPU-specific tiny model issues in CI/smoke tests
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    script = Path("sdtm_instruction_tuning_package/train_model.py").as_posix()
    print(f"Running train_model.py with tiny model: {model}")
    runpy.run_path(script, run_name="__main__")
    print("Completed tiny CLI run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
