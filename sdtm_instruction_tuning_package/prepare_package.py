#!/usr/bin/env python3
"""
Prepare the complete instruction tuning package
Copies all necessary files and creates the package structure
"""

import os
import shutil
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_package_structure(package_dir: str):
    """Create the package directory structure"""
    package_path = Path(package_dir)
    
    # Create directories
    directories = [
        "data",
        "kb",
        "kb/sdtmig_v3_4_complete", 
        "scripts",
        "output",
        "cache",
        "logs"
    ]
    
    for dir_name in directories:
        (package_path / dir_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_name}")


def copy_kb_files(source_kb: str, target_kb: str):
    """Copy knowledge base files"""
    source_path = Path(source_kb)
    target_path = Path(target_kb)
    
    # Essential KB files
    kb_files = [
        "class_definitions.json",
        "domains_by_class.json",
        "proto_define.json",
        "pattern_definitions.json",
        "cdisc_ct.json",
        "variables_all.json",
        "common_ct_terms.json",
        "qrs_instruments.json"
    ]
    
    copied_count = 0
    for kb_file in kb_files:
        source_file = source_path / kb_file
        if source_file.exists():
            shutil.copy2(source_file, target_path / kb_file)
            logger.info(f"Copied KB file: {kb_file}")
            copied_count += 1
        else:
            logger.warning(f"KB file not found: {kb_file}")
    
    logger.info(f"Copied {copied_count} KB files")


def create_dataset_script(package_path: Path):
    """Create a script to generate the dataset"""
    script_content = '''#!/usr/bin/env python3
"""
Generate instruction tuning dataset from reference annotations
Run this script after setting up the package
"""

import sys
sys.path.append('..')
from create_instruction_dataset import main

if __name__ == "__main__":
    print("Generating instruction tuning dataset...")
    main()
    print("Dataset generation completed!")
'''
    
    script_path = package_path / "scripts" / "generate_dataset.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)


def create_config_file(package_path: Path):
    """Create configuration file for training"""
    config = {
        "model_config": {
            "base_model": "Qwen/Qwen3-4B-Instruct",
            "use_4bit": True,
            "quantization": {
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "use_nested_quant": False
            }
        },
        "training_config": {
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "max_seq_length": 2048,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "save_steps": 100
        },
        "lora_config": {
            "r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },
        "data_config": {
            "train_split": 0.9,
            "eval_split": 0.1,
            "preprocessing_num_workers": 4
        }
    }
    
    config_path = package_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Created training configuration file")


def main():
    # Paths resolved relative to repo root (no hardcoded host-specific paths)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    package_dir = os.path.join(repo_root.as_posix(), "sdtm_instruction_tuning_package")
    kb_source = os.path.join(repo_root.as_posix(), "kb/sdtmig_v3_4_complete")
    
    logger.info(f"Creating package in: {package_dir}")
    
    # Create package structure
    create_package_structure(package_dir)
    
    # Copy KB files
    kb_target = os.path.join(package_dir, "kb/sdtmig_v3_4_complete")
    copy_kb_files(kb_source, kb_target)
    
    # Create helper scripts
    package_path = Path(package_dir)
    create_dataset_script(package_path)
    create_config_file(package_path)
    
    # Copy main scripts (already created)
    logger.info("Package preparation completed!")
    logger.info(f"Package location: {package_dir}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Copy reference data to data/reference/")
    print("2. Copy CRF JSON files to data/crf_json/")
    print("3. Run: python scripts/generate_dataset.py")
    print("4. Run: ./run_training.sh")


if __name__ == "__main__":
    main()
