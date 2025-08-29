#!/usr/bin/env python3
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
