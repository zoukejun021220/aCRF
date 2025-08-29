#!/usr/bin/env python3
"""
Stage A: CRF Digitizer using Qwen2.5-VL-7B
Main entry point for the digitizer module
"""

import argparse
import logging
import torch
from pathlib import Path

from core.digitizer import CRFDigitizer

logger = logging.getLogger(__name__)


def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description="Stage A: CRF Digitizer")
    parser.add_argument("pdf_path", type=str, help="Path to CRF PDF")
    parser.add_argument("--doc-id", type=str, help="Document ID (default: filename)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu)")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--fallback-only", action="store_true",
                       help="Use only fallback extraction (no vision model)")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
        args.no_4bit = True  # Disable quantization on CPU
    
    try:
        # Initialize digitizer
        digitizer = CRFDigitizer(
            model_path=args.model,
            device=args.device,
            use_4bit=not args.no_4bit,
            fallback_only=args.fallback_only
        )
        
        # Process PDF
        result = digitizer.process_pdf(
            pdf_path=args.pdf_path,
            doc_id=args.doc_id
        )
        
        print(f"\nDigitization Summary:")
        print(f"- Document: {result['doc_id']}")
        print(f"- Pages: {result['num_pages']}")
        print(f"- Total items: {result['total_items']}")
        print(f"- Item breakdown: {result['item_breakdown']}")
        print(f"- Output: {Path(result['page_files'][0]).parent}")
        print(f"\nTiming:")
        print(f"- Total time: {result['processing_time']['total_seconds']:.2f} seconds")
        print(f"- Average per page: {result['processing_time']['avg_page_seconds']:.2f} seconds")
        print(f"- Processing speed: {result['processing_time']['pages_per_minute']:.1f} pages/minute")
        
    except Exception as e:
        logger.error(f"Error during digitization: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    main()