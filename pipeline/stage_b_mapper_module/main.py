#!/usr/bin/env python3
"""
Stage B: Unified SDTM Mapper
Main entry point for the mapper module
"""

import argparse
import logging
from pathlib import Path

from core.mapper import UnifiedSDTMMapper

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified SDTM Mapper - Annotate CRF fields to SDTM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # File processing command
    file_parser = subparsers.add_parser('process-file', help='Process a single CRF JSON file')
    file_parser.add_argument('json_file', help='Path to CRF JSON file')
    file_parser.add_argument('--output', help='Output file for annotations')
    file_parser.add_argument('--kb-path', help='Path to KB directory')
    file_parser.add_argument('--proto-define', help='Path to proto_define.json')
    file_parser.add_argument('--model', default="Qwen/Qwen2.5-14B-Instruct", help='LLM model')
    file_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    file_parser.add_argument('--debug-output', help='Debug output file')
    
    # Directory processing command
    dir_parser = subparsers.add_parser('process-dir', help='Process directory of CRF JSON files')
    dir_parser.add_argument('directory', help='Directory containing page JSON files')
    dir_parser.add_argument('--output-dir', help='Output directory for results')
    dir_parser.add_argument('--kb-path', help='Path to KB directory')
    dir_parser.add_argument('--proto-define', help='Path to proto_define.json')
    dir_parser.add_argument('--model', default="Qwen/Qwen2.5-14B-Instruct", help='LLM model')
    dir_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize mapper
    mapper = UnifiedSDTMMapper(
        model_name=args.model,
        proto_define_path=args.proto_define,
        kb_path=args.kb_path,
        debug=args.debug
    )
    
    try:
        if args.command == 'process-file':
            # Process single file
            results = mapper.process_crf_json_file(Path(args.json_file))
            
            # Save results
            if args.output:
                output_path = Path(args.output)
            else:
                input_path = Path(args.json_file)
                output_dir = Path("annotations")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{input_path.stem}_annotated.json"
                
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"\nAnnotation complete:")
            print(f"- Total questions: {results['summary']['total_questions']}")
            print(f"- Annotated: {results['summary']['annotated']}")
            print(f"- Skipped: {results['summary']['skipped']}")
            print(f"- Valid: {results['summary']['valid']}")
            print(f"- Output: {output_path}")
            
            # Save debug if requested
            if args.debug and args.debug_output:
                mapper.save_debug_json(args.debug_output, results)
                print(f"- Debug output: {args.debug_output}")
                
        elif args.command == 'process-dir':
            # Process directory
            output_dir = Path(args.output_dir) if args.output_dir else None
            results = mapper.process_crf_json_directory(Path(args.directory), output_dir)
            
            print(f"\nDirectory processing complete:")
            print(f"- Total pages: {results['summary']['total_pages']}")
            print(f"- Total questions: {results['summary']['total_questions']}")
            print(f"- Annotated: {results['summary']['total_annotated']}")
            print(f"- Valid: {results['summary']['total_valid']}")
            
            if results['summary'].get('domains'):
                print("\nDomains found:")
                for domain, count in sorted(results['summary']['domains'].items()):
                    print(f"  {domain}: {count}")
                    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    main()