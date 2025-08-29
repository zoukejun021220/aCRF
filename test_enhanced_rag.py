#!/usr/bin/env python3
"""
Test the Enhanced RAG system with sample CRF questions
Usage: python test_enhanced_rag.py [--page page_num] [--kb-path path/to/kb]
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add pipeline to path
import sys
sys.path.append(str(Path(__file__).parent / "pipeline"))

# Direct import to avoid loading the full mapper module
from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem, RAGCandidate

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_crf_questions(json_path: Path) -> List[Dict[str, Any]]:
    """Load questions from CRF JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    items = data.get("items", [])
    
    current_question = None
    current_inputs = []
    
    for item in items:
        if item.get("tag") == "<Q>":
            # Save previous question
            if current_question:
                questions.append({
                    "question": current_question,
                    "inputs": current_inputs
                })
            current_question = item
            current_inputs = []
        elif item.get("tag") == "<INPUT>":
            current_inputs.append(item)
    
    # Don't forget last question
    if current_question:
        questions.append({
            "question": current_question,
            "inputs": current_inputs
        })
    
    return questions


def format_rag_result(candidate: RAGCandidate) -> str:
    """Format a RAG candidate for display"""
    lines = []
    lines.append(f"  Score: {candidate.score:.3f}")
    lines.append(f"  Domain: {candidate.domain.domain_code} - {candidate.domain.domain_name}")
    lines.append(f"  Variable: {candidate.variable.var_name} ({candidate.variable.label})")
    lines.append(f"    Role: {candidate.variable.role}")
    lines.append(f"    Definition: {candidate.variable.definition[:100]}...")
    
    if candidate.ct_terms:
        lines.append("  Controlled Terms:")
        for ct in candidate.ct_terms[:3]:  # Show top 3
            lines.append(f"    - {ct.submission_value}: {ct.preferred_term}")
    
    lines.append("  Score Breakdown:")
    for component, score in candidate.score_breakdown.items():
        lines.append(f"    {component}: {score:.3f}")
    
    return "\n".join(lines)


def test_sample_questions(rag_system: EnhancedRAGSystem):
    """Test with some sample questions"""
    print("\n" + "="*80)
    print("TESTING WITH SAMPLE QUESTIONS")
    print("="*80 + "\n")
    
    sample_questions = [
        {
            "text": "Date of adverse event onset",
            "options": [],
            "section": "Adverse Events",
            "expected": "AE domain, AESTDTC variable"
        },
        {
            "text": "Was the subject hospitalized?",
            "options": ["Yes", "No", "Unknown"],
            "section": "Adverse Events",
            "expected": "AE domain, AEHOSP variable with NY codelist"
        },
        {
            "text": "Hemoglobin result",
            "options": [],
            "section": "Laboratory",
            "expected": "LB domain, LBORRES variable, LBTEST='Hemoglobin'"
        },
        {
            "text": "Subject's date of birth",
            "options": [],
            "section": "Demographics",
            "expected": "DM domain, BRTHDTC variable"
        },
        {
            "text": "Concomitant medication name",
            "options": [],
            "section": "Concomitant Medications",
            "expected": "CM domain, CMTRT variable"
        }
    ]
    
    for i, sample in enumerate(sample_questions, 1):
        print(f"\nSample Question {i}:")
        print(f"Q: {sample['text']}")
        print(f"Section: {sample['section']}")
        if sample['options']:
            print(f"Options: {', '.join(sample['options'])}")
        print(f"Expected: {sample['expected']}")
        print("-" * 40)
        
        # Search with RAG
        candidates = rag_system.search(
            question=sample['text'],
            options=sample['options'],
            section=sample['section'],
            top_k=3
        )
        
        print(f"Found {len(candidates)} candidates:")
        for j, candidate in enumerate(candidates, 1):
            print(f"\nCandidate {j}:")
            print(format_rag_result(candidate))


def test_crf_page(rag_system: EnhancedRAGSystem, page_path: Path):
    """Test with actual CRF page"""
    print("\n" + "="*80)
    print(f"TESTING WITH CRF PAGE: {page_path.name}")
    print("="*80 + "\n")
    
    # Load questions
    questions = load_crf_questions(page_path)
    print(f"Found {len(questions)} questions on this page\n")
    
    # Process each question
    for i, q_data in enumerate(questions, 1):
        question = q_data["question"]
        inputs = q_data["inputs"]
        
        # Extract options from inputs
        options = []
        for inp in inputs:
            if inp.get("input_type") == "option":
                options.append(inp.get("text", ""))
        
        print(f"\nQuestion {i}/{len(questions)}:")
        print(f"Q: {question.get('text', '')}")
        print(f"Section: {question.get('section', 'Unknown')}")
        print(f"Form: {question.get('form', 'Unknown')}")
        if options:
            print(f"Options: {', '.join(options[:5])}")  # Show first 5
        print("-" * 40)
        
        # Search with RAG
        candidates = rag_system.search(
            question=question.get('text', ''),
            options=options,
            section=question.get('section'),
            form_name=question.get('form'),
            top_k=3
        )
        
        print(f"Top {len(candidates)} candidates:")
        for j, candidate in enumerate(candidates, 1):
            print(f"\nCandidate {j}:")
            print(format_rag_result(candidate))
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced RAG System")
    parser.add_argument("--page", type=int, help="Page number to test (e.g., 1 for page_000.json)")
    parser.add_argument("--kb-path", default="kb/sdtmig_v3_4_complete", 
                       help="Path to knowledge base")
    parser.add_argument("--crf-dir", default="crf_json", 
                       help="Directory containing CRF JSON files")
    parser.add_argument("--sample-only", action="store_true", 
                       help="Only test with sample questions")
    parser.add_argument("--dense-model", default="BAAI/bge-m3",
                       help="Dense retrieval model")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base",
                       help="Reranker model")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    print("Initializing Enhanced RAG System...")
    try:
        rag_system = EnhancedRAGSystem(
            kb_path=Path(args.kb_path),
            dense_model=args.dense_model,
            reranker_model=args.reranker_model
        )
        print("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return 1
    
    # Test with sample questions
    test_sample_questions(rag_system)
    
    # Test with CRF page if specified
    if not args.sample_only and args.page is not None:
        # Find the page file
        crf_dirs = list(Path(args.crf_dir).glob("*"))
        if not crf_dirs:
            print(f"\nNo CRF directories found in {args.crf_dir}")
            return 1
        
        # Use first directory
        crf_dir = crf_dirs[0]
        page_file = crf_dir / f"page_{args.page-1:03d}.json"
        
        if page_file.exists():
            test_crf_page(rag_system, page_file)
        else:
            print(f"\nPage file not found: {page_file}")
            print(f"Available files in {crf_dir}:")
            for f in sorted(crf_dir.glob("page_*.json"))[:10]:
                print(f"  {f.name}")
    
    return 0


if __name__ == "__main__":
    exit(main())