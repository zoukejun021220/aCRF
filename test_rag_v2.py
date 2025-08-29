#!/usr/bin/env python3
"""Test enhanced RAG system V2 with improved scoring"""

import sys
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add pipeline to path
sys.path.append(str(Path(__file__).parent / "pipeline"))

from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem, RAGCandidate

def test_sample_questions(rag: EnhancedRAGSystem):
    """Test with sample questions"""
    
    sample_questions = [
        {
            "question": "Date of adverse event onset",
            "section": "Adverse Events",
            "options": None,
            "expected": "AE domain, AESTDTC variable"
        },
        {
            "question": "Was the subject hospitalized?",
            "section": "Adverse Events", 
            "options": ["Yes", "No", "Unknown"],
            "expected": "AE domain, AEHOSP variable with NY codelist"
        },
        {
            "question": "Hemoglobin result",
            "section": "Laboratory",
            "options": None,
            "expected": "LB domain, LBORRES variable, LBTEST='Hemoglobin'"
        },
        {
            "question": "Subject's date of birth",
            "section": "Demographics",
            "options": None,
            "expected": "DM domain, BRTHDTC variable"
        },
        {
            "question": "Concomitant medication name",
            "section": "Concomitant Medications",
            "options": None,
            "expected": "CM domain, CMTRT variable"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING WITH SAMPLE QUESTIONS")
    print("="*80)
    
    for i, sample in enumerate(sample_questions, 1):
        print(f"\n\nSample Question {i}:")
        print(f"Q: {sample['question']}")
        print(f"Section: {sample['section']}")
        if sample['options']:
            print(f"Options: {', '.join(sample['options'])}")
        print(f"Expected: {sample['expected']}")
        print("-" * 40)
        
        # Search
        results = rag.search(
            question=sample['question'],
            options=sample['options'],
            section=sample['section'],
            top_k=3
        )
        
        print(f"Found {len(results)} candidates:")
        
        for j, candidate in enumerate(results, 1):
            print(f"\nCandidate {j}:")
            print(f"  Score: {candidate.score:.3f}")
            print(f"  Domain: {candidate.domain.domain_code} - {candidate.domain.domain_name}")
            print(f"  Variable: {candidate.variable.var_name} ({candidate.variable.label})")
            print(f"    Role: {candidate.variable.role or 'N/A'}")
            
            # Show variable definition (truncated)
            definition = candidate.variable.definition or candidate.variable.label
            if len(definition) > 100:
                definition = definition[:100] + "..."
            print(f"    Definition: {definition}")
            
            # Check if expected variable was found
            if i == 1 and candidate.variable.var_name == "AESTDTC":
                print("    ✓ FOUND EXPECTED VARIABLE!")
            elif i == 3 and candidate.variable.var_name == "LBORRES":
                print("    ✓ FOUND EXPECTED VARIABLE!")
            elif i == 4 and candidate.variable.var_name == "BRTHDTC":
                print("    ✓ FOUND EXPECTED VARIABLE!")
            elif i == 5 and candidate.variable.var_name == "CMTRT":
                print("    ✓ FOUND EXPECTED VARIABLE!")
                
            if candidate.ct_terms:
                print(f"  CT Terms: {len(candidate.ct_terms)} linked")
                for ct in candidate.ct_terms[:2]:
                    print(f"    - {ct.submission_value}: {ct.preferred_term}")
                    
            print(f"  Score Breakdown:")
            for component, value in candidate.score_breakdown.items():
                print(f"    {component}: {value:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Test enhanced RAG system')
    parser.add_argument('--use-reranker', action='store_true', default=False,
                       help='(Ignored) Reranker is enabled in unified system')
    parser.add_argument('--model', type=str, default='BAAI/bge-small-en-v1.5',
                       help='Dense retrieval model')
    args = parser.parse_args()
    
    # Initialize RAG system
    kb_path = Path("kb/sdtmig_v3_4_complete")
    
    print("Initializing Enhanced RAG System V2...")
    print(f"KB Path: {kb_path}")
    print(f"Dense Model: {args.model}")
    print(f"Use Reranker: {args.use_reranker}")
    
    rag = EnhancedRAGSystem(
        kb_path,
        dense_model=args.model,
        reranker_model="BAAI/bge-reranker-base"
    )
    
    print("RAG system initialized successfully!")
    
    # Test with samples
    test_sample_questions(rag)

if __name__ == "__main__":
    main()
