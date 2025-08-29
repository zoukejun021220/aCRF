#!/usr/bin/env python3
"""Test enhanced RAG system without reranker to debug"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add pipeline to path
sys.path.append(str(Path(__file__).parent / "pipeline"))

from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem

# Monkey patch to skip reranking
original_search = EnhancedRAGSystem.search

def search_without_reranking(self, question, options=None, section=None, form_name=None, top_k=5):
    # Build query
    query = self._build_query(question, options, section, form_name)
    
    # Retrieve from all indices
    logger.info("Retrieving candidates...")
    domain_candidates = self.retriever.retrieve(query, "domain", top_k=50, alpha=0.7)
    variable_candidates = self.retriever.retrieve(query, "variable", top_k=200, alpha=0.7)
    ct_candidates = self.retriever.retrieve(query, "ct", top_k=200, alpha=0.5)
    
    # Convert to format for linking (skip reranking)
    domain_tuples = [(c, c.hybrid_score) for c in domain_candidates[:20]]
    variable_tuples = [(c, c.hybrid_score) for c in variable_candidates[:100]]
    ct_tuples = [(c, c.hybrid_score) for c in ct_candidates[:100]]
    
    # Link and score candidates
    logger.info("Linking candidates...")
    final_candidates = self._link_and_score(
        query, options,
        domain_tuples,
        variable_tuples,
        ct_tuples
    )
    
    # Sort by final score and return top K
    final_candidates.sort(key=lambda x: x.score, reverse=True)
    return final_candidates[:top_k]

# Apply monkey patch
EnhancedRAGSystem.search = search_without_reranking

def test_specific_query():
    """Test specific query for BRTHDTC"""
    
    # Initialize RAG system
    kb_path = Path("kb/sdtmig_v3_4_complete")
    rag = EnhancedRAGSystem(kb_path, dense_model="BAAI/bge-small-en-v1.5")
    
    print("\n" + "="*80)
    print("TESTING WITHOUT RERANKER - Subject's date of birth")
    print("="*80)
    
    # Search
    results = rag.search(
        question="Subject's date of birth",
        options=None,
        section="Demographics",
        top_k=10
    )
    
    print(f"\nFound {len(results)} candidates:")
    
    for i, candidate in enumerate(results, 1):
        print(f"\nCandidate {i}:")
        print(f"  Score: {candidate.score:.3f}")
        print(f"  Domain: {candidate.domain.domain_code} - {candidate.domain.domain_name}")
        print(f"  Variable: {candidate.variable.var_name} ({candidate.variable.label})")
        print(f"    Role: {candidate.variable.role or 'N/A'}")
        
        if candidate.variable.var_name == "BRTHDTC":
            print("    ⭐ FOUND BRTHDTC!")
            
        print(f"  Score Breakdown:")
        for component, value in candidate.score_breakdown.items():
            print(f"    {component}: {value:.3f}")

if __name__ == "__main__":
    test_specific_query()