#!/usr/bin/env python3
"""Diagnose RAG retrieval issues"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add pipeline to path
sys.path.append(str(Path(__file__).parent / "pipeline"))

from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem
from stage_b_mapper_module.rag.indices.index_builder import CDISCIndexBuilder

def diagnose_retrieval():
    """Diagnose why expected variables aren't being retrieved"""
    
    # Initialize system
    kb_path = Path("kb/sdtmig_v3_4_complete")
    
    print("Building indices to check for expected variables...")
    index_builder = CDISCIndexBuilder(kb_path)
    domain_docs, variable_docs, ct_docs = index_builder.build_all_indices()
    
    # Check for specific variables
    expected_vars = [
        ("AESTDTC", "AE"),
        ("BRTHDTC", "DM"),
        ("CMTRT", "CM"),
        ("LBORRES", "LB"),
        ("AEHOSP", "AE")
    ]
    
    print("\n" + "="*80)
    print("CHECKING FOR EXPECTED VARIABLES IN INDEX")
    print("="*80)
    
    for var_name, expected_domain in expected_vars:
        print(f"\nLooking for {var_name} in {expected_domain} domain:")
        
        # Find all instances of this variable
        found_vars = [v for v in variable_docs if v.var_name == var_name]
        
        if not found_vars:
            print(f"  ❌ NOT FOUND in variable index!")
        else:
            for var_doc in found_vars:
                print(f"  ✓ Found in domain: {var_doc.domain_code}")
                print(f"    Label: {var_doc.label}")
                print(f"    Role: {var_doc.role or 'No role'}")
                print(f"    Embedding text: {var_doc.embedding_text[:100]}...")
                
                # Check if it's in the expected domain
                if var_doc.domain_code == expected_domain:
                    print(f"    ✓ Found in expected domain!")
                else:
                    print(f"    ⚠️  Found in {var_doc.domain_code} instead of {expected_domain}")
    
    # Check domain retrieval
    print("\n" + "="*80)
    print("CHECKING DOMAIN INDEX")
    print("="*80)
    
    expected_domains = ["AE", "DM", "CM", "LB"]
    for domain_code in expected_domains:
        found = [d for d in domain_docs if d.domain_code == domain_code]
        if found:
            print(f"✓ {domain_code}: {found[0].domain_name}")
        else:
            print(f"❌ {domain_code}: NOT FOUND")
    
    # Test retrieval with actual system
    print("\n" + "="*80)
    print("TESTING RETRIEVAL SYSTEM")
    print("="*80)
    
    rag = EnhancedRAGSystem(kb_path, 
                           dense_model="BAAI/bge-small-en-v1.5",
                           reranker_model="BAAI/bge-reranker-base")
    
    # Test specific query
    query = "Date of adverse event onset"
    print(f"\nQuery: {query}")
    
    # Get raw retrieval results
    enriched_query = rag._build_query(query, None, "Adverse Events", None)
    print(f"Enriched query: {enriched_query}")
    
    # Retrieve variables
    variable_candidates = rag.retriever.retrieve(enriched_query, "variable", top_k=20)
    
    print("\nTop 20 variable candidates:")
    for i, cand in enumerate(variable_candidates[:20]):
        var_doc = cand.doc
        print(f"{i+1}. {var_doc.var_name} ({var_doc.domain_code}): {var_doc.label}")
        print(f"   Dense: {cand.dense_score:.3f}, Sparse: {cand.sparse_score:.3f}, Hybrid: {cand.hybrid_score:.3f}")
        if var_doc.var_name == "AESTDTC":
            print("   ⭐ FOUND EXPECTED VARIABLE!")

if __name__ == "__main__":
    diagnose_retrieval()