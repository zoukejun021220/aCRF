#!/usr/bin/env python3
"""Test imports for debugging"""

import sys
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

print("Testing imports...")

try:
    print("1. Importing schemas...")
    from stage_b_mapper_module.rag.indices.schemas import DomainDocument, VariableDocument
    print("   ✓ Schemas imported successfully")
except Exception as e:
    print(f"   ✗ Error importing schemas: {e}")

try:
    print("2. Importing index builder...")
    from stage_b_mapper_module.rag.indices.index_builder import CDISCIndexBuilder
    print("   ✓ Index builder imported successfully")
except Exception as e:
    print(f"   ✗ Error importing index builder: {e}")

try:
    print("3. Importing hybrid retriever...")
    from stage_b_mapper_module.rag.retrievers.hybrid_retriever import HybridRetriever
    print("   ✓ Hybrid retriever imported successfully")
except Exception as e:
    print(f"   ✗ Error importing hybrid retriever: {e}")

try:
    print("4. Importing cross encoder...")
    from stage_b_mapper_module.rag.rerankers.cross_encoder import CrossEncoderReranker
    print("   ✓ Cross encoder imported successfully")
except Exception as e:
    print(f"   ✗ Error importing cross encoder: {e}")

try:
    print("5. Importing enhanced RAG system...")
    from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem
    print("   ✓ Enhanced RAG system imported successfully")
except Exception as e:
    print(f"   ✗ Error importing enhanced RAG system: {e}")

print("\nAll critical imports tested.")

# Test KB path
kb_path = Path("kb/sdtmig_v3_4_complete")
if kb_path.exists():
    print(f"\n✓ KB path exists: {kb_path}")
    # Check for key files
    key_files = [
        "proto_define.json",
        "domains_by_class.json",
        "variables_all.json"
    ]
    for file in key_files:
        if (kb_path / file).exists():
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
else:
    print(f"\n✗ KB path not found: {kb_path}")

print("\nImport test complete.")