# Enhanced RAG System for SDTM Mapping

This is a battle-tested RAG implementation for mapping CRF questions to SDTM domains, variables, and controlled terminology.

## Architecture

### Three Separate Indices
1. **Domain Index** - SDTM domains with class, name, and definitions
2. **Variable Index** - SDTM variables with roles, labels, and relationships  
3. **CT Index** - Controlled terminology terms with definitions, synonyms, and NCI codes

### Retrieval Pipeline
1. **Query Construction** - Enriches question with options, section, form, and format hints
2. **Hybrid Retrieval** - BGE-M3 (dense) + BM25 (sparse) for complementary coverage
3. **Cross-Encoder Reranking** - BGE-reranker for high-precision ranking
4. **Linking & Scoring** - Links variables→domains and variables→CT with role/format bonuses

## Key Features

- **Authoritative Sources**: Built from CDISC proto_define.json and KB files
- **Smart CT Matching**: Recognizes Yes/No→NY, paired TEST/TESTCD codelists
- **Role-Aware**: Timing roles for dates, Result Qualifiers for units
- **Format Detection**: Infers date/time, numeric+units, Yes/No patterns
- **Section Bonus**: Matches "Adverse Events"→AE, "Laboratory"→LB, etc.

## Usage

### Basic Usage
```python
from stage_b_mapper_module.rag import EnhancedRAGSystem

# Initialize
rag = EnhancedRAGSystem(kb_path="kb/sdtmig_v3_4_complete")

# Search
candidates = rag.search(
    question="Date of adverse event onset",
    options=[],
    section="Adverse Events",
    top_k=5
)

# Results include linked domain, variable, and CT
for candidate in candidates:
    print(f"Domain: {candidate.domain.domain_code}")
    print(f"Variable: {candidate.variable.var_name}")
    print(f"CT Terms: {[ct.submission_value for ct in candidate.ct_terms]}")
    print(f"Score: {candidate.score:.3f}")
```

### Test Script
```bash
# Test with sample questions
python test_enhanced_rag.py --sample-only

# Test with specific CRF page
python test_enhanced_rag.py --page 1

# Use custom KB path
python test_enhanced_rag.py --kb-path path/to/kb --page 2
```

## Models Used

- **Dense Retrieval**: BAAI/bge-m3 (8k context, multi-vector support)
- **Sparse Retrieval**: BM25 (for acronyms and exact matches)
- **Reranking**: BAAI/bge-reranker-base (cross-encoder)

## Configuration

### Retrieval Parameters
- Domain candidates: 50 → rerank to 10
- Variable candidates: 200 → rerank to 50  
- CT candidates: 200 → rerank to 100
- Final results: top 5 linked triads

### Scoring Formula
```
score = 0.45 * variable_score 
      + 0.25 * ct_score
      + 0.20 * domain_score
      + 0.05 * role_bonus
      + 0.03 * format_bonus  
      + 0.02 * section_bonus
```

## Performance Tips

1. **GPU Recommended**: Both BGE-M3 and reranker benefit from GPU
2. **Batch Processing**: Process multiple questions together
3. **Caching**: Index embeddings are cached after first load
4. **Model Selection**: Can swap BGE-M3 for MedCPT for biomedical text

## Dependencies

See `requirements_rag.txt` for full list. Key packages:
- sentence-transformers
- FlagEmbedding (for BGE models)
- rank-bm25
- transformers
- torch