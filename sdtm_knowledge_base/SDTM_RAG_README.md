# SDTM RAG Knowledge Base System

## Overview

This enhanced RAG (Retrieval-Augmented Generation) system structures SDTM knowledge following the pattern:
**Domain → Variable → Definition → Codelist/aliases → When/Then patterns**

This mirrors SDTMIG structure for precise retrieval and mapping of CRF questions to SDTM variables.

## Key Components

### 1. Knowledge Base Builder (`sdtm_rag_builder.py`)
- Structures SDTM data into typed chunks
- Creates synonym mappings for common CRF terms
- Generates when/then patterns for conditional usage
- Incorporates SDTM-MSG annotation instructions
- Outputs small, focused chunks optimized for retrieval

### 2. Hybrid Retriever (`sdtm_rag_retriever.py`)
- BM25 for keyword matching
- Vector embeddings for semantic similarity
- Lightweight reranking combining both scores
- Direct alias matching for common CRF terms

### 3. Qwen Agent Layer (`sdtm_qwen_agent.py`)
- Automatic knowledge retrieval during annotation
- Structured prompting with retrieved context
- Confidence scoring and validation
- Batch processing with memory management

### 4. Enhanced Annotator (`sdtm_rag_annotator.py`)
- Integration point for CRF annotation
- Form-level context awareness
- Detailed mapping reports
- Statistics and confidence tracking

## Setup

1. Install dependencies:
```bash
cd sdtm_knowledge_base
pip install -r requirements_rag.txt
```

2. Build the knowledge base:
```bash
chmod +x build_rag_kb.sh
./build_rag_kb.sh
```

This creates the `sdtm_rag_kb/` directory with:
- `sdtm_variables_structured.json` - Structured variable entries
- `sdtm_domain_index.json` - Domain to variable mappings
- `sdtm_alias_index.json` - CRF term to variable mappings
- `sdtm_rag_chunks.json` - Small chunks for retrieval

## Usage

### Basic Retrieval
```python
from sdtm_rag_retriever import create_sdtm_retriever

retriever = create_sdtm_retriever("sdtm_rag_kb")
results = retriever.retrieve("Date of Birth", top_k=5)

for result in results:
    print(f"{result.full_name}: {result.score:.3f}")
```

### CRF Annotation
```python
from sdtm_rag_annotator import SDTMRAGAnnotator

annotator = SDTMRAGAnnotator()
decision = annotator.agent.annotate_crf_question(
    "What is the patient's temperature?",
    context={"form_name": "Vital Signs"}
)

print(f"Variable: {decision.variable}")
print(f"Confidence: {decision.confidence}")
```

### Batch Processing
```python
# Annotate entire CRF form
with open("crf_form.json") as f:
    crf_data = json.load(f)

result = annotator.annotate_crf_form(
    crf_data,
    output_path="annotated_form.json"
)

# Generate report
report = annotator.generate_mapping_report(
    result,
    output_path="mapping_report.txt"
)
```

## Key Features

### 1. Structured Knowledge Chunks
Each SDTM variable has multiple chunk types:
- **variable_definition**: Main definition and metadata
- **synonym_mapping**: CRF term to variable mappings
- **when_then_pattern**: Conditional usage patterns
- **annotation_instruction**: SDTM-MSG guidelines and rules
- **annotation_example**: Real examples from annotation instructions
- **annotation_concept**: Key concepts like RELREC, supplemental qualifiers

### 2. Common CRF Synonyms
Automatically maps common CRF terms:
- "Date of Birth" → DM.BRTHDTC
- "Temperature" → VS.VSORRES (when VSTESTCD='TEMP')
- "Serious AE" → AE.AESER
- "Concomitant medication" → CM.CMTRT

### 3. Context-Aware Retrieval
- Form name influences domain filtering
- Section/page context improves accuracy
- When/then patterns for test-specific variables

### 4. Confidence Scoring
- High (≥0.8): Strong match with clear evidence
- Medium (0.5-0.8): Good match, may need review
- Low (<0.5): Weak match, requires validation

## Annotation Instructions Integration

The system now includes comprehensive SDTM-MSG annotation instructions that guide proper mapping:

1. **Output Format**: Structured JSON with all required fields (domain, variables, when clauses, etc.)
2. **Core Rules**: 
   - Findings domains require when/then patterns
   - Supplemental qualifiers for "Other, specify" fields
   - RELREC for cross-domain relationships
   - Origin tracking (Collected, Derived, Predecessor, etc.)
3. **Examples**: Real-world mapping examples for common scenarios
4. **Validation**: Built-in checks for proper annotation compliance

These instructions are automatically retrieved when relevant to improve mapping accuracy.

## Extending the System

### Adding Custom Synonyms
Edit `_generate_crf_synonyms()` in `sdtm_rag_builder.py`:
```python
synonym_map = {
    "NEWVAR": ["Custom Term 1", "Custom Term 2"],
    # ...
}
```

### Adding When/Then Patterns
Edit `_generate_when_then_patterns()` in `sdtm_rag_builder.py`:
```python
if entry.variable == "LBORRES":
    patterns.append({
        "when": "LBTESTCD='NEWTEST'",
        "then": "New test result",
        "example": "123"
    })
```

### Tuning Retrieval
Adjust in `retrieve()` method:
- `alpha`: Balance between BM25 and vector (default 0.5)
- `top_k`: Initial candidates before reranking
- `rerank_top_k`: Final results after reranking

## Performance Optimization

1. **GPU Usage**: Automatically uses GPU for embeddings if available
2. **Batch Processing**: Process multiple questions together
3. **Caching**: Embeddings are pre-computed during build
4. **Memory Management**: Automatic cache clearing between batches

## Monitoring

The system provides detailed statistics:
- Total questions processed
- Confidence distribution
- Domain distribution
- Average confidence scores
- Error tracking

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use 4-bit quantization
2. **Low Confidence**: Check if CRF terms need synonym mappings
3. **Wrong Domain**: Add form-specific domain hints in context
4. **Missing Variables**: Ensure knowledge base is built from latest SDTM version