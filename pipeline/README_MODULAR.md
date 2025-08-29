# Modularized Pipeline Structure

The original large pipeline files have been split into modular components for better maintainability.

## Directory Structure

```
pipeline/
├── stage_a_digitizer.py (original - can be removed)
├── stage_b_mapper_unified.py (original - can be removed)
├── stage_a_digitizer_module/
│   ├── __init__.py
│   ├── main.py (entry point)
│   ├── core/
│   │   ├── __init__.py
│   │   └── digitizer.py (main CRFDigitizer class)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_classes.py (BoundingBox, ExtractedLine, CRFItem)
│   │   └── vision_model.py (Qwen-VL model handling)
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── extraction_prompts.py (prompts for different extractions)
│   │   ├── page_processor.py (page-level processing logic)
│   │   ├── response_parser.py (model response parsing)
│   │   └── fallback_extractor.py (fallback extraction)
│   └── utils/
│       ├── __init__.py
│       └── pdf_extractor.py (PDF extraction utilities)
│
├── stage_b_mapper_module/
│   ├── __init__.py
│   ├── main.py (entry point)
│   ├── core/
│   │   ├── __init__.py
│   │   └── mapper.py (main UnifiedSDTMMapper class)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_classes.py (SDTMMapping, AnnotationResult, etc.)
│   │   ├── llm_model.py (language model handling)
│   │   └── rag_model.py (RAG for semantic search)
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── class_selector.py (SDTM class selection)
│   │   ├── domain_selector.py (domain selection)
│   │   ├── pattern_selector.py (pattern selection)
│   │   └── variable_selector.py (variable selection)
│   ├── kb_handlers/
│   │   ├── __init__.py
│   │   └── kb_loader.py (knowledge base loading)
│   └── utils/
│       ├── __init__.py
│       ├── file_processor.py (CRF JSON file processing)
│       └── debug_manager.py (debug information management)
│
├── proto_define_validator.py (shared dependency)
├── enhanced_validator.py (shared dependency)
└── rule_based_filter.py (shared dependency)
```

## Usage

### Stage A - CRF Digitizer

```bash
# From the stage_a_digitizer_module directory
python main.py path/to/crf.pdf --doc-id study001

# Or import as module
from stage_a_digitizer_module import CRFDigitizer
digitizer = CRFDigitizer()
result = digitizer.process_pdf(pdf_path)
```

### Stage B - SDTM Mapper

```bash
# Process single file
python stage_b_mapper_module/main.py process-file path/to/page_001.json --output annotations.json

# Process directory
python stage_b_mapper_module/main.py process-dir crf_json/study001 --output-dir results/

# Or import as module
from stage_b_mapper_module import UnifiedSDTMMapper
mapper = UnifiedSDTMMapper()
result = mapper.annotate_field(field_data)
```

## Key Benefits of Modular Structure

1. **Better Organization**: Each component has a clear responsibility
2. **Easier Testing**: Individual modules can be tested independently
3. **Reusability**: Components can be imported and used separately
4. **Maintainability**: Smaller files are easier to understand and modify
5. **Parallel Development**: Multiple developers can work on different modules

## Migration Notes

- The modular versions maintain the same logic as the original files
- All imports have been updated to work with the new structure
- The main entry points (main.py) provide the same CLI interfaces
- Original files can be removed once the modular versions are tested