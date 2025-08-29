# Repository Guidelines

## Project Structure & Modules
- `pipeline/`: Core code. Stage A (`stage_a_digitizer_module/`) and Stage B (`stage_b_mapper_module/`) each provide a `main.py` CLI. Shared utilities live alongside (`enhanced_validator.py`, `proto_define_validator.py`, `rule_based_filter.py`).
- `kb/`: Ready-to-use SDTM knowledge base (e.g., `kb/sdtmig_v3_4_complete`).
- `sdtm_knowledge_base/`: KB builder, retrievers, and docs; optional scripts to regenerate KB.
- `data/`, `annotations/`, `annotated_output/`: Inputs and pipeline outputs (gitignored where appropriate).
- Tests live at repo root as `test_*.py` utilities.

## Setup, Build & Run
- Create env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r pipeline/requirements_rag.txt`
  - Optional KB tools: `pip install -r sdtm_knowledge_base/requirements_rag.txt`
- Use existing KB or build one:
  - Use: `--kb-path kb/sdtmig_v3_4_complete`
  - Build: `bash sdtm_knowledge_base/build_rag_kb.sh` (outputs `sdtm_knowledge_base/sdtm_rag_kb/`)
- Run Stage A (digitize CRF PDF):
  - `python pipeline/stage_a_digitizer_module/main.py path/to/crf.pdf --doc-id study001`
- Run Stage B (map to SDTM):
  - File: `python pipeline/stage_b_mapper_module/main.py process-file crf_json/page_001.json --kb-path kb/sdtmig_v3_4_complete --output annotations/page_001_annotated.json`
  - Dir: `python pipeline/stage_b_mapper_module/main.py process-dir crf_json/study001 --kb-path kb/sdtmig_v3_4_complete --output-dir annotations/`

## Coding Style & Naming
- Python 3.10+; PEP 8; 4-space indent. Type hints for public APIs; module docstrings.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Formatting: Black (88 cols) and isort.
  - Example: `black pipeline sdtm_knowledge_base && isort pipeline sdtm_knowledge_base`

## Testing Guidelines
- Lightweight tests as scripts:
  - `python test_imports.py`
  - `python test_rag_v2.py --kb-path kb/sdtmig_v3_4_complete --page 1`
  - `python test_without_reranker.py`
- Optional: `pytest -q` (if installed). Name tests `test_*.py`; place fixtures near usage.
- Keep KB path assumptions explicit in tests/CLI args.

## Commit & Pull Requests
- Use Conventional Commits (e.g., `feat(mapper): improve variable selector`).
- PRs include: clear description, run instructions, linked issues, sample inputs/outputs, and doc updates (e.g., `pipeline/README_MODULAR.md`).
- Note model names/versions and any `--device` or `--kb-path` assumptions.

## Security & Configuration
- Do not commit credentials or large model artifacts; `.gitignore` covers common cases. Store tokens in env vars.
- Large models/embeddings may download at runtime; prefer GPU when available (`--device cuda`) or set `--fallback-only` for vision-less runs.

