# Repository Guidelines

This guide helps contributors work efficiently with the SDTM mapping pipeline and knowledge base tools.

## Project Structure & Module Organization
- `pipeline/`: Core code.
  - `stage_a_digitizer_module/`: Digitizes CRF PDFs. CLI: `main.py`.
  - `stage_b_mapper_module/`: Maps to SDTM. CLI: `main.py`.
  - Shared utilities: `enhanced_validator.py`, `proto_define_validator.py`, `rule_based_filter.py`.
- `kb/`: Ready-to-use SDTM knowledge base (e.g., `kb/sdtmig_v3_4_complete`).
- `sdtm_knowledge_base/`: KB builder, retrievers, docs, and scripts.
- `data/`, `annotations/`, `annotated_output/`: Inputs and pipeline outputs (gitignored where appropriate).
- Tests: simple scripts at repo root named `test_*.py`.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r pipeline/requirements_rag.txt`
  - Optional KB tools: `pip install -r sdtm_knowledge_base/requirements_rag.txt`
- Use KB or build one:
  - Use: pass `--kb-path kb/sdtmig_v3_4_complete`
  - Build: `bash sdtm_knowledge_base/build_rag_kb.sh` (outputs to `sdtm_knowledge_base/sdtm_rag_kb/`)
- Run Stage A (digitize CRF PDF):
  - `python pipeline/stage_a_digitizer_module/main.py path/to/crf.pdf --doc-id study001`
- Run Stage B (map to SDTM):
  - File: `python pipeline/stage_b_mapper_module/main.py process-file crf_json/page_001.json --kb-path kb/sdtmig_v3_4_complete --output annotations/page_001_annotated.json`
  - Dir: `python pipeline/stage_b_mapper_module/main.py process-dir crf_json/study001 --kb-path kb/sdtmig_v3_4_complete --output-dir annotations/`
- Tests:
  - `python test_imports.py`
  - `python test_rag_v2.py --kb-path kb/sdtmig_v3_4_complete --page 1`
  - `python test_without_reranker.py`

## Coding Style & Naming Conventions
- Python 3.10+; PEP 8; 4-space indent; type hints for public APIs; module docstrings.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Formatting: Black (88 cols) and isort.
  - Example: `black pipeline sdtm_knowledge_base && isort pipeline sdtm_knowledge_base`

## Testing Guidelines
- Prefer lightweight scripts; keep KB paths explicit via `--kb-path`.
- Optional: run `pytest -q` if installed. Name tests `test_*.py` and keep fixtures near usage.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(mapper): improve variable selector`).
- PRs: include description, run instructions, linked issues, sample inputs/outputs, and doc updates (e.g., `pipeline/README_MODULAR.md`). Note model names/versions and any `--device` or `--kb-path` assumptions.

## Security & Configuration
- Do not commit credentials or large model artifacts; use env vars for tokens. Large models/embeddings may download at runtime; prefer GPU when available (`--device cuda`) or set `--fallback-only` for vision-less runs.

