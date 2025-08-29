#!/usr/bin/env python3
"""
Prebuild and persist RAG embeddings for a KB so later runs are fast.

- Saves dense embeddings to `<KB_PATH>/.rag_cache/` (domain/variable/ct)
- Optionally packages the cache into a single tar.gz for distribution

Usage:
  python pipeline/build_kb_cache.py --kb-path kb/sdtmig_v3_4_complete \
      --dense-model "BAAI/bge-m3" --reranker-model "BAAI/bge-reranker-base" \
      [--pack] [--out kb/rag_cache_bge-m3.tar.gz]
"""

import argparse
import tarfile
from pathlib import Path

from stage_b_mapper_module.rag.enhanced_rag_system import EnhancedRAGSystem


def pack_cache(kb_path: Path, model_name: str, out_path: Path):
    cache_dir = kb_path / ".rag_cache"
    if not cache_dir.exists():
        print(f"No cache dir found at {cache_dir}. Nothing to pack.")
        return
    out_path = out_path or (kb_path / f"rag_cache_{slug(model_name)}.tar.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(cache_dir, arcname=cache_dir.name)
    print(f"Packed cache: {out_path}")


def slug(s: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in (s or 'model'))


def main():
    p = argparse.ArgumentParser(description="Build and persist RAG KB embedding cache")
    p.add_argument("--kb-path", required=True, help="Path to knowledge base directory")
    p.add_argument("--dense-model", default="BAAI/bge-m3", help="Dense retrieval model")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base", help="Reranker model")
    p.add_argument("--pack", action="store_true", help="Package cache directory into a tar.gz")
    p.add_argument("--out", help="Output tar.gz path when using --pack")
    args = p.parse_args()

    kb_path = Path(args.kb_path)
    print("Initializing RAG to build embeddings cache...")
    rag = EnhancedRAGSystem(kb_path=kb_path, dense_model=args.dense_model, reranker_model=args.reranker_model)
    # Touch the retriever to ensure caches are built (already done during init indexing)
    print(f"Cache built under: {kb_path / '.rag_cache'}")

    if args.pack:
        out = Path(args.out) if args.out else None
        pack_cache(kb_path, args.dense_model, out)


if __name__ == "__main__":
    main()

