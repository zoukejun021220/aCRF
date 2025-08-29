"""Hybrid retriever combining BGE-M3 dense retrieval with BM25 sparse retrieval"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import hashlib
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    try:
        nltk.download('punkt')
    except Exception:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    try:
        nltk.download('punkt_tab')
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except Exception:
    try:
        nltk.download('stopwords')
    except Exception:
        pass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """Candidate document from retrieval"""
    doc_id: int
    doc: Any  # Domain/Variable/CT document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    

class HybridRetriever:
    """Hybrid retriever using BGE-M3 (dense) + BM25 (sparse)"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 use_fp16: bool = True,
                 device: str = None,
                 cache_dir: Optional[Path] = None):
        """
        Initialize hybrid retriever
        
        Args:
            model_name: Dense retriever model (BGE-M3 or alternatives)
            use_fp16: Use half precision for efficiency
            device: Device to use (None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        logger.info(f"Initializing hybrid retriever with {model_name} on {self.device}")
        
        # Initialize dense retriever (BGE-M3)
        self.encoder = SentenceTransformer(model_name, device=self.device)
        if use_fp16 and self.device == 'cuda':
            self.encoder.half()
            
        # Storage for indices
        self.domain_index = None
        self.variable_index = None
        self.ct_index = None
        
        # BM25 indices
        self.domain_bm25 = None
        self.variable_bm25 = None
        self.ct_bm25 = None
        
        # Document storage
        self.domain_docs = []
        self.variable_docs = []
        self.ct_docs = []
        
        # Optional embedding cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Stop words for BM25
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback to basic stop words
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 
                              'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 
                              'from', 'by', 'that', 'this', 'it', 'are', 'was', 
                              'were', 'been', 'be', 'have', 'has', 'had', 'do', 
                              'does', 'did', 'will', 'can', 'could', 'should', 
                              'would', 'may', 'might', 'must', 'shall'}
        
    def index_documents(self, domain_docs: List, variable_docs: List, ct_docs: List):
        """Index all documents for both dense and sparse retrieval"""
        logger.info("Building dense embeddings...")
        # Prepare cache directory
        cache_base = None
        if self.cache_dir:
            cache_base = self._emb_cache_dir(domain_docs, variable_docs, ct_docs)
            try:
                cache_base.mkdir(parents=True, exist_ok=True)
            except Exception:
                cache_base = None
        
        # Build domain index
        if domain_docs:
            domain_texts = [doc.embedding_text for doc in domain_docs]
            self.domain_docs = domain_docs
            self.domain_embeddings = self._load_or_encode(cache_base, 'domain_embeddings.pt', domain_texts)
            # Build BM25 for domains
            domain_tokens = [self._tokenize(text) for text in domain_texts]
            self.domain_bm25 = BM25Okapi(domain_tokens)
            
        # Build variable index
        if variable_docs:
            variable_texts = [doc.embedding_text for doc in variable_docs]
            self.variable_docs = variable_docs
            self.variable_embeddings = self._load_or_encode(cache_base, 'variable_embeddings.pt', variable_texts)
            # Build BM25 for variables
            variable_tokens = [self._tokenize(text) for text in variable_texts]
            self.variable_bm25 = BM25Okapi(variable_tokens)
            
        # Build CT index
        if ct_docs:
            ct_texts = [doc.embedding_text for doc in ct_docs]
            self.ct_docs = ct_docs
            self.ct_embeddings = self._load_or_encode(cache_base, 'ct_embeddings.pt', ct_texts)
            # Build BM25 for CT
            ct_tokens = [self._tokenize(text) for text in ct_texts]
            self.ct_bm25 = BM25Okapi(ct_tokens)
            
        logger.info(f"Indexed {len(domain_docs)} domains, {len(variable_docs)} variables, "
                   f"{len(ct_docs)} CT terms")
    
    def retrieve(self, query: str, index_type: str, top_k: int = 200,
                alpha: float = 0.5) -> List[RetrievalCandidate]:
        """
        Hybrid retrieval from specified index
        
        Args:
            query: Query string
            index_type: One of 'domain', 'variable', 'ct'
            top_k: Number of candidates to retrieve
            alpha: Weight for dense retrieval (1-alpha for sparse)
            
        Returns:
            List of retrieval candidates with scores
        """
        # Select appropriate index
        if index_type == 'domain':
            docs = self.domain_docs
        elif index_type == 'variable':
            docs = self.variable_docs
        elif index_type == 'ct':
            docs = self.ct_docs
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        if not docs:
            return []
        
        # Assign embeddings and BM25 only after verifying docs exist
        if index_type == 'domain':
            dense_embeddings = self.domain_embeddings
            bm25 = self.domain_bm25
        elif index_type == 'variable':
            dense_embeddings = self.variable_embeddings
            bm25 = self.variable_bm25
        else:  # ct
            dense_embeddings = self.ct_embeddings
            bm25 = self.ct_bm25
            
        # Dense retrieval with BGE / SentenceTransformer
        query_text = self._with_query_instruction(query)
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        dense_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), dense_embeddings)
        dense_scores = dense_scores.cpu().numpy()
        
        # Sparse retrieval with BM25
        query_tokens = self._tokenize(query)
        sparse_scores = bm25.get_scores(query_tokens)
        
        # Normalize scores to [0, 1]
        dense_scores_norm = self._normalize_scores(dense_scores)
        sparse_scores_norm = self._normalize_scores(sparse_scores)
        
        # Combine scores
        hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * sparse_scores_norm
        
        # Get top-k candidates
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            candidate = RetrievalCandidate(
                doc_id=idx,
                doc=docs[idx],
                dense_score=float(dense_scores_norm[idx]),
                sparse_score=float(sparse_scores_norm[idx]),
                hybrid_score=float(hybrid_scores[idx])
            )
            candidates.append(candidate)
            
        return candidates
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        tokens = word_tokenize(text.lower())
        # Remove stop words and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return tokens
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _with_query_instruction(self, query: str) -> str:
        """Apply model-specific query instruction prompt if appropriate.
        For BGE-small/large/M3, adding an instruction prefix improves retrieval.
        If the query already appears prefixed, return as-is.
        """
        q = (query or "").strip()
        if q.lower().startswith("represent this"):
            return q
        mn = (self.model_name or "").lower()
        if "bge" in mn:
            return f"Represent this question for searching relevant passages: {q}"
        return q

    def _emb_cache_dir(self, domain_docs: List, variable_docs: List, ct_docs: List) -> Path:
        """Compute a stable cache subdir for embeddings based on model and corpus size.
        This keeps cache valid unless KB changes significantly (counts change).
        """
        model_slug = self._slug(self.model_name)
        finger = f"{model_slug}:{len(domain_docs)}:{len(variable_docs)}:{len(ct_docs)}"
        h = hashlib.sha1(finger.encode()).hexdigest()[:12]
        return self.cache_dir / model_slug / h

    def _slug(self, s: str) -> str:
        return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in (s or 'model'))

    def _load_or_encode(self, cache_base: Optional[Path], fname: str, texts: List[str]):
        """Load embeddings from cache if present, otherwise encode and cache."""
        if cache_base is not None:
            path = cache_base / fname
            if path.exists():
                try:
                    emb = torch.load(path, map_location=self.device)
                    if isinstance(emb, np.ndarray):
                        emb = torch.from_numpy(emb)
                    return emb
                except Exception:
                    pass
        # Encode
        emb = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        # Save
        if cache_base is not None:
            try:
                torch.save(emb.cpu(), cache_base / fname)
            except Exception:
                pass
        return emb
    
    def add_ct_hints_to_query(self, query: str, options: List[str]) -> str:
        """
        Enrich query with CT hints from options
        
        Args:
            query: Original query
            options: Field options (e.g., ["Yes", "No", "Unknown"])
            
        Returns:
            Enriched query
        """
        ct_hints = []
        
        # Common option mappings
        option_mappings = {
            "yes": ["Y", "NY", "C66742"],
            "no": ["N", "NY", "C66742"],
            "unknown": ["U", "UNK", "C66742"],
            "not applicable": ["NA", "C66742"],
            "not done": ["ND", "NOTDONE", "C66742"],
            # Severity
            "mild": ["MILD", "C66769"],
            "moderate": ["MODERATE", "C66769"],
            "severe": ["SEVERE", "C66769"]
        }
        
        for option in options:
            option_lower = option.lower().strip()
            if option_lower in option_mappings:
                ct_hints.extend(option_mappings[option_lower])
        
        if ct_hints:
            unique_hints = list(set(ct_hints))
            return f"{query} CT_HINTS: {' '.join(unique_hints)}"
            
        return query
