"""Cross-encoder reranker for high-precision ranking"""

import logging
import torch
from typing import List, Tuple, Any, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using BGE reranker or MS-MARCO models"""
    
    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-base",
                 device: str = None,
                 max_length: int = 512):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use
            max_length: Maximum sequence length
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        logger.info(f"Loading cross-encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def rerank(self, query: str, candidates: List[Any], 
               top_n: int = 50, get_text_fn=None) -> List[Tuple[Any, float]]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: Query string
            candidates: List of candidates (can be documents or any objects)
            top_n: Number of top candidates to return
            get_text_fn: Function to extract text from candidate (if not string)
            
        Returns:
            List of (candidate, score) tuples sorted by relevance
        """
        if not candidates:
            return []
            
        # Extract texts from candidates
        if get_text_fn:
            candidate_texts = [get_text_fn(c) for c in candidates]
        else:
            # Assume candidates have embedding_text attribute
            candidate_texts = []
            for c in candidates:
                if hasattr(c, 'embedding_text'):
                    candidate_texts.append(c.embedding_text)
                elif hasattr(c, 'doc') and hasattr(c.doc, 'embedding_text'):
                    candidate_texts.append(c.doc.embedding_text)
                else:
                    candidate_texts.append(str(c))
        
        # Score all candidates
        scores = self.score_pairs(query, candidate_texts)
        
        # Sort by score and return top N
        candidate_scores = list(zip(candidates, scores))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        return candidate_scores[:top_n]
    
    def score_pairs(self, query: str, documents: List[str]) -> List[float]:
        """
        Score query-document pairs
        
        Args:
            query: Query string
            documents: List of document texts
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
            
        scores = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Tokenize pairs
                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get scores
                outputs = self.model(**inputs)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    # Some models output logits
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                else:
                    # Others might output scores directly
                    batch_scores = outputs.cpu().numpy()
                
                scores.extend(batch_scores.tolist())
                
        return scores
    
    def rerank_multiple_indices(self, query: str, 
                               domain_candidates: List,
                               variable_candidates: List,
                               ct_candidates: List,
                               top_n_per_type: Dict[str, int] = None) -> Dict[str, List]:
        """
        Rerank candidates from multiple indices
        
        Args:
            query: Query string
            domain_candidates: Domain retrieval candidates
            variable_candidates: Variable retrieval candidates  
            ct_candidates: CT retrieval candidates
            top_n_per_type: Dict specifying top N for each type
            
        Returns:
            Dict with reranked candidates for each type
        """
        if top_n_per_type is None:
            top_n_per_type = {
                'domains': 10,
                'variables': 50,
                'ct': 100
            }
            
        results = {}
        
        # Rerank domains
        if domain_candidates:
            results['domains'] = self.rerank(
                query, domain_candidates, 
                top_n=top_n_per_type.get('domains', 10)
            )
        else:
            results['domains'] = []
            
        # Rerank variables
        if variable_candidates:
            results['variables'] = self.rerank(
                query, variable_candidates,
                top_n=top_n_per_type.get('variables', 50)
            )
        else:
            results['variables'] = []
            
        # Rerank CT terms
        if ct_candidates:
            results['ct'] = self.rerank(
                query, ct_candidates,
                top_n=top_n_per_type.get('ct', 100)
            )
        else:
            results['ct'] = []
            
        return results