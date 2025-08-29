"""RAG model wrapper for SDTM mapper - integrates with enhanced RAG system"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RAGModel:
    """Wrapper for enhanced RAG system to maintain compatibility"""
    
    def __init__(self, kb_path: Optional[Path] = None):
        self.enhanced_rag = None
        self.kb_path = kb_path
        
    def initialize(self):
        """Initialize RAG components"""
        if self.kb_path and self.kb_path.exists():
            try:
                from ..rag import EnhancedRAGSystem
                logger.info("Initializing Enhanced RAG System...")
                self.enhanced_rag = EnhancedRAGSystem(
                    kb_path=self.kb_path,
                    dense_model="BAAI/bge-m3",
                    reranker_model="BAAI/bge-reranker-base"
                )
                logger.info("Enhanced RAG system initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize enhanced RAG: {e}")
                logger.info("Falling back to basic RAG")
                self._init_basic_rag()
        else:
            logger.info("Using basic RAG (no KB path)")
            self._init_basic_rag()
    
    def _init_basic_rag(self):
        """Initialize basic RAG as fallback"""
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.test_code_embeddings = {}
        self.test_code_labels = []
        
    def search_test_code(self, query: str, domain: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for test codes using enhanced RAG
        
        Returns:
            List of (test_code, similarity_score) tuples
        """
        if self.enhanced_rag:
            # Use enhanced RAG system
            candidates = self.enhanced_rag.search(
                question=query,
                top_k=top_k
            )
            
            # Extract test codes from CT terms
            test_codes = []
            seen = set()
            
            for candidate in candidates:
                # Filter by domain if specified
                if domain and candidate.domain.domain_code != domain:
                    continue
                    
                # Get test codes from CT terms
                for ct_term in candidate.ct_terms:
                    if ct_term.submission_value not in seen:
                        test_codes.append((
                            ct_term.submission_value,
                            candidate.score  # Use overall score
                        ))
                        seen.add(ct_term.submission_value)
                        
            return test_codes[:top_k]
        else:
            # Fallback to basic search
            return self._basic_search(query, domain, top_k)
    
    def search_variables(self, query: str, domain: str = None, 
                        section: str = None, options: List[str] = None,
                        top_k: int = 10) -> List[Dict[str, any]]:
        """
        Search for variables using enhanced RAG
        
        Returns:
            List of variable candidates with metadata
        """
        if self.enhanced_rag:
            candidates = self.enhanced_rag.search(
                question=query,
                options=options,
                section=section,
                top_k=top_k
            )
            
            variables = []
            for candidate in candidates:
                # Filter by domain if specified
                if domain and candidate.domain.domain_code != domain:
                    continue
                    
                var_info = {
                    'name': candidate.variable.var_name,
                    'label': candidate.variable.label,
                    'role': candidate.variable.role,
                    'domain': candidate.variable.domain_code,
                    'definition': candidate.variable.definition,
                    'codelist': candidate.variable.codelist_name,
                    'score': candidate.score,
                    'ct_terms': [ct.submission_value for ct in candidate.ct_terms[:5]]
                }
                variables.append(var_info)
                
            return variables
        else:
            # Return empty list if no RAG
            return []
    
    def get_domain_suggestions(self, query: str, section: str = None,
                             form: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get domain suggestions using enhanced RAG
        
        Returns:
            List of (domain_code, score) tuples
        """
        if self.enhanced_rag:
            candidates = self.enhanced_rag.search(
                question=query,
                section=section,
                form_name=form,
                top_k=top_k * 3  # Get more to deduplicate
            )
            
            # Aggregate by domain
            domain_scores = {}
            for candidate in candidates:
                domain_code = candidate.domain.domain_code
                if domain_code not in domain_scores:
                    domain_scores[domain_code] = []
                domain_scores[domain_code].append(candidate.score)
            
            # Average scores per domain
            domain_results = []
            for domain_code, scores in domain_scores.items():
                avg_score = sum(scores) / len(scores)
                domain_results.append((domain_code, avg_score))
                
            # Sort by score
            domain_results.sort(key=lambda x: x[1], reverse=True)
            return domain_results[:top_k]
        else:
            return []
    
    def _basic_search(self, query: str, domain: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Basic search fallback"""
        if not hasattr(self, 'encoder'):
            return []
            
        # This would be the original basic implementation
        # For now, return empty
        return []
    
    # Backward compatibility methods
    def encode_test_codes(self, test_codes: Dict[str, Dict]):
        """Legacy method - no longer needed with enhanced RAG"""
        logger.info("encode_test_codes called - using enhanced RAG instead")