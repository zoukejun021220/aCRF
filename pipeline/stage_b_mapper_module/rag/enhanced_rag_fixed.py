"""Enhanced RAG system with proper hybrid retrieval and SDTM constraints"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re

from .indices.schemas import DomainDocument, VariableDocument, CTDocument
from .indices.index_builder import CDISCIndexBuilder
from .retrievers.hybrid_retriever import HybridRetriever
from .rerankers.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class RAGCandidate:
    """Final RAG candidate with linked domain, variable, and CT"""
    domain: DomainDocument
    variable: VariableDocument
    ct_terms: List[CTDocument]
    score: float
    score_breakdown: Dict[str, float]
    

class EnhancedRAGFixed:
    """Enhanced RAG with proper constraints and scoring"""
    
    # Domain mapping from sections
    SECTION_TO_DOMAINS = {
        'adverse event': ['AE'],
        'adverse': ['AE'],
        'demographics': ['DM'],
        'demographic': ['DM'],
        'laboratory': ['LB'],
        'lab': ['LB'],
        'concomitant medication': ['CM'],
        'conmed': ['CM'],
        'medical history': ['MH'],
        'vital signs': ['VS'],
        'vital': ['VS'],
        'physical examination': ['PE'],
        'exposure': ['EX'],
        'disposition': ['DS'],
        'subject characteristic': ['SC']
    }
    
    def __init__(self, kb_path: Path,
                 dense_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-base",
                 device: str = None):
        """Initialize enhanced RAG system"""
        self.kb_path = Path(kb_path)
        
        # Build indices
        logger.info("Building CDISC indices...")
        self.index_builder = CDISCIndexBuilder(kb_path)
        self.domain_docs, self.variable_docs, self.ct_docs = self.index_builder.build_all_indices()
        
        # Get relationships
        self.ct_relationships = self.index_builder.get_ct_relationships()
        
        # Initialize retriever
        logger.info("Initializing hybrid retriever...")
        self.retriever = HybridRetriever(model_name=dense_model, device=device)
        self.retriever.index_documents(self.domain_docs, self.variable_docs, self.ct_docs)
        
        # Initialize reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
        
    def search(self, question: str, options: List[str] = None, 
              section: str = None, form_name: str = None,
              top_k: int = 5) -> List[RAGCandidate]:
        """End-to-end RAG search with proper constraints"""
        
        # Get target domains from section
        target_domains = self._get_target_domains(section)
        
        # Build query with BGE instruction prefix
        query = self._build_query(question, options, section, form_name)
        retrieval_query = f"Represent this question for searching relevant passages: {query}"
        
        # Retrieve from all indices with true hybrid (60% dense, 40% BM25)
        logger.info("Retrieving candidates...")
        domain_candidates = self.retriever.retrieve(retrieval_query, "domain", top_k=50, alpha=0.6)
        variable_candidates = self.retriever.retrieve(retrieval_query, "variable", top_k=300, alpha=0.6)
        ct_candidates = self.retriever.retrieve(retrieval_query, "ct", top_k=200, alpha=0.5)
        
        # Filter variables by target domains (keep 90% from target, 10% spillover)
        if target_domains:
            filtered_vars = []
            other_vars = []
            
            for var_cand in variable_candidates:
                var_doc = var_cand.doc if hasattr(var_cand, 'doc') else var_cand
                if var_doc.domain_code in target_domains:
                    filtered_vars.append(var_cand)
                else:
                    other_vars.append(var_cand)
            
            # Take all from target domains + 10% from others
            spillover_count = max(5, int(len(filtered_vars) * 0.1))
            variable_candidates = filtered_vars + other_vars[:spillover_count]
        
        # Rerank constrained pools
        logger.info("Reranking candidates...")
        reranked = self.reranker.rerank_multiple_indices(
            query,  # Use original query for reranking
            domain_candidates[:20],
            variable_candidates[:100],
            ct_candidates[:100],
            top_n_per_type={'domains': 10, 'variables': 50, 'ct': 50}
        )
        
        # Link and score with proper constraints
        logger.info("Linking candidates...")
        final_candidates = self._link_and_score_constrained(
            query, options,
            reranked['domains'],
            reranked['variables'],
            reranked['ct'],
            target_domains
        )
        
        # Sort by final score
        final_candidates.sort(key=lambda x: x.score, reverse=True)
        return final_candidates[:top_k]
    
    def _get_target_domains(self, section: str) -> List[str]:
        """Get target domains from section name"""
        if not section:
            return []
        
        section_lower = section.lower()
        for key, domains in self.SECTION_TO_DOMAINS.items():
            if key in section_lower:
                return domains
        return []
    
    def _build_query(self, question: str, options: List[str] = None,
                    section: str = None, form_name: str = None) -> str:
        """Build enriched query"""
        query_parts = [f"Q: {question}"]
        
        if options:
            # Normalize options
            normalized_options = [self._normalize_option(opt) for opt in options]
            query_parts.append(f"Options/Format: {', '.join(normalized_options)}")
            
            # Add CT hints
            enriched_query = self.retriever.add_ct_hints_to_query(
                "\n".join(query_parts), options
            )
            query_parts = [enriched_query]
            
        if section:
            query_parts.append(f"Section: {section}")
            
        if form_name:
            query_parts.append(f"Form: {form_name}")
            
        # Add format hints
        format_hints = self._infer_format_hints(question, options)
        if format_hints:
            query_parts.append(f"Format hints: {format_hints}")
            
        return "\n".join(query_parts)
    
    def _normalize_option(self, option: str) -> str:
        """Normalize option text"""
        option = re.sub(r'^[\[\]○●□■\(\)]+\s*', '', option)
        return option.strip()
    
    def _infer_format_hints(self, question: str, options: List[str] = None) -> str:
        """Infer format hints from question and options"""
        hints = []
        question_lower = question.lower()
        
        # Date/time patterns
        if any(pattern in question_lower for pattern in ['date', 'time', 'when', 'onset', 'start']):
            hints.append("date/time → Timing role, --DTC suffix")
            
        # Result patterns  
        if any(pattern in question_lower for pattern in ['result', 'value', 'measurement']):
            hints.append("result → Result Qualifier role, ORRES/STRES*")
            
        # Unit patterns
        unit_patterns = r'(\d+\s*(mg|g|kg|ml|l|mmol|μg|mcg|%|bpm|mmhg))'
        if re.search(unit_patterns, question_lower):
            hints.append("numeric with units → Result Qualifier, ORRES/STRES*")
            
        # Yes/No patterns
        if options and any(opt.lower() in ['yes', 'no', 'unknown'] for opt in options):
            hints.append("Yes/No/Unknown → NY codelist (C66742)")
            
        return "; ".join(hints)
    
    def _link_and_score_constrained(self, query: str, options: List[str],
                                   domain_candidates: List[Tuple[Any, float]],
                                   variable_candidates: List[Tuple[Any, float]], 
                                   ct_candidates: List[Tuple[Any, float]],
                                   target_domains: List[str]) -> List[RAGCandidate]:
        """Link and score with proper SDTM constraints"""
        final_candidates = []
        query_lower = query.lower()
        
        # Create lookup maps
        domain_map = {}
        for d_cand, d_score in domain_candidates:
            d_doc = d_cand.doc if hasattr(d_cand, 'doc') else d_cand
            domain_map[d_doc.domain_code] = (d_doc, d_score)
        
        # Create CT map restricted to allowed codelists
        ct_map = {}
        for ct_cand, ct_score in ct_candidates:
            ct_doc = ct_cand.doc if hasattr(ct_cand, 'doc') else ct_cand
            if ct_doc.codelist_name not in ct_map:
                ct_map[ct_doc.codelist_name] = []
            ct_map[ct_doc.codelist_name].append((ct_doc, ct_score))
        
        # Process each variable
        for var_cand, var_score in variable_candidates:
            var_doc = var_cand.doc if hasattr(var_cand, 'doc') else var_cand
            
            # Get domain
            domain_doc, domain_score = domain_map.get(var_doc.domain_code, (None, 0.0))
            if not domain_doc:
                domain_doc = DomainDocument(
                    domain_code=var_doc.domain_code,
                    domain_name=var_doc.domain_code,
                    domain_class="SPECIAL PURPOSE",
                    definition=""
                )
                domain_score = 0.0
            
            # Get CT terms restricted to allowed codelists
            ct_terms = []
            ct_score = 0.0
            
            # Get allowed codelists for this variable
            var_key = (var_doc.var_name, var_doc.domain_code)
            allowed_codelists = self.ct_relationships.get(var_key, [])
            
            if isinstance(allowed_codelists, dict):
                allowed_codelists = list(allowed_codelists.keys())
            elif not isinstance(allowed_codelists, list):
                allowed_codelists = []
            
            if var_doc.codelist_name:
                allowed_codelists.append(var_doc.codelist_name)
            
            # Check for paired TEST/TESTCD
            if var_doc.var_name.endswith("TEST"):
                testcd_var = var_doc.var_name[:-4] + "CD"
                testcd_key = (testcd_var, var_doc.domain_code)
                testcd_codelists = self.ct_relationships.get(testcd_key, [])
                if isinstance(testcd_codelists, dict):
                    testcd_codelists = list(testcd_codelists.keys())
                elif isinstance(testcd_codelists, list):
                    allowed_codelists.extend(testcd_codelists)
            
            # Get best CT from allowed codelists only
            allowed_codelists = [str(cl) for cl in allowed_codelists if cl]
            for codelist in set(allowed_codelists):
                if codelist in ct_map:
                    ct_list = ct_map[codelist]
                    if ct_list:
                        best_ct = max(ct_list, key=lambda x: x[1])
                        ct_terms.append(best_ct[0])
                        ct_score = max(ct_score, best_ct[1])
            
            # Calculate structural bonuses based on SDTM rules
            suffix_bonus = self._calculate_suffix_bonus(var_doc, query_lower)
            role_bonus = self._calculate_role_bonus_sdtm(var_doc, query_lower, options)
            intent_bonus = self._calculate_intent_bonus(var_doc, query_lower)
            section_bonus = 0.0
            
            # Strong section bonus if in target domains
            if target_domains and var_doc.domain_code in target_domains:
                section_bonus = 1.0
            
            # Normalize scores
            var_score_norm = max(0, min(1, var_score))
            domain_score_norm = max(0, min(1, domain_score))
            ct_score_norm = max(0, min(1, ct_score))
            
            # New scoring formula with proper weights
            score_breakdown = {
                'crossencoder_var': var_score_norm * 0.40,  # Cross-encoder score
                'ct': ct_score_norm * 0.25,                 # CT alignment
                'domain': domain_score_norm * 0.15,         # Domain score
                'suffix_bonus': suffix_bonus * 0.10,        # SDTM suffix rules
                'role_bonus': role_bonus * 0.05,            # Role alignment
                'section_prior': section_bonus * 0.05       # Section prior
            }
            
            # Add intent bonus separately (can push score > 1.0 for perfect matches)
            if intent_bonus > 0:
                score_breakdown['intent_bonus'] = intent_bonus
            
            final_score = sum(score_breakdown.values())
            
            candidate = RAGCandidate(
                domain=domain_doc,
                variable=var_doc,
                ct_terms=ct_terms,
                score=final_score,
                score_breakdown=score_breakdown
            )
            
            final_candidates.append(candidate)
        
        return final_candidates
    
    def _calculate_suffix_bonus(self, var_doc: VariableDocument, query_lower: str) -> float:
        """Calculate bonus based on SDTM suffix rules"""
        var_name = var_doc.var_name
        
        # Strong bonuses for suffix matches
        if var_name.endswith("DTC") and any(p in query_lower for p in ['date', 'time', 'when', 'onset', 'start', 'birth']):
            return 0.7
        
        if var_name.endswith("STDTC") and any(p in query_lower for p in ['start', 'onset', 'begin']):
            return 0.8
        
        if var_name.endswith("ENDTC") and any(p in query_lower for p in ['end', 'stop']):
            return 0.8
            
        if (var_name.endswith("ORRES") or var_name.endswith("STRESN") or var_name.endswith("STRESC")) and \
           any(p in query_lower for p in ['result', 'value', 'measurement', 'finding']):
            return 0.7
            
        if var_name.endswith("TEST") and any(p in query_lower for p in ['test', 'examination', 'assessment']):
            return 0.3  # Lower bonus for TEST (we want ORRES for results)
            
        if var_name.endswith("TRT") and any(p in query_lower for p in ['treatment', 'medication', 'drug', 'therapy']):
            return 0.7
            
        return 0.0
    
    def _calculate_role_bonus_sdtm(self, var_doc: VariableDocument, query_lower: str, options: List[str]) -> float:
        """Calculate role bonus based on SDTM conventions"""
        role_lower = var_doc.role.lower() if var_doc.role else ""
        
        # Timing role for dates
        if 'timing' in role_lower and any(p in query_lower for p in ['date', 'time', 'when']):
            return 1.0
        
        # Result qualifier for results
        if 'result' in role_lower and any(p in query_lower for p in ['result', 'value', 'finding']):
            return 1.0
            
        # Record qualifier for Yes/No
        if 'record qualifier' in role_lower and options and \
           any(opt.lower() in ['yes', 'no', 'unknown'] for opt in options):
            return 0.8
            
        # Topic for main subject
        if 'topic' in role_lower and any(p in query_lower for p in ['name', 'what', 'type']):
            return 0.5
            
        return 0.0
    
    def _calculate_intent_bonus(self, var_doc: VariableDocument, query_lower: str) -> float:
        """Calculate intent-based heuristic bonus"""
        var_name = var_doc.var_name
        
        # Very strong intent matches
        if 'adverse event' in query_lower and 'onset' in query_lower and var_name == 'AESTDTC':
            return 0.7
            
        if 'hospitalized' in query_lower and var_name in ['AEHOSP', 'AESHOSP']:
            return 0.7
            
        if 'hemoglobin' in query_lower and 'result' in query_lower and var_name == 'LBORRES':
            return 0.7
            
        if 'birth' in query_lower and 'date' in query_lower and var_name == 'BRTHDTC':
            return 0.7
            
        if 'medication' in query_lower and 'name' in query_lower and var_name == 'CMTRT':
            return 0.7
            
        return 0.0