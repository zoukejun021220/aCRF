"""Enhanced RAG system with linking logic and scoring"""

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
    

class EnhancedRAGSystem:
    """Complete RAG system with retrieval, reranking, and linking"""
    
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
        'subject characteristic': ['SC'],
        # Eligibility sections
        'inclusion': ['IE'],
        'exclusion': ['IE'],
        'eligibility': ['IE'],
        # Visits
        'visit': ['SV'],
        'baseline': ['SV'],
        'screening': ['SV']
    }
    
    def __init__(self, kb_path: Path,
                 dense_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-base",
                 device: str = None):
        """
        Initialize enhanced RAG system
        
        Args:
            kb_path: Path to knowledge base
            dense_model: Model for dense retrieval
            reranker_model: Model for reranking
            device: Device to use
        """
        self.kb_path = Path(kb_path)
        
        # Build indices
        logger.info("Building CDISC indices...")
        self.index_builder = CDISCIndexBuilder(kb_path)
        self.domain_docs, self.variable_docs, self.ct_docs = self.index_builder.build_all_indices()
        # Fast domain lookup by code
        self._domain_by_code = {d.domain_code: d for d in self.domain_docs}
        
        # Get relationships
        self.ct_relationships = self.index_builder.get_ct_relationships()
        
        # Initialize retriever
        logger.info("Initializing hybrid retriever...")
        # Enable on-disk embedding cache under KB path
        cache_dir = self.kb_path / ".rag_cache"
        self.retriever = HybridRetriever(model_name=dense_model, device=device, cache_dir=cache_dir)
        self.retriever.index_documents(self.domain_docs, self.variable_docs, self.ct_docs)
        
        # Initialize reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
        
    def search(self, question: str, options: List[str] = None, 
              section: str = None, form_name: str = None,
              top_k: int = 5) -> List[RAGCandidate]:
        """
        End-to-end RAG search with linking
        
        Args:
            question: Question text
            options: Field options
            section: Section name
            form_name: Form name
            top_k: Number of final candidates to return
            
        Returns:
            List of linked RAG candidates
        """
        # Get target domains from section and form name
        target_domains = self._get_target_domains_from(section, form_name)
        
        # Build query
        query = self._build_query(question, options, section, form_name)
        
        # Add BGE instruction prefix for retrieval
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
            domain_candidates[:20] if domain_candidates else [],
            variable_candidates[:100] if variable_candidates else [],
            ct_candidates[:100] if ct_candidates else [],
            top_n_per_type={'domains': 10, 'variables': 50, 'ct': 50}
        )
        
        # Ensure section-prior domains are present in domain pool
        domain_tuples = reranked['domains']
        if target_domains:
            domain_tuples = self._inject_target_domains(domain_tuples, target_domains)

        # Link and score with proper constraints
        logger.info("Linking candidates...")
        final_candidates = self._link_and_score(
            query, options,
            domain_tuples,
            reranked['variables'],
            reranked['ct'],
            target_domains
        )
        
        # Sort by final score and return top K
        final_candidates.sort(key=lambda x: x.score, reverse=True)
        return final_candidates[:top_k]

    def search_with_details(self, question: str, options: List[str] = None,
                            section: str = None, form_name: str = None,
                            top_k: int = 3, ct_top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve and rerank with detailed candidates for domains, variables, and CT per variable.
        Returns a dict with 'domains', 'variables', 'ct_per_variable', and 'final_candidates'.
        """
        # Build query (same as search)
        query = self._build_query(question, options, section, form_name)
        retrieval_query = f"Represent this question for searching relevant passages: {query}"

        # Retrieve
        logger.info("Retrieving candidates...")
        domain_candidates = self.retriever.retrieve(retrieval_query, "domain", top_k=50, alpha=0.6)
        variable_candidates = self.retriever.retrieve(retrieval_query, "variable", top_k=300, alpha=0.6)
        ct_candidates = self.retriever.retrieve(retrieval_query, "ct", top_k=400, alpha=0.5)

        # Rerank
        logger.info("Reranking candidates...")
        reranked = self.reranker.rerank_multiple_indices(
            query,
            domain_candidates[:50] if domain_candidates else [],
            variable_candidates[:200] if variable_candidates else [],
            ct_candidates[:200] if ct_candidates else [],
            top_n_per_type={'domains': max(10, top_k), 'variables': max(50, top_k), 'ct': max(100, ct_top_k*10)}
        )

        # Inject section-prior domains if missing
        target_domains = self._get_target_domains_from(section, form_name)
        domain_tuples = reranked['domains']
        if target_domains:
            domain_tuples = self._inject_target_domains(domain_tuples, target_domains)
        variable_tuples = reranked['variables']
        ct_tuples = reranked['ct']

        # Prepare ct_map for quick lookup by codelist code/name
        ct_map: Dict[str, List[Tuple[CTDocument, float]]] = {}
        for ct_cand, ct_score in ct_tuples:
            ct_doc = ct_cand.doc if hasattr(ct_cand, 'doc') else ct_cand
            keys = set()
            if getattr(ct_doc, 'codelist_name', None):
                keys.add(str(ct_doc.codelist_name))
            if getattr(ct_doc, 'codelist_code', None):
                keys.add(str(ct_doc.codelist_code))
            if not keys and getattr(ct_doc, 'preferred_term', None):
                keys.add(str(ct_doc.preferred_term))
            for k in keys:
                ct_map.setdefault(k, []).append((ct_doc, float(ct_score)))

        # Top K domains and variables
        top_domains = []
        for d_cand, d_score in domain_tuples[:top_k]:
            d_doc = d_cand.doc if hasattr(d_cand, 'doc') else d_cand
            top_domains.append({'domain': d_doc, 'score': float(d_score)})

        top_variables = []
        for v_cand, v_score in variable_tuples[:top_k]:
            v_doc = v_cand.doc if hasattr(v_cand, 'doc') else v_cand
            top_variables.append({'variable': v_doc, 'score': float(v_score)})

        # CT suggestions per variable
        ct_per_variable: Dict[str, List[Dict[str, Any]]] = {}
        for item in top_variables:
            var_doc: VariableDocument = item['variable']

            # Build allowed codelists for this variable
            allowed_codelists: List[str] = []
            var_key = (var_doc.var_name, var_doc.domain_code)
            rel_lists = self.ct_relationships.get(var_key, [])
            if isinstance(rel_lists, dict):
                rel_lists = list(rel_lists.keys())
            if isinstance(rel_lists, list):
                allowed_codelists.extend(rel_lists)
            if var_doc.codelist_name:
                allowed_codelists.append(var_doc.codelist_name)

            # TEST/TESTCD support for findings results
            var_name_upper = (var_doc.var_name or '').upper()
            if var_name_upper.endswith("TEST"):
                testcd_var = var_doc.var_name[:-4] + "CD"
                testcd_key = (testcd_var, var_doc.domain_code)
                testcd_lists = self.ct_relationships.get(testcd_key, [])
                if isinstance(testcd_lists, dict):
                    testcd_lists = list(testcd_lists.keys())
                if isinstance(testcd_lists, list):
                    allowed_codelists.extend(testcd_lists)
            else:
                if any(var_name_upper.endswith(suf) for suf in ["ORRES", "STRESN", "STRESC", "STRESU"]) or \
                   any(k in var_name_upper for k in ["ORRES", "STRES", "RES"]):
                    for key in [
                        (f"{var_doc.domain_code}TEST", var_doc.domain_code),
                        (f"{var_doc.domain_code}TESTCD", var_doc.domain_code),
                        ("TEST", var_doc.domain_code),
                        ("TESTCD", var_doc.domain_code)
                    ]:
                        lists = self.ct_relationships.get(key, [])
                        if isinstance(lists, dict):
                            lists = list(lists.keys())
                        if isinstance(lists, list):
                            allowed_codelists.extend(lists)

            allowed_codelists = [str(cl) for cl in allowed_codelists if cl]
            ct_suggestions: List[Dict[str, Any]] = []

            # From retrieved ct_map
            seen_pairs = set()
            for codelist in set(allowed_codelists):
                if codelist in ct_map:
                    ct_list_sorted = sorted(ct_map[codelist], key=lambda x: x[1], reverse=True)
                    for ct_doc, s in ct_list_sorted:
                        key = (ct_doc.codelist_code, ct_doc.submission_value)
                        if key in seen_pairs:
                            continue
                        ct_suggestions.append({'ct': ct_doc, 'score': float(s)})
                        seen_pairs.add(key)
                        if len(ct_suggestions) >= ct_top_k:
                            break
                if len(ct_suggestions) >= ct_top_k:
                    break

            # Fallback inference if empty and options present
            if not ct_suggestions and options:
                inferred = self._infer_codelists_from_options(options)
                for cl in inferred:
                    if allowed_codelists and cl not in set(allowed_codelists):
                        continue
                    if cl in ct_map:
                        ct_list_sorted = sorted(ct_map[cl], key=lambda x: x[1], reverse=True)
                        for ct_doc, s in ct_list_sorted[:ct_top_k]:
                            ct_suggestions.append({'ct': ct_doc, 'score': float(s)})
                    else:
                        defaults = self._default_ct_terms_for(cl)
                        for d in defaults[:ct_top_k]:
                            ct_suggestions.append({'ct': d, 'score': 0.6})

            ct_per_variable[f"{var_doc.domain_code}.{var_doc.var_name}"] = ct_suggestions[:ct_top_k]

        # Final linked candidates (reuse search flow)
        final_candidates = self._link_and_score(
            query, options,
            domain_tuples,
            variable_tuples,
            ct_tuples,
            self._get_target_domains(section)
        )
        final_candidates.sort(key=lambda x: x.score, reverse=True)

        return {
            'domains': top_domains,
            'variables': top_variables,
            'ct_per_variable': ct_per_variable,
            'final_candidates': final_candidates[:top_k]
        }

    def _inject_target_domains(self, domain_tuples: List[Tuple[Any, float]], target_domains: List[str]) -> List[Tuple[Any, float]]:
        """Ensure that target domains (from section prior) are present in domain candidates.
        If missing, inject with a strong score above existing ones so they surface in top lists.
        """
        if not domain_tuples:
            domain_tuples = []
        present_codes = set()
        for cand, _ in domain_tuples:
            code = getattr(cand.doc, 'domain_code', None) if hasattr(cand, 'doc') else getattr(cand, 'domain_code', None)
            if code:
                present_codes.add(code)
        max_score = max([s for _, s in domain_tuples], default=0.0)
        inject_score = max_score + 2.0
        augmented = list(domain_tuples)
        for code in target_domains or []:
            if code not in present_codes and code in self._domain_by_code:
                augmented.insert(0, (self._domain_by_code[code], inject_score))
        return augmented
    
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
        # Remove checkbox/radio markers
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
    
    def _get_target_domains(self, section: str) -> List[str]:
        """Get target domains from section name"""
        if not section:
            return []
        section_lower = section.lower()
        for key, domains in self.SECTION_TO_DOMAINS.items():
            if key in section_lower:
                return domains
        return []

    def _get_target_domains_from(self, section: Optional[str], form_name: Optional[str]) -> List[str]:
        """Resolve target domains from section and form name."""
        # Prefer section mapping
        domains = self._get_target_domains(section) if section else []
        if domains:
            return domains
        # Fall back to form name cues
        if form_name:
            form_lower = form_name.lower()
            for key, d in self.SECTION_TO_DOMAINS.items():
                if key in form_lower:
                    return d
        return []
    
    def _link_and_score(self, query: str, options: List[str],
                       domain_candidates: List[Tuple[Any, float]],
                       variable_candidates: List[Tuple[Any, float]], 
                       ct_candidates: List[Tuple[Any, float]],
                       target_domains: List[str] = None) -> List[RAGCandidate]:
        """Link candidates and compute final scores with SDTM constraints"""
        final_candidates = []
        query_lower = query.lower()
        
        # Create lookup maps
        domain_map = {}
        for d_cand, d_score in domain_candidates:
            d_doc = d_cand.doc if hasattr(d_cand, 'doc') else d_cand
            domain_map[d_doc.domain_code] = (d_doc, d_score)
        
        ct_map = {}
        # Index CT candidates by both codelist name and code to improve linking
        for ct_cand, ct_score in ct_candidates:
            ct_doc = ct_cand.doc if hasattr(ct_cand, 'doc') else ct_cand
            keys = set()
            if getattr(ct_doc, 'codelist_name', None):
                keys.add(str(ct_doc.codelist_name))
            if getattr(ct_doc, 'codelist_code', None):
                keys.add(str(ct_doc.codelist_code))
            # Fallback to preferred term grouping if no codelist info
            if not keys and getattr(ct_doc, 'preferred_term', None):
                keys.add(str(ct_doc.preferred_term))
            for k in keys:
                if k not in ct_map:
                    ct_map[k] = []
                ct_map[k].append((ct_doc, ct_score))
        
        # Process each variable candidate
        for var_cand, var_score in variable_candidates:
            var_doc = var_cand.doc if hasattr(var_cand, 'doc') else var_cand
            
            # Get domain
            domain_doc, domain_score = domain_map.get(
                var_doc.domain_code, 
                (None, 0.0)
            )
            if not domain_doc:
                # Create a basic domain doc if missing
                domain_doc = DomainDocument(
                    domain_code=var_doc.domain_code,
                    domain_name=var_doc.domain_code,
                    domain_class="SPECIAL PURPOSE",
                    definition=""
                )
                domain_score = 0.0
                
            # Get allowed CT terms
            ct_terms = []
            ct_score = 0.0
            
            # Check CT relationships
            var_key = (var_doc.var_name, var_doc.domain_code)
            allowed_codelists = self.ct_relationships.get(var_key, [])
            
            # Ensure allowed_codelists is a list
            if isinstance(allowed_codelists, dict):
                allowed_codelists = list(allowed_codelists.keys())
            elif not isinstance(allowed_codelists, list):
                allowed_codelists = []
            
            # Also check if variable has codelist
            if var_doc.codelist_name:
                allowed_codelists.append(var_doc.codelist_name)
                
            # Check for paired TEST/TESTCD (both directions)
            if var_doc.var_name.endswith("TEST"):
                testcd_var = var_doc.var_name[:-4] + "CD"
                testcd_key = (testcd_var, var_doc.domain_code)
                testcd_codelists = self.ct_relationships.get(testcd_key, [])
                if isinstance(testcd_codelists, dict):
                    testcd_codelists = list(testcd_codelists.keys())
                elif isinstance(testcd_codelists, list):
                    allowed_codelists.extend(testcd_codelists)
            else:
                # For result variables (ORRES/STRES* etc.) in Findings domains, 
                # bring in TEST/TESTCD codelists for the same domain.
                var_name_upper = (var_doc.var_name or "").upper()
                if any(var_name_upper.endswith(suf) for suf in ["ORRES", "STRESN", "STRESC", "STRESU"]) or \
                   any(k in var_name_upper for k in ["ORRES", "STRES", "RES"]):
                    # Add TEST and TESTCD relationships for this domain
                    test_key = (f"{var_doc.domain_code}TEST", var_doc.domain_code)
                    testcd_key = (f"{var_doc.domain_code}TESTCD", var_doc.domain_code)
                    # Additionally try plain TEST/TESTCD variable names
                    alt_test_key = ("TEST", var_doc.domain_code)
                    alt_testcd_key = ("TESTCD", var_doc.domain_code)
                    for key in [test_key, testcd_key, alt_test_key, alt_testcd_key]:
                        lists = self.ct_relationships.get(key, [])
                        if isinstance(lists, dict):
                            lists = list(lists.keys())
                        if isinstance(lists, list):
                            allowed_codelists.extend(lists)
                
            # Get CT terms from retrieved pool
            allowed_codelists = [str(cl) for cl in allowed_codelists if cl]
            for codelist in set(allowed_codelists):
                if codelist in ct_map:
                    ct_list = ct_map[codelist]
                    if ct_list:
                        # Take top-k CT terms to expose plausible fill values
                        ct_list_sorted = sorted(ct_list, key=lambda x: x[1], reverse=True)
                        top_ct = [t[0] for t in ct_list_sorted[:3]]
                        # De-duplicate by (codelist_code, submission_value)
                        seen = set((t.codelist_code, t.submission_value) for t in ct_terms)
                        for t in top_ct:
                            key = (t.codelist_code, t.submission_value)
                            if key not in seen:
                                ct_terms.append(t)
                                seen.add(key)
                        ct_score = max(ct_score, ct_list_sorted[0][1])

            # Fallback: infer CT from options if none linked yet
            if not ct_terms and options:
                inferred_lists = self._infer_codelists_from_options(options)
                for cl in inferred_lists:
                    # Respect allowed lists if present
                    if allowed_codelists and cl not in set(allowed_codelists):
                        continue
                    if cl in ct_map and ct_map[cl]:
                        # Take top 2 terms as suggestions
                        sorted_terms = sorted(ct_map[cl], key=lambda x: x[1], reverse=True)
                        ct_terms.extend([t[0] for t in sorted_terms[:2]])
                        ct_score = max(ct_score, sorted_terms[0][1])
                    else:
                        # Provide sensible defaults for common codelists
                        defaults = self._default_ct_terms_for(cl)
                        if defaults:
                            ct_terms.extend(defaults)
                            # Assign a modest confidence for inferred CT
                            ct_score = max(ct_score, 0.6)
            
            # Calculate structural bonuses based on SDTM rules
            suffix_bonus = self._calculate_suffix_bonus(var_doc, query_lower)
            role_bonus = self._calculate_role_bonus_sdtm(var_doc, query_lower, options)
            intent_bonus = self._calculate_intent_bonus(var_doc, query_lower)
            section_bonus = 0.0
            unit_bonus = self._calculate_unit_bonus(var_doc, query_lower)
            
            # Strong section bonus if in target domains
            if target_domains and var_doc.domain_code in target_domains:
                section_bonus = 1.0
            
            # Normalize scores to 0-1 range
            var_score_norm = max(0, min(1, var_score))
            domain_score_norm = max(0, min(1, domain_score))
            ct_score_norm = max(0, min(1, ct_score))
            
            # Aggregate rule bonuses (role/suffix/section/unit) per recipe (10%)
            rule_bonus = min(1.0, (suffix_bonus + role_bonus + section_bonus + unit_bonus))

            # Scoring formula per recommended recipe
            score_breakdown = {
                'crossencoder_var': var_score_norm * 0.45,  # Variable importance
                'ct': ct_score_norm * 0.25,                 # CT alignment (retrieved or inferred)
                'domain': domain_score_norm * 0.20,         # Domain score
                'rule_bonus': rule_bonus * 0.10             # Role/suffix/section/unit
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

    def _infer_codelists_from_options(self, options: List[str]) -> List[str]:
        """Infer likely codelist codes from provided options."""
        if not options:
            return []
        opts = {str(o).strip().lower() for o in options}
        inferred = []
        # Yes/No/Unknown patterns → NY (C66742)
        yn_keys = {'yes', 'no', 'unknown', 'not applicable', 'not done', 'na', 'nd', 'u'}
        if any(k in opts for k in yn_keys):
            inferred.append('C66742')
        # Severity scale → AESEV (C66769)
        sev_keys = {'mild', 'moderate', 'severe'}
        if any(k in opts for k in sev_keys):
            inferred.append('C66769')
        return inferred

    def _default_ct_terms_for(self, codelist_code: str) -> List[CTDocument]:
        """Provide default CT term stubs for common codelists when retrieval is empty."""
        defaults: List[CTDocument] = []
        code = (codelist_code or '').upper()
        try:
            if code == 'C66742':  # NY codelist
                defaults = [
                    CTDocument(codelist_name='C66742', codelist_code='C66742', submission_value='Y', preferred_term='Yes', definition='Yes'),
                    CTDocument(codelist_name='C66742', codelist_code='C66742', submission_value='N', preferred_term='No', definition='No'),
                    CTDocument(codelist_name='C66742', codelist_code='C66742', submission_value='U', preferred_term='Unknown', definition='Unknown')
                ]
            elif code == 'C66769':  # Severity
                defaults = [
                    CTDocument(codelist_name='C66769', codelist_code='C66769', submission_value='MILD', preferred_term='Mild', definition='Mild'),
                    CTDocument(codelist_name='C66769', codelist_code='C66769', submission_value='MODERATE', preferred_term='Moderate', definition='Moderate'),
                    CTDocument(codelist_name='C66769', codelist_code='C66769', submission_value='SEVERE', preferred_term='Severe', definition='Severe')
                ]
        except Exception:
            defaults = []
        return defaults
    
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

    def _calculate_unit_bonus(self, var_doc: VariableDocument, query_lower: str) -> float:
        """Lightweight unit-aware bonus. If question mentions units (e.g., mg, mmol/L)
        and variable is a result qualifier, boost slightly.
        """
        unit_patterns = [
            'mg', 'g', 'kg', 'ml', 'l', 'mmol', 'umol', 'µmol', 'mcg', 'ug', '%', 'bpm', 'mmhg', '/l', '/dl', '/ul'
        ]
        if any(up in query_lower for up in unit_patterns):
            role_lower = (var_doc.role or '').lower()
            if 'result' in role_lower or 'qualifier' in role_lower:
                return 0.5
        return 0.0
