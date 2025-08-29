#!/usr/bin/env python3
"""
SDTM RAG Retriever with BM25 + Vector Reranking
Optimized for precise SDTM variable retrieval
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from dataclasses import dataclass
import re
from .sdtm_annotation_rules import SDTMAnnotationRules, AnnotationType, create_annotation_rules

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    chunk_id: str
    chunk_type: str
    domain: str
    variable: str
    full_name: str
    content: str
    score: float
    bm25_score: float
    vector_score: float
    metadata: Dict[str, Any]
    annotation_rules: Optional[Dict[str, Any]] = None


class SDTMRAGRetriever:
    """SDTM-focused RAG retriever with hybrid search"""
    
    def __init__(
        self, 
        kb_dir: Path,
        encoder_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = True,
        include_rules: bool = True
    ):
        self.kb_dir = Path(kb_dir)
        self.include_rules = include_rules
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Initialize encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(encoder_model, device=device)
        
        # Build search indices
        self._build_indices()
        
        # Initialize annotation rules if enabled
        if self.include_rules:
            self.annotation_rules = create_annotation_rules()
        
    def _load_knowledge_base(self):
        """Load RAG chunks and indices"""
        
        # Load chunks
        chunks_path = self.kb_dir / "sdtm_rag_chunks.json"
        with open(chunks_path) as f:
            self.chunks = json.load(f)
            
        # Load alias index
        alias_path = self.kb_dir / "sdtm_alias_index.json"
        with open(alias_path) as f:
            self.alias_index = json.load(f)
            
        # Load domain index  
        domain_path = self.kb_dir / "sdtm_domain_index.json"
        with open(domain_path) as f:
            self.domain_index = json.load(f)
            
        logger.info(f"Loaded {len(self.chunks)} RAG chunks")
        
    def _build_indices(self):
        """Build BM25 and vector indices"""
        
        # Prepare texts for BM25
        self.chunk_texts = []
        self.chunk_lookup = {}
        
        for i, chunk in enumerate(self.chunks):
            # Extract metadata fields to top level for easier access
            metadata = chunk.get('metadata', {})
            chunk['domain'] = metadata.get('domain', '')
            chunk['variable'] = metadata.get('variable', '')
            chunk['full_name'] = metadata.get('full_name', '')
            chunk['chunk_type'] = metadata.get('chunk_type', '')
            chunk['chunk_id'] = chunk.get('id', f'chunk_{i}')
            
            # Combine searchable text
            text = chunk.get('search_text', '')
            if not text:
                text = f"{chunk.get('content', '')} {chunk.get('full_name', '')}"
            
            self.chunk_texts.append(text.lower().split())
            self.chunk_lookup[i] = chunk
            
        # Build BM25 index
        self.bm25 = BM25Okapi(self.chunk_texts)
        
        # Build vector embeddings
        logger.info("Building vector embeddings...")
        contents = [chunk.get('content', '') for chunk in self.chunks]
        self.embeddings = self.encoder.encode(
            contents, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
    def retrieve(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        alpha: float = 0.5  # Weight for BM25 vs vector (0.5 = equal)
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant SDTM variables for a query
        
        Args:
            query: Search query (e.g., CRF question text)
            domain_filter: Optional domain to filter results
            top_k: Number of candidates to retrieve before reranking
            rerank_top_k: Final number of results after reranking
            alpha: Weight balance between BM25 and vector scores
            
        Returns:
            List of retrieval results sorted by relevance
        """
        
        # Preprocess query
        query_processed = self._preprocess_query(query)
        
        # Check for direct alias matches
        alias_matches = self._check_aliases(query_processed)
        
        # BM25 retrieval
        query_tokens = query_processed.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k BM25 results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Vector similarity for reranking
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        # Score and filter candidates
        candidates = []
        seen_variables = set()
        
        for idx in top_indices:
            chunk = self.chunk_lookup[idx]
            
            # Apply domain filter if specified
            if domain_filter and chunk['domain'] != domain_filter:
                continue
                
            # Calculate vector similarity
            vector_score = util.cos_sim(
                query_embedding, 
                self.embeddings[idx]
            ).item()
            
            # Combine scores
            bm25_score = bm25_scores[idx]
            combined_score = alpha * bm25_score + (1 - alpha) * vector_score
            
            # Boost score for direct alias matches
            if chunk['full_name'] in alias_matches:
                combined_score *= 1.5
                
            # Boost score for exact variable name matches
            if self._check_exact_match(query, chunk):
                combined_score *= 1.3
                
            # Prepare annotation rules if enabled
            annotation_rules = None
            if self.include_rules:
                annotation_rules = self._get_relevant_rules(chunk, query)
                
            result = RetrievalResult(
                chunk_id=chunk['chunk_id'],
                chunk_type=chunk['chunk_type'],
                domain=chunk['domain'],
                variable=chunk['variable'],
                full_name=chunk['full_name'],
                content=chunk['content'],
                score=combined_score,
                bm25_score=bm25_score,
                vector_score=vector_score,
                metadata=chunk.get('metadata', {}),
                annotation_rules=annotation_rules
            )
            
            # Deduplicate by variable (keep highest scoring)
            var_key = f"{result.domain}.{result.variable}"
            if var_key not in seen_variables:
                candidates.append(result)
                seen_variables.add(var_key)
                
        # Sort by combined score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Return top results after reranking
        return candidates[:rerank_top_k]
        
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Expand common abbreviations
        abbreviations = {
            "DOB": "date of birth",
            "M/F": "male female sex",
            "HR": "heart rate pulse",
            "BP": "blood pressure",
            "RR": "respiratory rate",
            "AE": "adverse event",
            "SAE": "serious adverse event",
            "Con Med": "concomitant medication",
            "Lab": "laboratory",
            "I/E": "inclusion exclusion",
            "Med Hx": "medical history"
        }
        
        query_lower = query.lower()
        for abbr, expansion in abbreviations.items():
            if abbr.lower() in query_lower:
                query = query.replace(abbr, expansion)
                query = query.replace(abbr.lower(), expansion)
                
        return query
        
    def _check_aliases(self, query: str) -> set:
        """Check for direct alias matches"""
        
        matches = set()
        query_lower = query.lower()
        
        # Check each word and phrase in the query
        words = query_lower.split()
        
        # Check individual words
        for word in words:
            if word in self.alias_index:
                matches.update(self.alias_index[word])
                
        # Check bigrams and trigrams
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                if phrase in self.alias_index:
                    matches.update(self.alias_index[phrase])
                    
        return matches
        
    def _check_exact_match(self, query: str, chunk: Dict) -> bool:
        """Check if query contains exact variable name"""
        
        query_upper = query.upper()
        variable = chunk['variable']
        
        # Check if variable name appears as a word
        if re.search(r'\b' + re.escape(variable) + r'\b', query_upper):
            return True
            
        # Check full name (DOMAIN.VARIABLE)
        full_name = chunk['full_name']
        if full_name.upper() in query_upper:
            return True
            
        return False
        
    def _get_relevant_rules(self, chunk: Dict, query: str) -> Dict[str, Any]:
        """Get relevant annotation rules for a chunk based on context"""
        
        rules_context = {
            "domain": chunk['domain'],
            "variable": chunk['variable'],
            "chunk_type": chunk['chunk_type'],
            "rules_summary": self.annotation_rules.get_rules_summary(),
            "specific_rules": [],
            "patterns": {},
            "examples": []
        }
        
        # Determine context based on domain and chunk type
        context = chunk['domain'].lower()
        
        # Special contexts
        if chunk['domain'] in ['VS', 'LB', 'EG', 'QS', 'FT', 'RS']:
            context = 'findings'
        elif 'SUPP' in chunk.get('metadata', {}).get('related_domains', []):
            context = 'supplemental'
        elif chunk['domain'] == 'DM':
            context = 'demographics'
        elif chunk['domain'] == 'AE':
            context = 'adverse_events'
            
        # Get relevant rules for context
        relevant_rules = self.annotation_rules.get_rules_for_context(context)
        
        for rule in relevant_rules:
            rules_context['specific_rules'].append({
                'rule_id': rule.rule_id,
                'title': rule.title,
                'description': rule.description,
                'examples': rule.examples
            })
            
        # Add specific patterns based on chunk type
        if context == 'findings':
            rules_context['patterns']['findings'] = {
                'pattern_a': "VSORRES/VSORRESU when VSTESTCD = TEMP",
                'pattern_b': "VSTESTCD = TEMP (near label) ... VSORRES/VSORRESU (near result)",
                'instruction': "Use Pattern A when space allows, Pattern B for crowded pages"
            }
            
        elif context == 'supplemental':
            rules_context['patterns']['supplemental'] = {
                'pattern': f"{chunk['variable']}OTH in SUPP{chunk['domain']}",
                'instruction': "For 'Other, specify' fields mapping to supplemental qualifiers"
            }
            
        # Add formatting examples specific to this variable
        if chunk['chunk_type'] == 'variable_definition':
            var_examples = []
            
            # Single variable
            var_examples.append(chunk['variable'])
            
            # With units (for findings)
            if context == 'findings' and 'ORRES' in chunk['variable']:
                var_examples.append(f"{chunk['variable']}/{chunk['variable'].replace('ORRES', 'ORRESU')}")
                
            # With test code (for findings)
            if context == 'findings':
                testcd_var = chunk['variable'].replace('ORRES', 'TESTCD').replace('STRESC', 'TESTCD')
                var_examples.append(f"{chunk['variable']} when {testcd_var} = EXAMPLE")
                
            rules_context['examples'] = var_examples
            
        return rules_context
        
    def get_variable_context(self, full_name: str) -> Dict[str, Any]:
        """Get full context for a specific variable"""
        
        context = {
            "variable": full_name,
            "chunks": [],
            "patterns": [],
            "synonyms": []
        }
        
        # Find all chunks for this variable
        for chunk in self.chunks:
            if chunk['full_name'] == full_name:
                if chunk['chunk_type'] == 'variable_definition':
                    context['definition'] = chunk['content']
                    context['metadata'] = chunk['metadata']
                elif chunk['chunk_type'] == 'synonym_mapping':
                    context['synonyms'].append(chunk['metadata']['synonym'])
                elif chunk['chunk_type'] == 'when_then_pattern':
                    context['patterns'].append(chunk['metadata']['pattern'])
                    
                context['chunks'].append(chunk)
                
        return context


class SDTMKnowledgeTool:
    """Tool for Qwen-Agent style integration"""
    
    def __init__(self, retriever: SDTMRAGRetriever):
        self.retriever = retriever
        self.name = "sdtm_knowledge"
        self.description = "Search SDTM knowledge base for variable mappings"
        
    def call(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Call the knowledge tool
        
        Args:
            query: Search query
            **kwargs: Additional parameters (domain_filter, top_k, etc.)
            
        Returns:
            List of formatted results for the agent
        """
        
        # Retrieve results
        results = self.retriever.retrieve(
            query,
            domain_filter=kwargs.get('domain_filter'),
            top_k=kwargs.get('top_k', 10),
            rerank_top_k=kwargs.get('rerank_top_k', 5)
        )
        
        # Format for agent consumption
        formatted_results = []
        for result in results:
            formatted_result = {
                "variable": result.full_name,
                "domain": result.domain,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata,
                "type": result.chunk_type
            }
            
            # Include annotation rules if available
            if result.annotation_rules:
                formatted_result["annotation_rules"] = result.annotation_rules
                formatted_result["annotation_guidance"] = self._format_annotation_guidance(result)
                
            formatted_results.append(formatted_result)
            
        return formatted_results
    
    def _format_annotation_guidance(self, result: RetrievalResult) -> str:
        """Format annotation guidance as concise instructions"""
        
        guidance = []
        rules = result.annotation_rules
        
        # Add domain annotation format
        guidance.append(f"Domain annotation: {rules['domain']} ({self._get_domain_label(rules['domain'])})")
        
        # Add variable annotation examples
        if rules['examples']:
            guidance.append(f"Variable annotations: {', '.join(rules['examples'][:3])}")
            
        # Add specific patterns
        if 'findings' in rules['patterns']:
            findings = rules['patterns']['findings']
            guidance.append(f"Findings: {findings['instruction']}")
            
        if 'supplemental' in rules['patterns']:
            supp = rules['patterns']['supplemental']
            guidance.append(f"Supplemental: {supp['pattern']}")
            
        # Add key rules
        key_rules = []
        for rule in rules['specific_rules'][:3]:  # Top 3 most relevant
            key_rules.append(f"• {rule['title']}")
            
        if key_rules:
            guidance.append("Key rules:\n" + "\n".join(key_rules))
            
        return "\n".join(guidance)
    
    def _get_domain_label(self, domain: str) -> str:
        """Get standard domain label"""
        
        domain_labels = {
            "DM": "Demographics",
            "VS": "Vital Signs",
            "AE": "Adverse Events",
            "CM": "Concomitant Medications",
            "EX": "Exposure",
            "LB": "Laboratory",
            "MH": "Medical History",
            "QS": "Questionnaires",
            "DS": "Disposition",
            "SU": "Substance Use",
            "EC": "Exposure as Collected",
            "EG": "ECG",
            "PE": "Physical Examination",
            "IE": "Inclusion/Exclusion",
            "SV": "Subject Visits",
            "DA": "Drug Accountability"
        }
        
        return domain_labels.get(domain, domain)


def create_sdtm_retriever(kb_dir: str = "sdtm_rag_kb") -> SDTMRAGRetriever:
    """Create SDTM retriever with default settings"""
    
    return SDTMRAGRetriever(
        kb_dir=Path(kb_dir),
        encoder_model="all-MiniLM-L6-v2",
        use_gpu=torch.cuda.is_available()
    )


def main():
    """Test the retriever"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Test SDTM RAG Retriever")
    parser.add_argument("--kb-dir", default="sdtm_rag_kb", help="Knowledge base directory")
    parser.add_argument("--query", help="Test query")
    parser.add_argument("--domain", help="Filter by domain")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create retriever
    retriever = create_sdtm_retriever(args.kb_dir)
    
    # Interactive mode if no query provided
    if not args.query:
        print("SDTM RAG Retriever - Interactive Mode")
        print("Enter queries (or 'quit' to exit):")
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
                
            results = retriever.retrieve(query, domain_filter=args.domain)
            
            print(f"\nTop results for '{query}':")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.full_name} (Score: {result.score:.3f})")
                print(f"   Type: {result.chunk_type}")
                print(f"   Content: {result.content[:200]}...")
                
    else:
        # Single query mode
        results = retriever.retrieve(args.query, domain_filter=args.domain)
        
        print(f"Results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.full_name}")
            print(f"   Score: {result.score:.3f} (BM25: {result.bm25_score:.3f}, Vector: {result.vector_score:.3f})")
            print(f"   Type: {result.chunk_type}")
            print(f"   Content: {result.content}")


if __name__ == "__main__":
    main()