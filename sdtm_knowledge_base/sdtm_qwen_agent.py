#!/usr/bin/env python3
"""
SDTM Qwen Agent with Automatic RAG Retrieval
Integrates SDTM knowledge base with Qwen for intelligent CRF annotation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import torch
from dataclasses import dataclass

from sdtm_rag_retriever import SDTMRAGRetriever, create_sdtm_retriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass 
class AnnotationDecision:
    """Structured annotation decision"""
    variable: str
    domain: str
    confidence: float
    reasoning: str
    evidence: List[Dict[str, Any]]
    alternatives: List[Tuple[str, float]]


class SDTMQwenAgent:
    """
    SDTM-aware agent that combines Qwen LLM with RAG retrieval
    """
    
    def __init__(
        self,
        qwen_client,  # QwenClient instance
        retriever: Optional[SDTMRAGRetriever] = None,
        kb_dir: str = "sdtm_rag_kb",
        auto_retrieve: bool = True
    ):
        self.llm = qwen_client
        self.retriever = retriever or create_sdtm_retriever(kb_dir)
        self.auto_retrieve = auto_retrieve
        
        # System prompts optimized for SDTM
        self.system_prompts = {
            "annotator": """You are an expert in CDISC SDTM standards for clinical trials. Your task is to map CRF questions to the correct SDTM variables based on the retrieved knowledge and annotation instructions.

Key principles:
1. Always use the exact DOMAIN.VARIABLE format (e.g., VS.VSORRES, AE.AETERM)
2. For Findings domains, use "when" patterns: VSORRES/VSORRESU when VSTESTCD=TEMP
3. Consider supplemental qualifiers for "Other, specify" fields
4. Match based on semantic meaning, not just keywords
5. Consider the CRF form type and context
6. Follow SDTM-MSG annotation guidelines for origins, RELREC, and not-submitted fields""",

            "validator": """You are a SDTM compliance validator. Review the proposed mapping and verify:
1. The domain is appropriate for the data type
2. The variable exists in that domain
3. The mapping follows SDTM conventions
4. Required companion variables are identified"""
        }
        
    def annotate_crf_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        retrieve_k: int = 5
    ) -> AnnotationDecision:
        """
        Annotate a CRF question with SDTM variable
        
        Args:
            question: CRF question text
            context: Additional context (form name, section, etc.)
            retrieve_k: Number of candidates to retrieve
            
        Returns:
            Structured annotation decision
        """
        
        # Step 1: Retrieve relevant SDTM knowledge
        if self.auto_retrieve:
            retrieval_results = self._retrieve_knowledge(question, context, retrieve_k)
        else:
            retrieval_results = []
            
        # Step 2: Build prompt with retrieved knowledge
        prompt = self._build_annotation_prompt(question, context, retrieval_results)
        
        # Step 3: Get LLM decision
        response = self.llm.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.1,
            system_prompt=self.system_prompts["annotator"]
        )
        
        # Step 4: Parse and validate decision
        decision = self._parse_annotation_response(response, retrieval_results)
        
        # Step 5: Optional validation pass
        if decision.confidence < 0.7:
            decision = self._validate_decision(decision, question, context)
            
        return decision
        
    def _retrieve_knowledge(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Retrieve relevant SDTM knowledge"""
        
        # Build enhanced query with context
        query = question
        if context:
            if 'form_name' in context:
                query = f"{context['form_name']}: {query}"
            if 'section' in context:
                query = f"{context['section']} - {query}"
                
        # Apply domain filter if context suggests it
        domain_filter = None
        if context and 'suggested_domain' in context:
            domain_filter = context['suggested_domain']
            
        # Retrieve
        results = self.retriever.retrieve(
            query,
            domain_filter=domain_filter,
            rerank_top_k=top_k
        )
        
        return results
        
    def _build_annotation_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """Build detailed prompt with retrieved knowledge"""
        
        prompt = f"Map this CRF question to the appropriate SDTM variable:\n\n"
        prompt += f"CRF Question: \"{question}\"\n"
        
        if context:
            prompt += "\nContext:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
                
        prompt += "\nRetrieved SDTM Knowledge:\n"
        
        # Separate instruction chunks from variable chunks
        instruction_chunks = []
        variable_chunks = []
        
        for result in retrieval_results:
            if result.chunk_type in ['annotation_instruction', 'annotation_concept', 'annotation_example']:
                instruction_chunks.append(result)
            else:
                variable_chunks.append(result)
                
        # Show instructions first if any
        if instruction_chunks:
            prompt += "\nRelevant Annotation Guidelines:\n"
            for inst in instruction_chunks[:3]:  # Limit to top 3 instructions
                prompt += f"- {inst.content}\n"
                
        # Then show variable candidates
        prompt += "\nVariable Candidates:\n"
        for i, result in enumerate(variable_chunks, 1):
            prompt += f"\n{i}. {result.full_name} (Relevance: {result.score:.2f})\n"
            prompt += f"   {result.content}\n"
            
            # Add patterns if available
            if 'pattern' in result.metadata:
                pattern = result.metadata['pattern']
                prompt += f"   Pattern: When {pattern.get('when', 'N/A')}, "
                prompt += f"then {pattern.get('then', 'N/A')}\n"
                
            # Add annotation rules if available
            if hasattr(result, 'annotation_rules') and result.annotation_rules:
                rules = result.annotation_rules
                
                # Add examples
                if rules.get('examples'):
                    prompt += f"   Annotation Examples: {', '.join(rules['examples'][:2])}\n"
                    
                # Add specific patterns
                if 'findings' in rules.get('patterns', {}):
                    findings = rules['patterns']['findings']
                    prompt += f"   Findings Pattern: {findings['instruction']}\n"
                    
                # Add key rules
                if rules.get('specific_rules'):
                    key_rule = rules['specific_rules'][0] if rules['specific_rules'] else None
                    if key_rule:
                        prompt += f"   Key Rule: {key_rule['title']}\n"
                
        prompt += "\nBased on the above knowledge, select the most appropriate SDTM variable.\n"
        prompt += "Provide your answer in this format:\n"
        prompt += "VARIABLE: [DOMAIN.VARIABLE]\n"
        prompt += "CONFIDENCE: [0.0-1.0]\n"
        prompt += "REASONING: [Brief explanation of why this mapping is correct]\n"
        
        return prompt
        
    def _parse_annotation_response(
        self,
        response: str,
        retrieval_results: List[RetrievalResult]
    ) -> AnnotationDecision:
        """Parse LLM response into structured decision"""
        
        # Default values
        variable = None
        domain = None
        confidence = 0.0
        reasoning = ""
        
        # Parse structured response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("VARIABLE:"):
                var_str = line.replace("VARIABLE:", "").strip()
                if '.' in var_str:
                    parts = var_str.split('.', 1)
                    domain = parts[0]
                    variable = var_str
                    
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_str)
                except:
                    confidence = 0.5
                    
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                
        # Fallback to top retrieval result if parsing fails
        if not variable and retrieval_results:
            top_result = retrieval_results[0]
            variable = top_result.full_name
            domain = top_result.domain
            confidence = min(top_result.score, 0.8)
            reasoning = "Selected based on highest retrieval score"
            
        # Build evidence from retrieval results
        evidence = []
        for result in retrieval_results[:3]:
            evidence.append({
                "variable": result.full_name,
                "score": result.score,
                "type": result.chunk_type,
                "content": result.content[:200]
            })
            
        # Get alternatives
        alternatives = [
            (r.full_name, r.score) 
            for r in retrieval_results[1:4]
        ]
        
        return AnnotationDecision(
            variable=variable or "UNKNOWN",
            domain=domain or "UNKNOWN",
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            alternatives=alternatives
        )
        
    def _validate_decision(
        self,
        decision: AnnotationDecision,
        question: str,
        context: Optional[Dict[str, Any]]
    ) -> AnnotationDecision:
        """Validate and potentially revise a low-confidence decision"""
        
        # Build validation prompt
        prompt = f"Validate this SDTM mapping decision:\n\n"
        prompt += f"Question: \"{question}\"\n"
        prompt += f"Proposed Variable: {decision.variable}\n"
        prompt += f"Confidence: {decision.confidence}\n"
        prompt += f"Reasoning: {decision.reasoning}\n\n"
        
        prompt += "Alternatives considered:\n"
        for var, score in decision.alternatives:
            prompt += f"- {var} (score: {score:.2f})\n"
            
        prompt += "\nIs this the best mapping? If not, which alternative is better and why?"
        prompt += "\nRespond with CONFIRM or REVISE:[variable] and explanation."
        
        # Get validation response
        response = self.llm.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.1,
            system_prompt=self.system_prompts["validator"]
        )
        
        # Check if revision needed
        if "REVISE:" in response:
            # Extract revised variable
            parts = response.split("REVISE:", 1)
            if len(parts) > 1:
                revised_var = parts[1].split()[0].strip()
                if '.' in revised_var:
                    decision.variable = revised_var
                    decision.domain = revised_var.split('.')[0]
                    decision.confidence = min(decision.confidence + 0.1, 0.8)
                    decision.reasoning = f"Revised after validation: {response}"
                    
        elif "CONFIRM" in response:
            decision.confidence = min(decision.confidence + 0.1, 0.9)
            
        return decision
        
    def batch_annotate(
        self,
        questions: List[Dict[str, Any]],
        batch_size: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[AnnotationDecision]:
        """
        Annotate multiple CRF questions
        
        Args:
            questions: List of dicts with 'text' and optional 'context'
            batch_size: Process in batches to manage memory
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of annotation decisions
        """
        
        results = []
        total = len(questions)
        
        for i in range(0, total, batch_size):
            batch = questions[i:i + batch_size]
            
            for j, q in enumerate(batch):
                try:
                    decision = self.annotate_crf_question(
                        q['text'],
                        q.get('context')
                    )
                    results.append(decision)
                    
                    if progress_callback:
                        progress_callback(i + j + 1, total)
                        
                except Exception as e:
                    logger.error(f"Error annotating question: {e}")
                    # Add error decision
                    results.append(AnnotationDecision(
                        variable="ERROR",
                        domain="ERROR", 
                        confidence=0.0,
                        reasoning=str(e),
                        evidence=[],
                        alternatives=[]
                    ))
                    
            # Clear GPU cache between batches
            if hasattr(self.llm, 'model'):
                torch.cuda.empty_cache()
                
        return results
        
    def explain_mapping(
        self,
        question: str,
        variable: str
    ) -> str:
        """Generate detailed explanation for a mapping"""
        
        # Get variable context
        context = self.retriever.get_variable_context(variable)
        
        prompt = f"Explain why the CRF question \"{question}\" "
        prompt += f"maps to SDTM variable {variable}.\n\n"
        
        prompt += f"Variable Definition:\n{context.get('definition', 'N/A')}\n\n"
        
        if context.get('synonyms'):
            prompt += f"Common CRF terms: {', '.join(context['synonyms'])}\n\n"
            
        if context.get('patterns'):
            prompt += "Usage patterns:\n"
            for pattern in context['patterns']:
                prompt += f"- When {pattern.get('when')}, then {pattern.get('then')}\n"
                
        prompt += "\nProvide a clear explanation for clinical data managers."
        
        explanation = self.llm.generate(
            prompt,
            max_new_tokens=150,
            temperature=0.3,
            system_prompt="You are a SDTM training expert explaining mappings to clinical data managers."
        )
        
        return explanation


def create_sdtm_agent(qwen_client, kb_dir: str = "sdtm_rag_kb") -> SDTMQwenAgent:
    """Create SDTM agent with default configuration"""
    
    return SDTMQwenAgent(
        qwen_client=qwen_client,
        kb_dir=kb_dir,
        auto_retrieve=True
    )


def main():
    """Test the SDTM agent"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Test SDTM Qwen Agent")
    parser.add_argument("--kb-dir", default="sdtm_rag_kb", help="Knowledge base directory")
    parser.add_argument("--model-path", help="Path to Qwen model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Import and create Qwen client
    import sys
    sys.path.append('..')
    from crf_annotator.src.annotation.qwen_client import create_qwen_client
    
    logger.info("Loading Qwen model...")
    qwen_client = create_qwen_client(use_4bit=True)
    
    # Create agent
    logger.info("Creating SDTM agent...")
    agent = create_sdtm_agent(qwen_client, args.kb_dir)
    
    # Test questions
    test_questions = [
        {
            "text": "What is the subject's date of birth?",
            "context": {"form_name": "Demographics"}
        },
        {
            "text": "Temperature (°C)",
            "context": {"form_name": "Vital Signs", "suggested_domain": "VS"}
        },
        {
            "text": "Did the patient experience any serious adverse events?",
            "context": {"form_name": "Adverse Events"}
        },
        {
            "text": "Concomitant medication name",
            "context": {"form_name": "Concomitant Medications"}
        }
    ]
    
    print("\nTesting SDTM Agent on sample questions:\n")
    
    for q in test_questions:
        print(f"Question: {q['text']}")
        if q.get('context'):
            print(f"Context: {q['context']}")
            
        decision = agent.annotate_crf_question(q['text'], q.get('context'))
        
        print(f"Variable: {decision.variable}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Alternatives: {[f'{v} ({s:.2f})' for v, s in decision.alternatives]}")
        print("-" * 50)


if __name__ == "__main__":
    main()