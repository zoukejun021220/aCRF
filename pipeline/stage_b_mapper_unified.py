#!/usr/bin/env python3
"""
Unified Stage B Mapper - Fully KB-Driven SDTM Annotation

This mapper follows the hierarchical SDTM annotation process using only KB data:
1. Class Selection - Uses KB class definitions and decision hierarchy
2. Domain Selection - Selects from KB domains within the chosen class
3. Pattern Selection - Determines HOW to map (direct, conditional, supplemental, etc.)
4. Variable Selection - Maps to specific SDTM variables from KB
5. Controlled Terminology - Applies CT from KB where applicable

All decision logic uses the LLM with system prompts built from KB definitions.
NO hardcoded domain knowledge - everything comes from the knowledge base:

The knowledge base (KB) provides:
- Class definitions and decision hierarchy (class_definitions.json)
- Domain hierarchy and descriptions (domains_by_class.json)
- Pattern definitions and selection criteria (pattern_definitions.json)
- Variable definitions and roles (proto_define.json)
- Controlled terminology codelists (cdisc_all_controlled_terms.json)
- Test codes for findings domains (embedded in CT)

This ensures the mapper can be updated by modifying KB files without code changes.
"""

import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from .proto_define_validator import ProtoDefineValidator
    from .enhanced_validator import EnhancedSDTMValidator
    from .rule_based_filter import RuleBasedFilter
except ImportError:
    from proto_define_validator import ProtoDefineValidator
    from enhanced_validator import EnhancedSDTMValidator
    from rule_based_filter import RuleBasedFilter

# Structured annotation classes
@dataclass
class SDTMMapping:
    """Single SDTM variable mapping"""
    variable: str
    value: Optional[str] = None
    condition: Optional[str] = None
    target: str = "input"  # input, question, option, section
    
@dataclass  
class VLMEntry:
    """Value-level metadata entry for findings"""
    domain: str
    testcd: str
    testcd_value: str
    variables: List[str]
    
@dataclass
class SUPPEntry:
    """Supplemental qualifier entry"""
    qnam: str
    qlabel: str
    domain: str
    
@dataclass
class RelationshipEntry:
    """Cross-domain relationship"""
    from_var: str
    to_var: str
    type: str = "RELREC"

@dataclass
class OptionAnnotation:
    """Annotation for a specific option value"""
    option_value: str
    annotation: str
    mappings: List[SDTMMapping]
    
@dataclass
class AnnotationResult:
    """Complete structured annotation result"""
    pattern: str
    domain: Optional[str]
    mappings: List[SDTMMapping]
    vlm: Optional[VLMEntry] = None
    supp: Optional[SUPPEntry] = None  
    relationships: List[RelationshipEntry] = None
    clarifying: List[str] = None
    confidence: float = 0.0
    validation_status: str = "pending"
    validation_message: str = ""
    # New fields for separate option annotations
    question_annotation: Optional[str] = None
    option_annotations: Optional[List[OptionAnnotation]] = None
    requires_separate_annotations: bool = False
    
    def to_legacy_format(self) -> str:
        """Convert to legacy single-string format for backward compatibility"""
        if not self.mappings:
            return "[NOT SUBMITTED]"
        
        # Handle findings pattern
        if self.pattern == "findings" and self.vlm:
            parts = []
            for var in self.vlm.variables:
                parts.append(var)
            annotation = " / ".join(parts)
            annotation += f" when {self.vlm.domain}TESTCD = {self.vlm.testcd_value}"
            return annotation
        
        # Handle supplemental
        if self.pattern == "supplemental" and self.supp:
            return f"{self.supp.qnam} in SUPP{self.supp.domain}"
        
        # Handle plain/conditional
        parts = []
        for mapping in self.mappings:
            if mapping.condition:
                parts.append(f"{mapping.variable} when {mapping.condition}")
            elif mapping.value:
                parts.append(f"{mapping.variable} = {mapping.value}")
            else:
                parts.append(mapping.variable)
        
        return " / ".join(parts) if parts else "[NOT SUBMITTED]"

# Enable debug logging if DEBUG env var is set
debug_mode = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedSDTMMapper:
    """Simplified hierarchical SDTM mapper using KB uniformly"""
    
    def __init__(self, 
                 proto_define_path: str = None,
                 kb_path: str = None,
                 model_name: str = "Qwen/Qwen2.5-14B-Instruct", 
                 use_4bit: bool = True,  # Default to True for efficient memory usage
                 use_rag: bool = True,
                 rag_top_k: int = 5,
                 confidence_threshold: float = 0.7,
                 debug_mode: bool = False,
                 use_all_patterns: bool = True):
        """
        Initialize the unified SDTM mapper
        
        Args:
            proto_define_path: Path to proto_define.json
            kb_path: Path to KB directory
            model_name: LLM model to use
            use_4bit: Whether to use 4-bit quantization
            use_rag: Whether to use RAG for domain selection
            rag_top_k: Number of RAG results to consider
            confidence_threshold: Minimum confidence for accepting mappings
        """
        
        self.model_name = model_name
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.use_all_patterns = use_all_patterns
        
        # Initialize debug data structure
        self.debug_conversations = [] if debug_mode else None
        
        # Initialize components
        logger.info("Initializing Unified SDTM Mapper...")
        
        # Load proto_define
        if proto_define_path:
            proto_define_path = Path(proto_define_path)
        else:
            # Try multiple locations
            possible_paths = [
                Path(__file__).parent.parent / "kb" / "sdtmig_v3_4_complete" / "proto_define.json",
                Path(__file__).parent / "kb" / "proto_define.json",
                Path(__file__).parent / "proto_define.json"
            ]
            for path in possible_paths:
                if path.exists():
                    proto_define_path = path
                    break
            else:
                proto_define_path = possible_paths[0]  # Default to first option
            
        if proto_define_path.exists():
            logger.info(f"Loading proto_define from: {proto_define_path}")
            with open(proto_define_path, 'r') as f:
                self.proto_define = json.load(f)
        else:
            logger.error(f"proto_define.json not found at: {proto_define_path}")
            self.proto_define = {"datasets": {}}
        
        # Set KB path
        if kb_path:
            self.kb_path = Path(kb_path)
        else:
            # Try default KB locations
            possible_kb_paths = [
                Path(__file__).parent.parent / "kb" / "sdtmig_v3_4_complete",
                Path(__file__).parent.parent / "kb" / "cdisc_integrated",
                Path(__file__).parent / "kb"
            ]
            for kb_dir in possible_kb_paths:
                if kb_dir.exists():
                    self.kb_path = kb_dir
                    break
            else:
                self.kb_path = possible_kb_paths[0]
        
        # Load KB resources
        if self.kb_path and self.kb_path.exists():
            logger.info(f"Loading KB from: {self.kb_path}")
            self._load_kb_resources(self.kb_path)
        else:
            logger.warning("No KB found, using empty collections")
            self.domains_by_class = {}
            self.common_ct = {}
            self._ct_data = {}
            self.class_definitions = {}
            self.class_hierarchy = {}
        
        # Load variables from variables_all.json
        self._load_all_variables()
        
        # Load controlled terminology
        self._load_controlled_terminology()
        
        # Initialize validators
        try:
            self.proto_validator = ProtoDefineValidator(self.proto_define)
            self.enhanced_validator = EnhancedSDTMValidator(kb_path)
            self.rule_filter = RuleBasedFilter()
        except Exception as e:
            logger.warning(f"Could not initialize validators: {e}")
            self.enhanced_validator = None
            self.rule_filter = None
        
        # Load QRS instruments if available
        qrs_path = Path(__file__).parent.parent / "kb" / "qrs_instruments.json"
        if qrs_path.exists():
            with open(qrs_path, 'r') as f:
                self.qrs_instruments = json.load(f)
        else:
            self.qrs_instruments = {}
        
        # Initialize LLM
        self._init_llm(use_4bit)
        
        # Initialize RAG if enabled
        if self.use_rag:
            self._init_rag()
            
        # Load pattern definitions from KB
        self.patterns = self._load_pattern_definitions(kb_path)
        
        # Caches
        self.field_result_cache = {}
        self.pattern_learning_cache = {}
        
        logger.info("Unified SDTM Mapper initialized successfully")
    
    def _load_controlled_terminology(self):
        """Load controlled terminology from cdisc_ct.json"""
        self.ct_by_variable = {}
        self.cdisc_ct = None
        
        if hasattr(self, 'kb_path') and self.kb_path:
            ct_path = self.kb_path / "cdisc_ct.json"
            if ct_path.exists():
                logger.info(f"Loading controlled terminology from: {ct_path}")
                with open(ct_path, 'r') as f:
                    ct_data = json.load(f)
                
                # Store full CT data for guided selection
                self.cdisc_ct = ct_data
                
                # Process codelists
                for codelist_info in ct_data.get('codelists', []):
                    codelist = codelist_info.get('codelist', {})
                    short_name = codelist.get('shortName', '')
                    terms = codelist_info.get('terms', [])
                    
                    # Try to extract variable name from shortName
                    # Common patterns: "Action Taken with Study Treatment", "Adverse Event Severity/Intensity"
                    if terms:
                        # Store by codelist short name
                        self.ct_by_variable[short_name] = {
                            'definition': codelist.get('definition', ''),
                            'extensible': codelist.get('extensible', 'true'),
                            'terms': [
                                {
                                    'code': term.get('submissionValue', ''),
                                    'value': term.get('preferredTerm', ''),
                                    'definition': term.get('definition', '')
                                }
                                for term in terms
                            ]
                        }
                
                logger.info(f"Loaded {len(self.ct_by_variable)} controlled terminology codelists")
            else:
                logger.warning(f"cdisc_ct.json not found at: {ct_path}")
        
        # Also check proto_define for CT references
        if hasattr(self, 'proto_define') and 'datasets' in self.proto_define:
            for domain_code, domain_data in self.proto_define['datasets'].items():
                for var_name, var_data in domain_data.get('variables', {}).items():
                    if 'codelist_ref' in var_data:
                        codelist_ref = var_data['codelist_ref']
                        # Store the reference for later lookup
                        if var_name not in self.ct_by_variable:
                            self.ct_by_variable[var_name] = {
                                'reference': codelist_ref,
                                'domain': domain_code
                            }
    
    def _load_all_variables(self):
        """Load all variables from variables_all.json"""
        if hasattr(self, 'kb_path') and self.kb_path:
            variables_path = self.kb_path / "variables_all.json"
            if variables_path.exists():
                logger.info(f"Loading variables from: {variables_path}")
                with open(variables_path, 'r') as f:
                    all_variables = json.load(f)
                
                # Store full list for guided selection
                self.variables_all = all_variables
                
                # Organize variables by domain
                self.domain_variables = {}
                for var in all_variables:
                    domain = var.get('domain')
                    if domain:
                        if domain not in self.domain_variables:
                            self.domain_variables[domain] = {}
                        var_name = var.get('name')
                        if var_name:
                            self.domain_variables[domain][var_name] = {
                                'label': var.get('label', ''),
                                'type': var.get('type', ''),
                                'role': var.get('role', ''),
                                'core': var.get('core', ''),
                                'codelist': var.get('codelist', ''),
                                'description': var.get('description', '')
                            }
                
                logger.info(f"Loaded variables for {len(self.domain_variables)} domains")
            else:
                logger.warning(f"variables_all.json not found at: {variables_path}")
                self.domain_variables = {}
                self.variables_all = []
        else:
            self.domain_variables = {}
            self.variables_all = []
    
    def _load_kb_resources(self, kb_path: Path):
        """Load KB resources"""
        kb_base = kb_path
        
        # Load domains by class
        domains_by_class_path = kb_base / "domains_by_class.json"
        if domains_by_class_path.exists():
            with open(domains_by_class_path, 'r') as f:
                self.domains_by_class = json.load(f)
            logger.info(f"Loaded {len(self.domains_by_class)} domain classes")
        else:
            self.domains_by_class = {}
            
        # Load common CT terms
        ct_path = kb_base / "common_ct_terms_complete.json"
        if ct_path.exists():
            with open(ct_path, 'r') as f:
                self.common_ct = json.load(f)
        else:
            # Try alternative path
            ct_path = kb_base / "common_ct_terms.json"
            if ct_path.exists():
                with open(ct_path, 'r') as f:
                    self.common_ct = json.load(f)
            else:
                self.common_ct = {}
            
        # Load all CT data
        all_ct_path = kb_base / "cdisc_all_controlled_terms.json"
        if all_ct_path.exists():
            with open(all_ct_path, 'r') as f:
                self._ct_data = json.load(f)
        else:
            self._ct_data = {}
            
        # Load class definitions
        class_def_path = kb_base / "class_definitions.json"
        if class_def_path.exists():
            with open(class_def_path, 'r') as f:
                class_data = json.load(f)
                self.class_definitions = class_data.get('sdtm_classes', {})
                self.class_hierarchy = class_data.get('decision_hierarchy', {})
            logger.info(f"Loaded {len(self.class_definitions)} class definitions")
        else:
            self.class_definitions = {}
            self.class_hierarchy = {}
    
    def _load_pattern_definitions(self, kb_path: Path = None) -> Dict[str, Any]:
        """Load pattern definitions from KB"""
        
        # Try to load from KB
        if kb_path:
            pattern_path = kb_path / "pattern_definitions.json"
            if pattern_path.exists():
                logger.info(f"Loading pattern definitions from: {pattern_path}")
                with open(pattern_path, 'r') as f:
                    pattern_data = json.load(f)
                    return pattern_data.get('annotation_patterns', {})
        
        # Try default KB locations
        possible_paths = [
            Path(__file__).parent.parent / "kb" / "sdtmig_v3_4_complete" / "pattern_definitions.json",
            Path(__file__).parent.parent / "kb" / "cdisc_integrated" / "pattern_definitions.json",
            Path(__file__).parent / "kb" / "pattern_definitions.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading pattern definitions from: {path}")
                with open(path, 'r') as f:
                    pattern_data = json.load(f)
                    return pattern_data.get('annotation_patterns', {})
        
        # If no KB pattern file found, load patterns as empty
        logger.warning("No pattern definitions found in KB, using minimal defaults")
        return {
            "direct": {
                "description": "Direct mapping to SDTM variable",
                "formula": "<SDTM.Variable>",
                "keywords": []
            },
            "not_submitted": {
                "description": "Not submitted to SDTM",
                "formula": "[NOT SUBMITTED]",
                "keywords": []
            }
        }
    
    def _init_llm(self, use_4bit: bool):
        """Initialize the LLM model"""
        logger.info(f"Loading LLM model: {self.model_name}")
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Ensure pad token is set for LLaMA models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("LLM model loaded successfully")
    
    def _init_rag(self):
        """Initialize RAG components"""
        try:
            logger.info("Initializing RAG components...")
            self.rag_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build domain embeddings
            self.domain_embeddings = {}
            self.domain_texts = {}
            
            for domain_code, domain_info in self.proto_define.get("datasets", {}).items():
                # Create searchable text
                text_parts = [
                    f"Domain: {domain_code}",
                    f"Name: {domain_info.get('label', '')}",
                    f"Description: {domain_info.get('description', '')}",
                    f"Class: {domain_info.get('class', '')}",
                    f"Structure: {domain_info.get('structure', '')}"
                ]
                
                # Add key variables
                variables = domain_info.get("variables", {})
                var_names = list(variables.keys())[:10]
                if var_names:
                    text_parts.append(f"Variables: {', '.join(var_names)}")
                
                domain_text = " ".join(text_parts)
                self.domain_texts[domain_code] = domain_text
                
                # Create embedding
                embedding = self.rag_encoder.encode(domain_text)
                self.domain_embeddings[domain_code] = embedding
            
            logger.info(f"RAG initialized with {len(self.domain_embeddings)} domain embeddings")
            
        except Exception as e:
            logger.warning(f"Could not initialize RAG: {e}")
            self.use_rag = False
    
    def annotate_field_enhanced(self, field_data: Dict[str, Any]) -> AnnotationResult:
        """Enhanced annotation supporting structured results"""
        try:
            # Convert from main annotate_field method
            result = self.annotate_field(field_data)
            
            # Convert to structured format
            return self._convert_to_structured_result(result)
            
        except Exception as e:
            logger.error(f"Error in enhanced annotation: {str(e)}")
            return AnnotationResult(
                pattern="not_submitted",
                domain=None,
                mappings=[SDTMMapping(variable="[NOT SUBMITTED]")],
                confidence=0.0,
                validation_status="error",
                validation_message=str(e)
            )
    
    def annotate_field(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate field using either all-patterns or step-by-step approach
        """
        if self.use_all_patterns:
            return self._annotate_field_all_patterns(field_data)
        else:
            return self.annotate_field_original(field_data)
    
    def _annotate_field_all_patterns(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        All-patterns annotation using single LLM call
        Shows all patterns and selects best match
        """
        try:
            # Start new debug conversation group if in debug mode
            if self.debug_mode and self.debug_conversations is not None:
                debug_field_start = len(self.debug_conversations)
            
            # Check cache first
            cache_key = self._get_cache_key(field_data)
            if cache_key in self.field_result_cache:
                logger.info(f"Using cached mapping for: {field_data.get('label', '')}")
                return self.field_result_cache[cache_key]
            
            # Check for operational fields
            if self._is_operational_field(field_data):
                result = {
                    "annotation": "[NOT SUBMITTED]",
                    "domain": None,
                    "pattern": "not_submitted",
                    "sdtm_class": None,
                    "valid": True,
                    "validation_message": "Operational field",
                    "confidence": 0.99
                }
                self.field_result_cache[cache_key] = result
                return result
            
            # Single-step all-patterns annotation
            logger.debug(f"\nAll-patterns annotation for '{field_data.get('label', '')}'")
            result = self._annotate_all_patterns(field_data)
            
            # Validate using proto_define
            if result['domain'] and result['annotation'] != "[NOT SUBMITTED]":
                result['valid'], result['validation_message'] = self._validate_with_proto_define(
                    result['annotation'], result['domain'], result['pattern']
                )
            
            # Cache result
            self.field_result_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error annotating field '{field_data.get('label', '')}': {str(e)}")
            return {
                "annotation": "[NOT SUBMITTED]",
                "domain": None,
                "pattern": "not_submitted",
                "valid": False,
                "validation_message": f"Error: {str(e)}",
                "confidence": 0.0
            }
    
    def _annotate_all_patterns(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Three-step annotation process: domain → pattern → variable/CT"""
        
        # Step 1: Domain Selection (show ALL domains)
        domain, domain_confidence = self._select_domain_only(field_data)
        
        if not domain:
            return {
                "annotation": "[NOT SUBMITTED]",
                "domain": None,
                "pattern": "not_submitted",
                "sdtm_class": None,
                "valid": False,
                "validation_message": "Could not determine domain",
                "confidence": 0.0
            }
        
        # Step 2: Pattern Selection (based on selected domain)
        pattern, pattern_confidence = self._select_pattern_for_domain(field_data, domain)
        
        if pattern == 'not_submitted':
            return {
                "annotation": "[NOT SUBMITTED]",
                "domain": domain,
                "pattern": "not_submitted",
                "sdtm_class": None,
                "valid": True,
                "validation_message": "Field not submitted to SDTM",
                "confidence": 0.99
            }
        
        # Check if separate annotations are needed (especially for IE domain with Yes/No)
        if self._check_requires_separate_annotations(field_data, domain) or \
           (domain == 'IE' and field_data.get('options') and len(field_data.get('options', [])) == 2):
            # Build separate annotations for question and options
            domain_vars = self._get_domain_variables(domain)
            separate_result = self._build_separate_annotations(field_data, domain, domain_vars)
            
            # Determine SDTM class from domain
            sdtm_class = None
            for class_name, domains in self.domains_by_class.items():
                if any(d.get('code') == domain for d in domains):
                    sdtm_class = class_name
                    break
            
            # Combine results with separate annotations
            result = {
                "annotation": separate_result['combined_annotation'],
                "domain": domain,
                "pattern": pattern,
                "sdtm_class": sdtm_class,
                "valid": True,
                "validation_message": "",
                "confidence": pattern_confidence,
                "requires_separate_annotations": True,
                "question_annotation": separate_result['question_annotation'],
                "option_annotations": separate_result['option_annotations']
            }
        else:
            # Step 3: Variable/CT Selection (based on pattern and domain)
            annotation, var_confidence = self._select_variables_and_ct(field_data, domain, pattern)
            
            # Determine SDTM class from domain
            sdtm_class = None
            for class_name, domains in self.domains_by_class.items():
                if any(d.get('code') == domain for d in domains):
                    sdtm_class = class_name
                    break
            
            # Combine results
            result = {
                "annotation": annotation,
                "domain": domain,
                "pattern": pattern,
                "sdtm_class": sdtm_class,
                "valid": True,
                "validation_message": "",
                "confidence": (domain_confidence + pattern_confidence + var_confidence) / 3,
                "requires_separate_annotations": False
            }
        
        return result
    
    def _select_domain_only(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Step 1: Select domain by showing ALL available domains"""
        
        # Get ALL domain descriptions
        all_domains = self._get_all_domain_descriptions()
        
        # Build system prompt
        system_prompt = """You are an SDTM domain selection expert. Select the most appropriate SDTM domain for the given CRF field.

IMPORTANT: You must select from the available domains listed below. Consider:
- The field content and medical context
- The form name and section (especially for Inclusion/Exclusion criteria → IE domain)
- Whether it's demographic, event, intervention, or finding data

Return ONLY the domain code (e.g., DM, AE, VS, IE, etc.)."""
        
        # Build user prompt
        user_prompt = f"""Select the SDTM domain for this CRF field:

Question Text: {field_data.get('label', '')}
Form: {field_data.get('form', '')}
Section: {field_data.get('section', '')}
Field Type: {field_data.get('input_type', '')}
Options: {field_data.get('options', [])}

{all_domains}

Based on the field information above, which SDTM domain should this field map to?
Return only the domain code."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 1: Domain Selection for '{field_data.get('label', '')}'"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract domain
        domain = response.strip().upper()
        
        # Validate domain exists
        valid_domain = False
        for class_domains in self.domains_by_class.values():
            if any(d.get('code') == domain for d in class_domains):
                valid_domain = True
                break
        
        if valid_domain:
            return domain, 0.9
        else:
            # Try to extract just the domain code if response contains extra text
            words = domain.split()
            for word in words:
                if len(word) <= 6:  # Domain codes are typically short
                    for class_domains in self.domains_by_class.values():
                        if any(d.get('code') == word for d in class_domains):
                            return word, 0.7
            
            return None, 0.0
    
    def _select_pattern_for_domain(self, field_data: Dict[str, Any], domain: str) -> Tuple[str, float]:
        """Step 2: Select annotation pattern based on the chosen domain"""
        
        # Get pattern descriptions
        pattern_descriptions = self._get_all_patterns_description()
        
        # Build system prompt
        system_prompt = f"""You are an SDTM annotation pattern expert. The domain {domain} has been selected.
Now select the appropriate annotation pattern based on the field characteristics.

Consider:
- Field type (text, radio, checkbox, date)
- Whether it has "other, specify" text
- Whether it's a measurement/test (for findings domains)
- Whether it maps to multiple domains
- Whether it needs supplemental qualifiers

{pattern_descriptions}

Return ONLY the pattern name from the list above."""
        
        # Build user prompt
        user_prompt = f"""Select the annotation pattern for this field in domain {domain}:

Question Text: {field_data.get('label', '')}
Field Type: {field_data.get('input_type', '')}
Options: {field_data.get('options', [])}
Has Units: {field_data.get('has_units', False)}

Context:
- Contains "other specify"? {'yes' if 'other' in field_data.get('label', '').lower() and 'specify' in field_data.get('label', '').lower() else 'no'}
- Is measurement/test? {'yes' if domain in ['VS', 'LB', 'EG', 'PE', 'QS'] else 'no'}
- Is checkbox with multiple options? {'yes' if field_data.get('input_type') == 'checkbox' else 'no'}

Which pattern should be used?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 2: Pattern Selection for domain {domain}"
        response = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        
        # Parse pattern from response
        response_lower = response.strip().lower()
        
        # Map to exact pattern names
        pattern_map = {
            'not submitted': 'not_submitted',
            'direct mapping': 'direct',
            'direct': 'direct',
            'fixed value': 'checkbox_fixed',
            'checkbox': 'checkbox_fixed',
            'conditional single': 'test_measurement',
            'test measurement': 'test_measurement',
            'measurement': 'test_measurement',
            'other specify': 'other_specify',
            'supplemental': 'supplemental',
            'cross domain': 'cross_domain',
            'cross-domain': 'cross_domain'
        }
        
        for key, value in pattern_map.items():
            if key in response_lower:
                return value, 0.9
        
        # Default to direct if unclear
        return 'direct', 0.5
    
    def _select_variables_and_ct(self, field_data: Dict[str, Any], domain: str, pattern: str) -> Tuple[str, float]:
        """Step 3: Select variables and controlled terminology based on pattern"""
        
        # Get domain variables
        domain_vars = self._get_domain_variables(domain)
        
        # Route to appropriate handler based on pattern - all use new two-step approach
        if pattern == 'variable_with_ct':
            return self._build_variable_with_ct_annotation(field_data, domain, domain_vars)
        elif pattern == 'direct':
            return self._build_direct_annotation_guided(field_data, domain, domain_vars)
        elif pattern == 'checkbox_fixed':
            return self._build_checkbox_annotation_guided(field_data, domain, domain_vars)
        elif pattern == 'test_measurement':
            return self._build_test_annotation_guided(field_data, domain, domain_vars)
        elif pattern == 'other_specify':
            return self._build_other_specify_annotation_guided(field_data, domain, domain_vars)
        elif pattern == 'supplemental':
            return self._build_supplemental_annotation_guided(field_data, domain)
        elif pattern == 'cross_domain':
            return self._build_cross_domain_annotation_guided(field_data, domain, domain_vars)
        else:
            # Fallback
            return f"{domain}.UNKNOWN", 0.3
    
    def _check_requires_separate_annotations(self, field_data: Dict[str, Any], domain: str) -> bool:
        """Check if field requires separate annotations for question vs options"""
        
        # Quick checks
        if not field_data.get('options'):
            return False
            
        # Ask LLM to determine based on field characteristics
        system_prompt = """Determine if this field requires separate annotations for the question text vs each option.

Consider if:
- The question text and option values need different SDTM variables
- The question identifies WHAT is being collected while options contain the actual values
- Each option might have a different annotation

Return EXACTLY one word: YES or NO"""
        
        user_prompt = f"""Field: {field_data.get('label', '')}
Domain: {domain}
Options: {field_data.get('options', [])}
Field Type: {field_data.get('input_type', '')}

Does this field need separate annotations for question vs options?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Checking if separate annotations needed for: {field_data.get('label', '')}"
        response = self._query_llm_with_messages(messages, max_tokens=10, debug_context=debug_ctx)
        
        return response.strip().upper() == "YES"
    
    def _select_pattern_and_domain(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select pattern and domain using all-patterns prompt"""
        
        # Build comprehensive system prompt
        system_prompt = self._build_all_patterns_system_prompt()
        
        # Build user prompt with field info
        user_prompt = self._build_all_patterns_user_prompt(field_data)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Pattern and domain selection for: {field_data.get('label', '')}"
        response = self._query_llm_with_messages(messages, max_tokens=250, debug_context=debug_ctx)
        
        # Parse response
        return self._parse_all_patterns_response(response, field_data)
    
    def _build_all_patterns_system_prompt(self) -> str:
        """Build system prompt showing ALL annotation patterns"""
        
        return """You are an SDTM annotation expert. Analyze the CRF field and select the appropriate annotation pattern.

ALL AVAILABLE ANNOTATION PATTERNS:

1. **[NOT SUBMITTED]**
   - Use for: Administrative/operational fields (signatures, page numbers, data entry tracking)
   - Example: Monitor initials, investigator signature

2. **<Domain>.<Variable>** (Direct Mapping)
   - Use for: Simple one-to-one field-to-variable mapping
   - Examples: DM.SEX, AE.AETERM, CM.CMTRT
   - Format: Always use domain prefix for clarity

3. **<Variable> = <Value>** (Variable with CT)
   - Use for: Fields that map to variables with specific controlled terminology values
   - Examples: AESER = Y, DSTERM = COMPLETED
   - The value must be from controlled terminology

4. **<Var1> / <Var2> = <Value>** (Multiple Variables, Same Value)
   - Use for: One field populating both term and decode
   - Example: DS.DSTERM / DS.DSDECOD = COMPLETED
   - Common when term equals its standardized version

5. **<Variable> in SUPP<Domain>** (Supplemental Single)
   - Use for: Non-standard data, "Other, specify" fields
   - Example: RACEOTH in SUPPDM
   - QNAM is 8-char uppercase identifier

6. **<Var1> / <Var2> in SUPP<Domain>** (Supplemental Multiple)
   - Use for: Related non-standard values from one question
   - Example: ICDUR / ICDURU in SUPPIC (duration and unit)

7. **<Variable> when <TestVar> = <TestCode>** (Conditional Single)
   - Use for: Findings domains where result depends on test
   - Example: VSORRES when VSTESTCD = SYSBP
   - TestCode must be from domain's test code list

8. **<Var1> / <Var2> when <TestVar> = <TestCode>** (Conditional Multiple)
   - Use for: Result and unit in findings domains
   - Example: VSORRES / VSORRESU when VSTESTCD = WEIGHT

9. **<Variable> when <Var1> = <Val1> and <Var2> = <Val2>** (Composite Condition)
   - Use for: Complex conditions with multiple identifiers
   - Example: FAORRES when FATESTCD = OCCUR and FAOBJ = DEATH

10. **<Variable>OTH when <Variable> = OTHER** (Other Specify)
    - Use for: Free text when main field = "Other"
    - Example: CMROUTEOTH when CMROUTE = OTHER

11. **--<Variable> [<Domain1>.<Var>, <Domain2>.<Var>]** (Cross-Domain)
    - Use for: One field populating multiple domains
    - Example: --DTC [DM.RFSTDTC, SV.SVSTDTC] for informed consent date

12. **<Option1> → <Var1>, <Option2> → <Var2>** (Multiple Checkboxes)
    - Use for: Check-all-that-apply with separate Y/N variables
    - Example: Death? → AE.AESDTH, Hospitalization? → AE.AESHOSP

13. **<DateField> + <TimeField> → <Variable>DTC** (Combined DateTime)
    - Use for: Separate date/time fields combining to ISO datetime
    - Example: Start Date + Start Time → AE.AESTDTC

14. **<Variable> = f(<Fields>)** (Derived)
    - Use for: Calculated fields not directly collected
    - Example: DM.AGE = calculated from BRTHDTC

CRITICAL RULES:
- Choose the MOST SPECIFIC pattern that fits
- Findings domains (VS, LB, EG, etc.) typically use conditional patterns
- "Other, specify" ALWAYS uses supplemental or OTH pattern
- Multiple checkboxes → Multiple indicator variables
- Administrative fields → [NOT SUBMITTED]

Respond with exactly this format:
PATTERN: [pattern name]
DOMAIN: [domain code]
ANNOTATION: [complete annotation formula]
REASONING: [brief explanation]"""
    
    def _build_all_patterns_user_prompt(self, field_data: Dict[str, Any]) -> str:
        """Build user prompt with field context"""
        
        # Get domain suggestions based on form/section
        likely_domains = self._suggest_domains_for_field(field_data)
        
        prompt = f"""Annotate this CRF field:

Field Label: {field_data.get('label', '')}
Field Type: {field_data.get('input_type', 'text')}
Form/Section: {field_data.get('form', 'Unknown')} / {field_data.get('section', 'Unknown')}
Has Options: {bool(field_data.get('options', []))}
Options: {', '.join(map(str, field_data.get('options', []))) if field_data.get('options') else 'None'}
Has Units: {field_data.get('has_units', False)}
Units: {field_data.get('units', 'None')}

Context clues:
- Is administrative? {'yes' if self._is_operational_field(field_data) else 'no'}
- Contains "other specify"? {'yes' if 'other' in field_data.get('label', '').lower() and 'specify' in field_data.get('label', '').lower() else 'no'}
- Is measurement/test? {'yes' if any(kw in field_data.get('label', '').lower() for kw in ['pressure', 'rate', 'weight', 'height', 'temperature', 'test', 'result']) else 'no'}
- Has date/time component? {'yes' if any(kw in field_data.get('label', '').lower() for kw in ['date', 'time', 'when']) else 'no'}

Likely SDTM Domains based on form/section: {', '.join(likely_domains[:5])}

Select the appropriate pattern and provide the complete annotation."""
        
        return prompt
    
    def _suggest_domains_for_field(self, field_data: Dict[str, Any]) -> List[str]:
        """Suggest likely domains based on field context"""
        
        form = field_data.get('form', '').lower()
        section = field_data.get('section', '').lower()
        label = field_data.get('label', '').lower()
        
        suggestions = []
        
        # Form-based suggestions
        if 'demograph' in form:
            suggestions.extend(['DM', 'SC'])
        elif 'adverse' in form or 'ae' in form:
            suggestions.extend(['AE', 'FA'])
        elif 'vital' in form:
            suggestions.extend(['VS'])
        elif 'lab' in form:
            suggestions.extend(['LB'])
        elif 'medic' in form:
            if 'history' in form:
                suggestions.extend(['MH'])
            else:
                suggestions.extend(['CM', 'EX'])
        elif 'disposition' in form:
            suggestions.extend(['DS'])
        
        # Label-based suggestions
        if 'blood pressure' in label or 'pulse' in label or 'temperature' in label:
            suggestions.append('VS')
        elif 'adverse event' in label:
            suggestions.append('AE')
        elif 'medication' in label or 'drug' in label:
            suggestions.extend(['CM', 'EX'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for domain in suggestions:
            if domain not in seen:
                seen.add(domain)
                unique_suggestions.append(domain)
        
        return unique_suggestions if unique_suggestions else ['DM', 'AE', 'VS', 'LB', 'CM']
    
    def _parse_all_patterns_response(self, response: str, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the structured response from all-patterns annotation"""
        
        # Initialize result
        result = {
            "annotation": "[NOT SUBMITTED]",
            "domain": None,
            "pattern": "not_submitted",
            "sdtm_class": None,
            "valid": True,
            "validation_message": "",
            "confidence": 0.0
        }
        
        # Parse response lines
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('PATTERN:'):
                pattern_text = line.replace('PATTERN:', '').strip().lower()
                # Map to internal pattern names
                pattern_map = {
                    'not submitted': 'not_submitted',
                    'direct mapping': 'direct',
                    'direct': 'direct',
                    'fixed value': 'fixed_value',
                    'variable with ct': 'variable_with_ct',
                    'variable_with_ct': 'variable_with_ct',
                    'multiple variables': 'multiple_variables',
                    'supplemental single': 'supplemental',
                    'supplemental': 'supplemental',
                    'supplemental multiple': 'supplemental',
                    'conditional single': 'conditional_population',
                    'conditional': 'conditional_population',
                    'conditional multiple': 'conditional_population',
                    'composite condition': 'conditional_composite',
                    'other specify': 'conditional_other_specify',
                    'cross-domain': 'wildcard_cross_domain',
                    'multiple checkboxes': 'multiple_response_checkboxes',
                    'combined datetime': 'combined_datetime',
                    'derived': 'derived'
                }
                
                for key, value in pattern_map.items():
                    if key in pattern_text:
                        result['pattern'] = value
                        break
                        
            elif line.startswith('DOMAIN:'):
                domain_text = line.replace('DOMAIN:', '').strip().upper()
                # Extract just the domain code (first word)
                result['domain'] = domain_text.split()[0] if domain_text else None
                
            elif line.startswith('ANNOTATION:'):
                result['annotation'] = line.replace('ANNOTATION:', '').strip()
                
            elif line.startswith('REASONING:'):
                # Can use for confidence adjustment
                reasoning = line.replace('REASONING:', '').strip()
                if 'high confidence' in reasoning.lower() or 'clear' in reasoning.lower():
                    result['confidence'] = 0.9
                elif 'uncertain' in reasoning.lower() or 'guess' in reasoning.lower():
                    result['confidence'] = 0.6
                else:
                    result['confidence'] = 0.8
        
        # Set default confidence if not set
        if result['confidence'] == 0.0:
            result['confidence'] = 0.75
        
        # Determine SDTM class from domain
        if result['domain']:
            for class_name, domains in self.domains_by_class.items():
                if any(d.get('code') == result['domain'] for d in domains):
                    result['sdtm_class'] = class_name
                    break
        
        return result
    
    def _extract_code_from_label(self, label: str) -> Optional[str]:
        """Extract code from label text (e.g., 'INC1' from 'Age ≥ 18 years (INC1)')"""
        import re
        # Look for code in parentheses at the end
        match = re.search(r'\(([A-Z0-9]+)\)$', label)
        if match:
            return match.group(1)
        return None
    
    def _build_separate_annotations(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Dict[str, Any]:
        """Build separate annotations for question and each option"""
        
        result = {
            "question_annotation": "",
            "option_annotations": [],
            "combined_annotation": ""
        }
        
        # Step 1: Annotate the question text
        system_prompt = f"""You are annotating the QUESTION TEXT only (not the options) for domain {domain}.

The question text identifies what is being collected.
Options will be annotated separately.

Available {domain} Variables:"""
        
        # Show relevant variables
        for var in domain_vars:
            if var.get('role') == 'Topic' or 'test' in var.get('label', '').lower():
                system_prompt += f"\n- {var['name']}: {var.get('label', '')}"
        
        system_prompt += "\n\nReturn ONLY the annotation formula, nothing else.\nExample output: IETEST when IETESTCD = INC1"
        
        user_prompt = f"""Question: {field_data.get('label', '')}
Domain: {domain}

What is the annotation for the question text?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Question annotation for: {field_data.get('label', '')}"
        question_resp = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        # Clean up response - remove any prefix like "Annotation:"
        question_resp = question_resp.strip()
        if question_resp.startswith("Annotation:"):
            question_resp = question_resp.replace("Annotation:", "").strip()
        result['question_annotation'] = question_resp
        
        # Step 2: Annotate each option
        if field_data.get('options'):
            for option in field_data['options']:
                option_system_prompt = f"""You are annotating a specific OPTION VALUE for domain {domain}.

The question has already been annotated as: {result['question_annotation']}

Available {domain} Variables for options:"""
                
                # Show relevant variables for options
                for var in domain_vars:
                    if any(kw in var.get('label', '').lower() for kw in ['response', 'result', 'outcome', 'met']):
                        option_system_prompt += f"\n- {var['name']}: {var.get('label', '')}"
                
                option_system_prompt += "\n\nReturn ONLY the annotation formula, nothing else.\nExamples: [NOT SUBMITTED] or IEORRES = N or IEYN = N"
                
                option_user_prompt = f"""Option Value: {option}
Question: {field_data.get('label', '')}
Domain: {domain}

What is the annotation for this option?"""
                
                option_messages = [
                    {"role": "system", "content": option_system_prompt},
                    {"role": "user", "content": option_user_prompt}
                ]
                
                debug_ctx = f"Option '{option}' annotation"
                option_resp = self._query_llm_with_messages(option_messages, max_tokens=50, debug_context=debug_ctx)
                # Clean up response
                option_resp = option_resp.strip()
                if option_resp.startswith("Annotation:"):
                    option_resp = option_resp.replace("Annotation:", "").strip()
                
                result['option_annotations'].append({
                    "option": option,
                    "annotation": option_resp
                })
        
        # Build combined annotation for display
        combined_parts = [f"Question: {result['question_annotation']}"]
        for opt_ann in result['option_annotations']:
            combined_parts.append(f"Option '{opt_ann['option']}': {opt_ann['annotation']}")
        result['combined_annotation'] = " | ".join(combined_parts)
        
        return result
    
    def _build_variable_with_ct_annotation(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build variable with controlled terminology annotation using two-step approach"""
        
        # Step 1: Select variables - show ALL domain variables
        variables = self._select_variables_for_pattern(field_data, domain, domain_vars, 'variable_with_ct')
        
        if not variables:
            return f"{domain}.UNKNOWN", 0.3
        
        # Step 2: Select controlled terminology values for each variable
        annotations = []
        for var in variables:
            ct_value = self._select_ct_value_for_variable(field_data, domain, var)
            if ct_value:
                annotations.append(f"{var} = {ct_value}")
            else:
                annotations.append(var)
        
        # Combine annotations
        if len(annotations) == 1:
            return annotations[0], 0.9
        else:
            return " / ".join(annotations), 0.9
    
    def _select_variables_for_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict], pattern: str) -> List[str]:
        """Select variables showing ALL available variables in the domain"""
        
        # Build comprehensive variable list grouped by role
        vars_by_role = {}
        for var in domain_vars:
            role = var.get('role', 'Other')
            if role not in vars_by_role:
                vars_by_role[role] = []
            vars_by_role[role].append(var)
        
        system_prompt = f"""You are an SDTM variable selection expert for domain {domain}.
Based on the pattern '{pattern}' and field information, select the appropriate SDTM variables.

Pattern: {pattern}

Available {domain} Variables (ALL variables shown):"""
        
        # Show ALL variables organized by role
        for role in ['Identifier', 'Topic', 'Qualifier', 'Grouping Qualifier', 'Timing', 'Result Qualifier', 'Other']:
            if role in vars_by_role:
                system_prompt += f"\n\n{role} Variables:"
                for var in vars_by_role[role]:
                    var_desc = f"\n- {var['name']}: {var.get('label', '')}"
                    if var.get('codelist'):
                        var_desc += f" [Has CT: {var['codelist']}]"
                    system_prompt += var_desc
        
        system_prompt += "\n\nReturn ONLY the variable names separated by commas (e.g., IETESTCD,IECAT,IEYN)"
        
        user_prompt = f"""Select variables for this field:

Question Text: {field_data.get('label', '')}
Field Type: {field_data.get('input_type', '')}
Options: {field_data.get('options', [])}
Form: {field_data.get('form', '')}
Section: {field_data.get('section', '')}

Which variables should be used?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Variable selection for {pattern} pattern in {domain}"
        response = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        
        # Parse response
        variables = [v.strip() for v in response.strip().split(',')]
        return variables
    
    def _select_ct_value_for_variable(self, field_data: Dict[str, Any], domain: str, variable: str) -> str:
        """Select controlled terminology value showing ALL possible CT values"""
        
        # Get CT values from multiple sources
        ct_values = self._get_all_ct_values_for_variable(variable, domain)
        
        if not ct_values:
            # No CT found, return appropriate value based on field
            if field_data.get('options'):
                return field_data['options'][0].upper()
            return 'Y'
        
        system_prompt = f"""You are selecting a controlled terminology value for SDTM variable {variable}.

Variable: {variable}
Domain: {domain}

ALL Available Controlled Terminology Values:"""
        
        # Show ALL CT values
        for i, ct in enumerate(ct_values):
            ct_code = ct.get('submissionValue', ct.get('code', ''))
            ct_term = ct.get('preferredTerm', ct.get('decode', ct.get('value', '')))
            ct_def = ct.get('definition', '')
            system_prompt += f"\n{i+1}. {ct_code}: {ct_term}"
            if ct_def:
                system_prompt += f"\n   Definition: {ct_def}"
        
        system_prompt += "\n\nReturn ONLY the submission value/code, nothing else.\nExamples: Y, N, INCLUSION, EXCLUSION, INC1"
        
        user_prompt = f"""Select the CT value for this field:

Field: {field_data.get('label', '')}
Options: {field_data.get('options', [])}
Context: This is {field_data.get('section', '')} in {field_data.get('form', '')}

Which controlled term value should be used?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"CT value selection for {variable}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        return response.strip().upper()
    
    def _get_all_ct_values_for_variable(self, variable: str, domain: str) -> List[Dict]:
        """Get ALL controlled terminology values from cdisc_ct.json and other sources"""
        ct_values = []
        
        # First check if we have cdisc_ct loaded
        if hasattr(self, 'cdisc_ct') and self.cdisc_ct:
            # Search for codelists that match the variable name
            for codelist in self.cdisc_ct.get('codelists', []):
                cl_name = codelist['codelist'].get('shortName', '').lower()
                var_lower = variable.lower()
                
                # Check if variable name appears in codelist name
                if var_lower in cl_name or (var_lower.replace(domain.lower(), '') in cl_name):
                    ct_values.extend(codelist.get('terms', []))
                    if ct_values:  # Found matching codelist
                        break
                        
            # Special handling for common patterns if no specific match found
            if not ct_values:
                if variable.endswith('YN'):
                    # Look for Yes/No Response codelist
                    for codelist in self.cdisc_ct.get('codelists', []):
                        cl_name = codelist['codelist'].get('shortName', '').lower()
                        if 'yes' in cl_name and 'no' in cl_name and 'response' in cl_name:
                            ct_values.extend(codelist.get('terms', []))
                            break
                    
                    # Fallback to standard Y/N
                    if not ct_values:
                        ct_values = [
                            {'submissionValue': 'Y', 'preferredTerm': 'Yes', 'definition': 'The affirmative response'},
                            {'submissionValue': 'N', 'preferredTerm': 'No', 'definition': 'The non-affirmative response'}
                        ]
                elif variable.endswith('TESTCD'):
                # Look for domain-specific test codes
                    domain_test = f"{domain} Test Code"
                for codelist in self.cdisc_ct.get('codelists', []):
                    if domain_test.lower() in codelist['codelist'].get('shortName', '').lower():
                        ct_values.extend(codelist.get('terms', []))
                        break
            else:
                # General search by variable name
                for codelist in self.cdisc_ct.get('codelists', []):
                    cl_name = codelist['codelist'].get('shortName', '').lower()
                    if variable.lower() in cl_name:
                        ct_values.extend(codelist.get('terms', []))
                        if len(ct_values) > 50:  # Limit if too many
                            break
        
        return ct_values
    
    def _build_direct_annotation_guided(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build direct mapping annotation with two-step variable and CT selection"""
        
        # Step 1: Select variable showing ALL domain variables
        variables = self._select_variables_for_pattern(field_data, domain, domain_vars, 'direct')
        
        if not variables:
            return f"{domain}.UNKNOWN", 0.3
        
        variable = variables[0]  # For direct mapping, we expect one variable
        
        # Check if this variable has controlled terminology
        has_ct = False
        for var in domain_vars:
            if var.get('name') == variable and var.get('codelist'):
                has_ct = True
                break
        
        if has_ct:
            # Step 2: Select CT value
            ct_value = self._select_ct_value_for_variable(field_data, domain, variable)
            if ct_value and ct_value != 'NONE':
                return f"{domain}.{variable} = {ct_value}", 0.9
        
        return f"{domain}.{variable}", 0.9
    
    def _build_checkbox_annotation_guided(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build checkbox/fixed value annotation"""
        
        relevant_vars = [v for v in domain_vars if any(kw in v.get('label', '').lower() 
                        for kw in ['flag', 'indicator', 'term', 'code'])]
        
        var_desc = "Available variables for checkbox/fixed value:\n"
        for var in relevant_vars:  # Show ALL relevant variables
            var_desc += f"- {var['name']}: {var.get('label', '')}\n"
        
        system_prompt = f"""You are selecting variables for a checkbox field in domain {domain}.
This will use the fixed value pattern where the variable is set to a specific value.

Return format:
VARIABLE: <variable_name>"""
        
        user_prompt = f"""Select variable for this checkbox field:

Question Text: {field_data.get('label', '')}
Options: {field_data.get('options', [])}

{var_desc}

Which variable should store this checkbox value?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 3: Checkbox variable selection in {domain}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        variable = response.strip()
        if variable.startswith('VARIABLE:'):
            variable = variable.replace('VARIABLE:', '').strip()
        
        # For yes/no options
        if field_data.get('options') and len(field_data.get('options', [])) == 2:
            return f"{domain}.{variable} = Y / N", 0.9
        else:
            return f"{domain}.{variable}", 0.8
    
    def _build_test_annotation_guided(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build test/measurement annotation for findings domains"""
        
        # Get test codes
        testcd_var = f"{domain}TESTCD"
        test_codes = []
        
        if testcd_var in self.ct_by_variable:
            test_codes = self.ct_by_variable[testcd_var]
        
        tc_desc = "Available test codes:\n"
        if test_codes:
            for tc in test_codes[:15]:
                tc_desc += f"- {tc.get('code', '')}: {tc.get('preferredTerm', '')}\n"
        
        system_prompt = f"""You are selecting test codes for a findings domain {domain}.
Select the appropriate test code and specify if units are needed.

Return format:
TESTCODE: <code>
HAS_UNITS: <yes/no>"""
        
        user_prompt = f"""Select test code for this measurement:

Question Text: {field_data.get('label', '')}
Has Units: {field_data.get('has_units', False)}

{tc_desc}

Which test code should be used?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 3: Test code selection for {domain}"
        response = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        
        # Parse response
        test_code = None
        has_units = False
        
        for line in response.strip().split('\n'):
            if line.startswith('TESTCODE:'):
                test_code = line.replace('TESTCODE:', '').strip()
            elif line.startswith('HAS_UNITS:'):
                has_units = 'yes' in line.lower()
        
        if test_code:
            if has_units or field_data.get('has_units'):
                return f"{domain}.{domain}ORRES / {domain}.{domain}ORRESU when {domain}.{domain}TESTCD = {test_code}", 0.9
            else:
                return f"{domain}.{domain}ORRES when {domain}.{domain}TESTCD = {test_code}", 0.9
        
        return f"{domain}.{domain}ORRES", 0.5
    
    def _build_other_specify_annotation_guided(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build other specify annotation"""
        
        # Find main variable
        label_parts = field_data.get('label', '').lower().split('other')[0].strip().split()
        
        relevant_vars = []
        for var in domain_vars:
            if any(part in var.get('label', '').lower() for part in label_parts if len(part) > 3):
                relevant_vars.append(var)
        
        var_desc = "Relevant variables:\n"
        for var in relevant_vars[:10]:
            var_desc += f"- {var['name']}: {var.get('label', '')}\n"
        
        system_prompt = f"""You are creating an "other specify" annotation for domain {domain}.
Identify the main variable that would have value "OTHER".

Return format:
MAIN_VARIABLE: <variable_name>"""
        
        user_prompt = f"""Create other specify annotation:

Question Text: {field_data.get('label', '')}

{var_desc}

Which is the main variable?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 3: Other specify variable selection for {domain}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        main_var = response.strip()
        if main_var.startswith('MAIN_VARIABLE:'):
            main_var = main_var.replace('MAIN_VARIABLE:', '').strip()
        
        # Generate OTH variable
        if len(main_var) <= 5:
            oth_var = main_var + "OTH"
        else:
            oth_var = main_var[:5] + "OTH"
        
        return f"{domain}.{oth_var} when {domain}.{main_var} = OTHER", 0.9
    
    def _build_supplemental_annotation_guided(self, field_data: Dict[str, Any], domain: str) -> Tuple[str, float]:
        """Build supplemental qualifier annotation"""
        
        system_prompt = f"""You are creating a supplemental qualifier (QNAM) for domain {domain}.
Create an 8-character uppercase identifier based on the field label.

Rules:
- Maximum 8 characters
- All uppercase
- No spaces or special characters
- Should be descriptive of the field content

Return format:
QNAM: <identifier>"""
        
        user_prompt = f"""Create QNAM for this supplemental field:

Question Text: {field_data.get('label', '')}
Domain: {domain}

What should the QNAM be?"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Step 3: QNAM creation for SUPP{domain}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        qnam = response.strip()
        if qnam.startswith('QNAM:'):
            qnam = qnam.replace('QNAM:', '').strip()
        
        return f"{qnam} in SUPP{domain}", 0.9
    
    def _build_cross_domain_annotation_guided(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> Tuple[str, float]:
        """Build cross-domain annotation"""
        
        # For dates that populate multiple domains
        if 'date' in field_data.get('label', '').lower():
            if 'consent' in field_data.get('label', '').lower():
                return "--DTC [DM.RFICDTC, DS.DSSTDTC when DSCAT = 'PROTOCOL MILESTONE']", 0.9
            elif 'baseline' in field_data.get('label', '').lower():
                return "--DTC [DM.RFSTDTC, SV.SVSTDTC]", 0.9
        
        # Default to single domain
        return self._build_direct_annotation_guided(field_data, domain, domain_vars)
    
    def _guided_annotation_builder(self, field_data: Dict[str, Any], pattern: str, domain: str) -> str:
        """Build annotation using guided selection based on pattern formula"""
        
        # Load domain variables from KB
        domain_vars = self._get_domain_variables(domain)
        
        # Build annotation based on pattern
        if pattern == 'direct':
            return self._guided_direct_pattern(field_data, domain, domain_vars)
        elif pattern == 'fixed_value':
            return self._guided_fixed_value_pattern(field_data, domain, domain_vars)
        elif pattern == 'conditional_population':
            return self._guided_conditional_pattern(field_data, domain, domain_vars)
        elif pattern == 'supplemental':
            return self._guided_supplemental_pattern(field_data, domain)
        elif pattern == 'multiple_variables':
            return self._guided_multiple_variables_pattern(field_data, domain, domain_vars)
        elif pattern == 'conditional_other_specify':
            return self._guided_other_specify_pattern(field_data, domain, domain_vars)
        else:
            # Fallback to simple annotation
            return f"{domain}.{domain}TERM"
    
    def _get_domain_variables(self, domain: str) -> List[Dict]:
        """Get all variables for a domain from variables_all.json"""
        domain_vars = []
        
        # Load from variables_all.json
        if hasattr(self, 'variables_all') and self.variables_all:
            domain_vars = [v for v in self.variables_all if v.get('domain') == domain]
        
        # Fallback to proto_define
        if not domain_vars and domain in self.proto_define.get('datasets', {}):
            proto_vars = self.proto_define['datasets'][domain].get('variables', {})
            domain_vars = [{'name': k, **v} for k, v in proto_vars.items()]
        
        return domain_vars
    
    def _guided_direct_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> str:
        """Guided selection for direct pattern: <Domain>.<Variable>"""
        
        # Build prompt showing all domain variables
        system_prompt = f"""Select the appropriate SDTM variable from {domain} domain.

Pattern: Direct mapping <Domain>.<Variable>
You must select ONE variable from the list below.

Available {domain} Variables:"""
        
        # Group variables by role
        vars_by_role = {}
        for var in domain_vars:
            role = var.get('role', 'Other')
            if role not in vars_by_role:
                vars_by_role[role] = []
            vars_by_role[role].append(var)
        
        # Build variable list
        for role in ['Identifier', 'Topic', 'Qualifier', 'Timing', 'Result Qualifier', 'Other']:
            if role in vars_by_role:
                system_prompt += f"\n\n{role} Variables:"
                for var in vars_by_role[role]:
                    system_prompt += f"\n- {var['name']}: {var.get('label', '')}"
                    if var.get('codelist'):
                        system_prompt += f" [CT: {var['codelist']}]"
        
        system_prompt += "\n\nReturn ONLY the variable name."
        
        user_prompt = f"""Field: {field_data.get('label', '')}
Type: {field_data.get('input_type', '')}
Options: {field_data.get('options', [])}

Select the most appropriate variable."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        debug_ctx = f"Guided variable selection for direct pattern: {field_data.get('label', '')}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        variable = response.strip().upper()
        
        # Validate variable exists
        valid_vars = [v['name'] for v in domain_vars]
        if variable not in valid_vars:
            # Try common patterns
            if f"{domain}TERM" in valid_vars:
                variable = f"{domain}TERM"
            elif f"{domain}TRT" in valid_vars:
                variable = f"{domain}TRT"
            else:
                variable = valid_vars[0] if valid_vars else f"{domain}TERM"
        
        return f"{domain}.{variable}"
    
    def _guided_fixed_value_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> str:
        """Guided selection for fixed value pattern: <Variable> = <Value>"""
        
        # First select variable
        variable_prompt = f"""Select the appropriate variable for this checkbox/fixed value field.

Domain: {domain}
Pattern: Fixed value assignment

Common variables for fixed values:"""
        
        # Show relevant variables
        relevant_vars = []
        for var in domain_vars:
            label_lower = var.get('label', '').lower()
            if any(kw in label_lower for kw in ['term', 'code', 'flag', 'indicator', 'status']):
                relevant_vars.append(var)
        
        for var in relevant_vars[:10]:
            variable_prompt += f"\n- {var['name']}: {var.get('label', '')}"
            if var.get('codelist'):
                variable_prompt += f" [CT: {var['codelist']}]"
        
        variable_prompt += "\n\nReturn ONLY the variable name."
        
        messages = [
            {"role": "system", "content": variable_prompt},
            {"role": "user", "content": f"Field: {field_data.get('label', '')}"}
        ]
        
        response = self._query_llm_with_messages(messages, max_tokens=50, 
                                                debug_context="Variable selection for fixed value")
        variable = response.strip().upper()
        
        # Now select the controlled term value
        ct_value = self._select_controlled_term(variable, domain, field_data)
        
        return f"{domain}.{variable} = {ct_value}"
    
    def _select_controlled_term(self, variable: str, domain: str, field_data: Dict[str, Any]) -> str:
        """Select appropriate controlled term for a variable"""
        
        # Load CT for this variable
        ct_values = self._get_ct_for_variable_selection(variable, domain)
        
        if not ct_values:
            # No CT found, use field label or option
            if field_data.get('options'):
                return field_data['options'][0].upper()
            else:
                return field_data.get('label', 'Y').upper()
        
        # Build CT selection prompt
        ct_prompt = f"""Select the appropriate controlled term value.

Variable: {variable}
Field: {field_data.get('label', '')}

Available Controlled Terms:"""
        
        for i, ct in enumerate(ct_values[:20]):
            ct_prompt += f"\n{i+1}. {ct['code']}: {ct.get('preferredTerm', ct.get('value', ''))}"
        
        ct_prompt += "\n\nReturn ONLY the code value."
        
        messages = [
            {"role": "system", "content": ct_prompt},
            {"role": "user", "content": "Select the most appropriate term."}
        ]
        
        response = self._query_llm_with_messages(messages, max_tokens=50,
                                                debug_context=f"CT selection for {variable}")
        
        return response.strip().upper()
    
    def _get_ct_for_variable_selection(self, variable: str, domain: str) -> List[Dict]:
        """Get controlled terminology values for guided selection"""
        ct_values = []
        
        # Load from cdisc_ct.json if available
        if hasattr(self, 'cdisc_ct') and self.cdisc_ct:
            # Search for matching codelist
            for codelist in self.cdisc_ct.get('codelists', []):
                # Match by variable name patterns
                if variable in ['SEX', 'RACE', 'ETHNIC', 'COUNTRY']:
                    if variable.lower() in codelist['codelist'].get('shortName', '').lower():
                        ct_values.extend(codelist.get('terms', []))
                        break
                elif variable.endswith('TESTCD'):
                    domain_test = f"{domain} Test Code"
                    if domain_test.lower() in codelist['codelist'].get('shortName', '').lower():
                        ct_values.extend(codelist.get('terms', []))
                        break
        
        # Fallback to common patterns
        if not ct_values:
            if variable == 'SEX':
                ct_values = [
                    {'code': 'M', 'preferredTerm': 'Male'},
                    {'code': 'F', 'preferredTerm': 'Female'},
                    {'code': 'U', 'preferredTerm': 'Unknown'}
                ]
            elif variable.endswith('YN') or variable in ['AESER', 'AESCAN', 'AESDTH']:
                ct_values = [
                    {'code': 'Y', 'preferredTerm': 'Yes'},
                    {'code': 'N', 'preferredTerm': 'No'}
                ]
        
        return ct_values
    
    def _guided_conditional_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> str:
        """Guided selection for conditional pattern in findings domains"""
        
        # For findings domains, we need TESTCD
        testcd_var = f"{domain}TESTCD"
        orres_var = f"{domain}ORRES"
        orresu_var = f"{domain}ORRESU"
        
        # Select test code
        test_codes = self._get_ct_for_variable_selection(testcd_var, domain)
        
        if test_codes:
            # Show test codes for selection
            tc_prompt = f"""Select the test code for this measurement.

Field: {field_data.get('label', '')}
Domain: {domain}

Available Test Codes:"""
            
            # Filter relevant test codes based on field label
            field_label_lower = field_data.get('label', '').lower()
            relevant_codes = []
            
            for tc in test_codes:
                tc_name = tc.get('preferredTerm', '').lower()
                if any(kw in field_label_lower for kw in tc_name.split()):
                    relevant_codes.append(tc)
            
            # Show relevant codes first, then others
            codes_to_show = relevant_codes[:10] if relevant_codes else test_codes[:10]
            
            for tc in codes_to_show:
                tc_prompt += f"\n- {tc['code']}: {tc.get('preferredTerm', '')}"
            
            tc_prompt += "\n\nReturn ONLY the test code."
            
            messages = [
                {"role": "system", "content": tc_prompt},
                {"role": "user", "content": "Select the appropriate test code."}
            ]
            
            response = self._query_llm_with_messages(messages, max_tokens=50,
                                                    debug_context=f"Test code selection for {domain}")
            test_code = response.strip().upper()
        else:
            # Guess based on label
            test_code = self._guess_test_code(field_data.get('label', ''))
        
        # Build annotation
        if field_data.get('has_units'):
            return f"{orres_var} / {orresu_var} when {testcd_var} = {test_code}"
        else:
            return f"{orres_var} when {testcd_var} = {test_code}"
    
    def _guess_test_code(self, label: str) -> str:
        """Guess test code based on label"""
        label_lower = label.lower()
        
        # Common patterns
        if 'systolic' in label_lower:
            return 'SYSBP'
        elif 'diastolic' in label_lower:
            return 'DIABP'
        elif 'pulse' in label_lower or 'heart rate' in label_lower:
            return 'PULSE'
        elif 'weight' in label_lower:
            return 'WEIGHT'
        elif 'height' in label_lower:
            return 'HEIGHT'
        elif 'temperature' in label_lower:
            return 'TEMP'
        else:
            # Extract uppercase abbreviation
            words = label.split()
            if words:
                return ''.join(w[0] for w in words if w).upper()[:8]
            return 'TEST'
    
    def _guided_supplemental_pattern(self, field_data: Dict[str, Any], domain: str) -> str:
        """Guided selection for supplemental pattern"""
        
        # Generate QNAM based on field
        label = field_data.get('label', '')
        
        # Common patterns
        if 'other' in label.lower() and 'specify' in label.lower():
            # Extract the main concept
            parts = label.lower().replace('other', '').replace('specify', '').replace(',', '').strip().split()
            if parts:
                qnam = parts[-1].upper()[:8]
                if len(qnam) < 8:
                    qnam = qnam + 'OTH'
            else:
                qnam = 'OTHSPEC'
        else:
            # Create from label
            words = label.split()[:2]
            qnam = ''.join(w[:4].upper() for w in words if w)[:8]
        
        return f"{qnam} in SUPP{domain}"
    
    def _guided_multiple_variables_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> str:
        """Guided selection for multiple variables pattern"""
        
        # Common pattern is term/decode
        term_vars = [v for v in domain_vars if v['name'].endswith('TERM')]
        decod_vars = [v for v in domain_vars if v['name'].endswith('DECOD')]
        
        if term_vars and decod_vars:
            # Select matching pair
            term_var = term_vars[0]['name']
            decod_var = None
            
            # Find matching DECOD
            term_prefix = term_var.replace('TERM', '')
            for dv in decod_vars:
                if dv['name'].startswith(term_prefix):
                    decod_var = dv['name']
                    break
            
            if decod_var:
                # Get value
                value = self._select_controlled_term(term_var, domain, field_data)
                return f"{domain}.{term_var} / {domain}.{decod_var} = {value}"
        
        # Fallback
        return f"{domain}.{domain}TERM"
    
    def _guided_other_specify_pattern(self, field_data: Dict[str, Any], domain: str, domain_vars: List[Dict]) -> str:
        """Guided selection for other specify pattern"""
        
        # Find the main variable (usually ends with OTH)
        oth_vars = [v for v in domain_vars if v['name'].endswith('OTH')]
        
        if oth_vars:
            oth_var = oth_vars[0]['name']
            main_var = oth_var.replace('OTH', '')
            
            return f"{domain}.{oth_var} when {domain}.{main_var} = OTHER"
        else:
            # Generate variable name
            label_parts = field_data.get('label', '').split('-')[0].strip().split()
            if label_parts:
                main_concept = label_parts[-1].upper()
                return f"{domain}.{main_concept}OTH when {domain}.{main_concept} = OTHER"
            
        return f"{domain}.OTHSPEC"
    
    def process_crf_json_file(self, json_path: Path) -> Dict[str, Any]:
        """Process a single CRF JSON file and annotate all items"""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"CRF JSON file not found: {json_path}")
        
        logger.info(f"Processing CRF JSON: {json_path}")
        
        # Load JSON
        with open(json_path, 'r') as f:
            crf_data = json.load(f)
        
        # Process each item
        results = {
            "file": str(json_path),
            "page_id": crf_data.get("page_id"),
            "annotations": [],
            "summary": {
                "total_items": 0,
                "annotated": 0,
                "not_submitted": 0,
                "errors": 0
            }
        }
        
        # Group items by question
        questions = {}
        for item in crf_data.get("items", []):
            if item.get("tag") == "<Q>":
                qid = item.get("qid")
                questions[qid] = {
                    "question": item,
                    "inputs": []
                }
            elif item.get("tag") == "<INPUT>":
                parent_qid = item.get("parent_qid")
                if parent_qid and parent_qid in questions:
                    questions[parent_qid]["inputs"].append(item)
        
        # Process each question with its inputs
        for qid, q_data in questions.items():
            question = q_data["question"]
            inputs = q_data["inputs"]
            
            # Build field data for annotation
            field_data = self._build_field_data_from_crf(question, inputs)
            
            try:
                # Annotate the field
                annotation_result = self.annotate_field(field_data)
                
                annotation_entry = {
                    "qid": qid,
                    "seq": question.get("seq"),
                    "text": question.get("text"),
                    "annotation": annotation_result.get("annotation"),
                    "domain": annotation_result.get("domain"),
                    "pattern": annotation_result.get("pattern"),
                    "confidence": annotation_result.get("confidence", 0),
                    "valid": annotation_result.get("valid", False)
                }
                
                # Add separate annotation fields if present
                if annotation_result.get("requires_separate_annotations", False):
                    annotation_entry["requires_separate_annotations"] = True
                    annotation_entry["question_annotation"] = annotation_result.get("question_annotation")
                    annotation_entry["option_annotations"] = annotation_result.get("option_annotations")
                
                results["annotations"].append(annotation_entry)
                
                results["summary"]["annotated"] += 1
                if annotation_result.get("annotation") == "[NOT SUBMITTED]":
                    results["summary"]["not_submitted"] += 1
                    
            except Exception as e:
                logger.error(f"Error annotating question {qid}: {str(e)}")
                results["annotations"].append({
                    "qid": qid,
                    "seq": question.get("seq"),
                    "text": question.get("text"),
                    "error": str(e)
                })
                results["summary"]["errors"] += 1
            
            results["summary"]["total_items"] += 1
        
        return results
    
    def process_crf_json_directory(self, directory: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Process all CRF JSON files in a directory"""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all JSON files
        json_files = sorted(directory.glob("page_*.json"))
        
        logger.info(f"Found {len(json_files)} CRF JSON files in {directory}")
        
        # Process each file
        all_results = {
            "directory": str(directory),
            "files_processed": len(json_files),
            "pages": [],
            "summary": {
                "total_items": 0,
                "annotated": 0,
                "not_submitted": 0,
                "errors": 0
            }
        }
        
        for json_file in json_files:
            page_results = self.process_crf_json_file(json_file)
            all_results["pages"].append(page_results)
            
            # Update summary
            all_results["summary"]["total_items"] += page_results["summary"]["total_items"]
            all_results["summary"]["annotated"] += page_results["summary"]["annotated"]
            all_results["summary"]["not_submitted"] += page_results["summary"]["not_submitted"]
            all_results["summary"]["errors"] += page_results["summary"]["errors"]
        
        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{directory.name}_annotations.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Annotations saved to: {output_file}")
        
        return all_results
    
    def _build_field_data_from_crf(self, question: Dict, inputs: List[Dict]) -> Dict[str, Any]:
        """Build field data structure from CRF question and inputs"""
        
        # Extract question text and metadata
        field_data = {
            "label": question.get("text") or "",  # Ensure None becomes empty string
            "form": question.get("form") or "",
            "section": question.get("section") or "",
            "seq": question.get("seq"),
            "qid": question.get("qid")
        }
        
        # Determine input type and options from inputs
        if inputs:
            input_types = [inp.get("input_type", "") for inp in inputs]
            
            if "option" in input_types:
                # Radio/checkbox options
                field_data["input_type"] = "checkbox" if len(inputs) > 2 else "radio"
                field_data["options"] = [inp.get("text", "") for inp in inputs if inp.get("input_type") == "option"]
            elif "text" in input_types:
                # Text input
                field_data["input_type"] = "text"
                # Check for date format hints
                text_hints = [inp.get("text") or "" for inp in inputs]
                if any("yyyy" in hint.lower() or "date" in hint.lower() for hint in text_hints if hint):
                    field_data["input_type"] = "date"
            else:
                field_data["input_type"] = inputs[0].get("input_type", "text")
        else:
            # No inputs, try to infer from question
            field_data["input_type"] = "text"
        
        # Check for units in question text
        if "(" in field_data["label"] and ")" in field_data["label"]:
            # Extract potential units
            import re
            units_match = re.search(r'\(([^)]+)\)$', field_data["label"])
            if units_match:
                potential_units = units_match.group(1)
                # Check if it's actually units (common patterns)
                if any(unit in potential_units.lower() for unit in ['mg', 'ml', 'mmhg', 'bpm', 'kg', 'cm', '°c', '°f']):
                    field_data["has_units"] = True
                    field_data["units"] = potential_units
        
        return field_data
    
    def annotate_field_original(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Original step-by-step annotation (kept for backward compatibility)"""
        try:
            # Start new debug conversation group if in debug mode
            if self.debug_mode and self.debug_conversations is not None:
                debug_field_start = len(self.debug_conversations)
            
            # Check cache first
            cache_key = self._get_cache_key(field_data)
            if cache_key in self.field_result_cache:
                logger.info(f"Using cached mapping for: {field_data.get('label', '')}")
                return self.field_result_cache[cache_key]
            
            # Check for operational fields
            if self._is_operational_field(field_data):
                result = {
                    "annotation": "[NOT SUBMITTED]",
                    "domain": None,
                    "pattern": "not_submitted",
                    "sdtm_class": None,
                    "valid": True,
                    "validation_message": "Operational field",
                    "confidence": 0.99
                }
                self.field_result_cache[cache_key] = result
                return result
            
            # Step 1: Domain Selection (direct, showing all domains)
            logger.debug(f"\nStep 1: Domain Selection for '{field_data.get('label', '')}'")
            domain, domain_confidence = self._select_domain_direct(field_data)
            
            # Step 2: Pattern Selection (using KB patterns)
            logger.debug(f"\nStep 2: Pattern Selection (Domain: {domain})")
            pattern, pattern_confidence = self._select_pattern_from_kb(field_data, domain)
            
            # Step 3: Variable Selection (using KB variables)
            logger.debug(f"\nStep 3: Variable Selection (Domain: {domain}, Pattern: {pattern})")
            annotation, var_confidence = self._select_variables_from_kb(field_data, pattern, domain)
            
            # Calculate combined confidence
            confidence = (pattern_confidence * 0.25 + 
                         domain_confidence * 0.45 + 
                         var_confidence * 0.30)
            
            # Determine class from domain
            sdtm_class = None
            for class_name, domains in self.domains_by_class.items():
                if any(d.get('code') == domain for d in domains):
                    sdtm_class = class_name
                    break
            
            # Validate using proto_define
            valid = True
            validation_message = ""
            if domain and annotation != "[NOT SUBMITTED]":
                valid, validation_message = self._validate_with_proto_define(
                    annotation, domain, pattern
                )
            
            result = {
                "annotation": annotation,
                "domain": domain,
                "pattern": pattern,
                "sdtm_class": sdtm_class,
                "valid": valid,
                "validation_message": validation_message,
                "confidence": confidence,
                "decision_path": {
                    "domain": f"{domain} ({domain_confidence:.2f})",
                    "pattern": f"{pattern} ({pattern_confidence:.2f})",
                    "variable": f"confidence: {var_confidence:.2f}"
                }
            }
            
            # Add debug conversations if in debug mode
            if self.debug_mode and self.debug_conversations is not None:
                debug_field_end = len(self.debug_conversations)
                result["debug_conversations"] = self.debug_conversations[debug_field_start:debug_field_end]
            
            # Cache result
            self.field_result_cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error annotating field: {str(e)}")
            return {
                "annotation": "[NOT SUBMITTED]",
                "domain": None,
                "pattern": "unknown",
                "sdtm_class": None,
                "valid": True,
                "validation_message": f"Error: {str(e)}",
                "confidence": 0.0,
                "decision_path": {"error": str(e)}
            }
    
    def _is_operational_field(self, field_data: Dict[str, Any]) -> bool:
        """Check if field is operational and not submitted to SDTM"""
        label = field_data.get('label', '').lower()
        
        # Operational keywords
        operational_keywords = [
            'page', 'initials', 'signature', 'sign', 'initial',
            'monitor', 'investigator', 'data entry', 'crf status',
            'visit date', 'assessment date', 'form status'
        ]
        
        return any(keyword in label for keyword in operational_keywords)
    
    def _select_pattern_from_kb(self, field_data: Dict[str, Any], domain: Optional[str]) -> Tuple[str, float]:
        """Select pattern using KB pattern definitions and decision criteria"""
        
        # Build system prompt for pattern selection
        system_prompt = """You are an SDTM annotation expert. Select the EXACT pattern that matches how this CRF field should be mapped.

IMPORTANT: Each pattern has a SPECIFIC FORMULA that MUST be followed:

1. direct: <VARIABLE>
   - Use when: Field maps directly to one SDTM variable
   - Example output: CMTRT, AEDECOD, VSTESTCD

2. conditional_population: <VARIABLE> when <TESTCD> = "<VALUE>"
   - Use when: Findings domains where result depends on test code
   - Example output: VSORRES when VSTESTCD = "SYSBP"

3. fixed_value: <VARIABLE> = "<VALUE>"
   - Use when: Field always maps to a constant value
   - Example output: AESER = "Y", DSTERM = "COMPLETED"

4. conditional_assignment: <VARIABLE> = "<VALUE>" when <CONDITION>
   - Use when: Different field values map to different SDTM values
   - Example output: IESTRESC = "Y" when response = "No"

5. supplemental: <QNAM> in SUPP<DOMAIN>
   - Use when: "Other, specify" or non-standard fields
   - Example output: AEOTH in SUPPAE

6. not_submitted: [NOT SUBMITTED]
   - Use when: Operational/administrative fields
   - Example output: [NOT SUBMITTED]

7. multiple_variables: <VARIABLE1> / <VARIABLE2>
   - Use when: One field maps to multiple variables
   - Example output: VSORRES / VSORRESU

Select ONLY the pattern name based on the field characteristics."""
        
        # Get all available patterns description
        patterns_desc = self._get_all_patterns_description()
        
        prompt = f"""Select the annotation pattern for this field:

Field Label: {field_data.get('label', '')}
Field Type: {field_data.get('input_type', '')}
Has Units: {field_data.get('has_units', False)}
Options: {field_data.get('options', [])}
Domain: {domain}

{patterns_desc}

Consider:
- Is this an "other, specify" field? → supplemental
- Is this a measurement with units? → conditional_population (findings)
- Is this a checkbox or fixed response? → fixed_value
- Does this have multiple choice options? → conditional_assignment
- Is this a simple text/date field? → direct
- Is this for tracking/operations only? → not_submitted

Return only the pattern name."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Pattern Selection for field: {field_data.get('label', '')} (Domain: {domain})"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract pattern
        response_clean = response.strip().lower()
        
        # Map to exact pattern names
        for pattern_key in self.patterns.keys():
            if pattern_key in response_clean:
                return pattern_key, 0.9
        
        # Fallback to scoring approach
        label_lower = field_data.get('label', '').lower()
        input_type = field_data.get('input_type', '').lower()
        has_units = field_data.get('has_units', False)
        options = field_data.get('options', [])
        
        # Score each pattern based on keywords and decision criteria
        pattern_scores = {}
        
        for pattern_key, pattern_info in self.patterns.items():
            score = 0.0
            
            # Check keywords
            keywords = pattern_info.get('keywords', [])
            for keyword in keywords:
                if keyword in label_lower or keyword in input_type:
                    score += 0.2
            
            # Check specific criteria for each pattern
            if pattern_key == "supplemental":
                if 'other' in label_lower and ('specify' in label_lower or 'describe' in label_lower):
                    score += 0.8
            
            elif pattern_key == "not_submitted":
                if self._is_operational_field(field_data):
                    score += 0.9
            
            elif pattern_key == "fixed_value":
                if input_type == 'checkbox' and not options:
                    score += 0.7
            
            elif pattern_key == "conditional_population":
                # Findings pattern is more likely for certain domains
                if domain and domain in ['VS', 'LB', 'EG', 'PE', 'QS', 'FA', 'IE', 'PC', 'PP']:
                    score += 0.3
                if has_units or ('result' in label_lower and 'test' in label_lower):
                    score += 0.8
                if any(kw in label_lower for kw in ['measurement', 'vital', 'lab', 'test']):
                    score += 0.2
            
            elif pattern_key == "conditional_assignment":
                if options and len(options) > 1:
                    score += 0.5
                    if any(kw in label_lower for kw in ['severity', 'grade', 'category']):
                        score += 0.3
            
            elif pattern_key == "relationship":
                if any(kw in label_lower for kw in ['link', 'related', 'associated']):
                    score += 0.6
            
            elif pattern_key == "derived":
                if any(kw in label_lower for kw in ['calculated', 'derived', 'age']):
                    score += 0.7
            
            pattern_scores[pattern_key] = score
        
        # Select pattern with highest score
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        
        # If no clear winner, default to direct
        if best_pattern[1] < 0.3:
            return "direct", 0.7
        
        return best_pattern[0], min(0.95, best_pattern[1])
    
    def _select_class_from_kb(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select SDTM class using KB class definitions and hierarchy"""
        
        # Load class definitions from KB if available
        if hasattr(self, 'class_definitions') and self.class_definitions:
            # Build system prompt from KB class definitions
            system_prompt = self._build_class_selection_prompt()
        else:
            # Fallback if KB not loaded
            system_prompt = """You are helping identify the SDTM General Observation Class for CRF fields.
Follow the decision hierarchy:
1. Is this field about something done to the subject? → Interventions
2. Does this field describe something that happened to the subject? → Events  
3. Is this field recording a measurement or observation? → Findings
4. Is this basic subject information? → Special Purpose"""
        
        prompt = f"""Identify the SDTM observation class for this CRF field:

Field Label: {field_data.get('label', '')}
Form Name: {field_data.get('form_name', '')}
Section: {field_data.get('section', '')}
Field Type: {field_data.get('input_type', '')}
Has Units: {field_data.get('has_units', False)}
Options: {field_data.get('options', [])}

Apply the decision criteria and return only the class name."""
        
        # Query LLM with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Class Selection for field: {field_data.get('label', '')}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract class
        response_clean = response.strip()
        
        # Map to exact class names from KB
        for class_name in self.class_definitions.keys():
            if class_name.lower() in response_clean.lower():
                return class_name, 0.9
        
        # Use keyword-based fallback from KB definitions
        return self._class_selection_fallback(field_data)
    
    def _build_class_selection_prompt(self) -> str:
        """Build system prompt from KB class definitions"""
        prompt = "You are helping identify the SDTM General Observation Class for CRF fields.\n\n"
        
        # Add overview from hierarchy
        if hasattr(self, 'class_hierarchy') and self.class_hierarchy:
            prompt += f"{self.class_hierarchy.get('description', '')}\n\n"
        
        # Add detailed class descriptions from KB
        prompt += "SDTM Observation Classes:\n\n"
        
        for class_name, class_info in self.class_definitions.items():
            prompt += f"**{class_name}**\n"
            
            # Use full description if available
            if 'full_description' in class_info:
                prompt += f"{class_info['full_description']}\n\n"
            else:
                prompt += f"{class_info['description']}\n\n"
            
            # Add decision criteria
            if 'decision_criteria' in class_info:
                prompt += f"Decision Criteria: {class_info['decision_criteria']}\n"
            
            # Add key questions
            prompt += f"Key Questions:\n"
            for q in class_info['key_questions']:
                prompt += f"- {q}\n"
            
            # Add example keywords
            if 'keywords' in class_info:
                prompt += f"\nKeywords: {', '.join(class_info['keywords'][:10])}\n"
            
            prompt += "\n"
        
        # Add decision hierarchy with detailed questions
        if hasattr(self, 'class_hierarchy') and self.class_hierarchy:
            prompt += "\nDecision Process:\n"
            for step in self.class_hierarchy.get('steps', []):
                prompt += f"\nStep {step['order']}: {step.get('detailed_question', step['question'])}\n"
                prompt += f"If YES → {step['if_yes']}\n"
                if 'examples' in step:
                    prompt += f"Examples: {step['examples']}\n"
        
        prompt += "\nAnalyze the field and return only the class name (Interventions, Events, Findings, or Special Purpose)."
        
        return prompt
    
    def _class_selection_fallback(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Fallback class selection using KB keywords"""
        
        label_lower = field_data.get('label', '').lower()
        field_text = f"{label_lower} {field_data.get('form_name', '').lower()} {field_data.get('section', '').lower()}"
        
        best_class = None
        best_score = 0.0
        
        # Score each class based on keyword matches from KB
        for class_name, class_info in self.class_definitions.items():
            score = 0.0
            
            # Check keywords
            keywords = class_info.get('keywords', [])
            for keyword in keywords:
                if keyword in field_text:
                    score += 0.1
            
            # Check if form name matches example domains
            form_name_upper = field_data.get('form_name', '').upper()
            for domain in class_info.get('example_domains', []):
                if domain in form_name_upper:
                    score += 0.3
            
            # Special checks based on class characteristics
            if class_name == "Findings" and field_data.get('has_units'):
                score += 0.3
            elif class_name == "Events" and any(kw in label_lower for kw in ['occurred', 'happened', 'experienced']):
                score += 0.2
            elif class_name == "Interventions" and any(kw in label_lower for kw in ['administered', 'given', 'dose']):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        # Default to Findings if no clear match
        if best_class is None or best_score < 0.3:
            return "Findings", 0.5
            
        return best_class, min(0.9, best_score)
    
    def _select_domain_direct(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select domain directly without class selection"""
        
        # Get all domain descriptions
        all_domains = self._get_all_domain_descriptions()
        
        # Build system prompt
        system_prompt = """You are helping select the appropriate SDTM domain for a CRF field.

Select the SDTM domain that best fits the CRF field based on the field label, form name, and section. 

Key considerations:
- Fields from "Inclusion Criteria" or "Exclusion Criteria" sections should map to IE domain
- Fields labeled with (INC#) or (EX#) patterns indicate inclusion/exclusion criteria
- Match the field content to the domain description
- Consider the form name and section as important context

Return only the domain code (e.g., 'AE', 'VS', 'CM', 'IE')."""
        
        prompt = f"""Select the SDTM domain for this field:

Field Label: {field_data.get('label', '')}
Form Name: {field_data.get('form_name', '')}
Section: {field_data.get('section', '')}
Field Type: {field_data.get('input_type', '')}
Options: {field_data.get('options', [])}

{all_domains}

Analyze the field and return only the domain code."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Domain Selection (Direct) for field: {field_data.get('label', '')}"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract domain code
        response_clean = response.strip().upper()
        
        # Validate domain exists
        for class_domains in self.domains_by_class.values():
            for domain in class_domains:
                if domain.get('code', '') == response_clean:
                    return response_clean, 0.9
        
        # Fallback
        return None, 0.0
    
    def _select_domain_from_kb(self, field_data: Dict[str, Any], sdtm_class: str) -> Tuple[str, float]:
        """Select domain following Step 2 from instructions"""
        
        # Build domain descriptions based on class
        domain_descriptions = self._get_domain_descriptions_for_class(sdtm_class)
        
        # System prompt based on Step 2 of instructions
        system_prompt = f"""You are helping select the appropriate SDTM domain for a CRF field.

Once you know the observation class ({sdtm_class}), determine the specific SDTM domain that best fits the CRF field. SDTM domains are more granular categories within each class, usually corresponding to common CRF forms or topics. To choose a domain, ask: "What standard domain does this piece of data belong to?". Use the CRF form context (e.g., form title or subject matter) and the question text for clues.

Questions for Domain Selection: 
- Is the CRF page clearly associated with a known domain? For example, an Adverse Event form almost certainly maps to AE. A concomitant meds form maps to CM. A "Study Drug Administration" log maps to EX. A "Vital Signs" form maps to VS. 
- If the form title or content doesn't obviously match a single domain, break the form into fields and consider each field separately – sometimes one form can collect data for multiple domains (though this is discouraged).

Always prefer standard domains from SDTMIG v3.4. Only if absolutely no standard domain applies should you propose a new one.

Return only the domain code (e.g., 'AE', 'VS', 'CM')."""
        
        prompt = f"""Select the SDTM domain for this field:

Field Label: {field_data.get('label', '')}
Form Name: {field_data.get('form_name', '')}
Section: {field_data.get('section', '')}
Observation Class: {sdtm_class}

{domain_descriptions}

Apply the domain selection criteria and return only the domain code."""
        
        # Query LLM with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Domain Selection for field: {field_data.get('label', '')} (Class: {sdtm_class})"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract domain code
        response_clean = response.strip().upper()
        
        # Validate domain exists in KB
        if sdtm_class in self.domains_by_class:
            for domain in self.domains_by_class[sdtm_class]:
                if domain.get('code', '') == response_clean:
                    return response_clean, 0.9
        
        # Try to find domain in any class if not found
        for class_name, domains in self.domains_by_class.items():
            for domain in domains:
                if domain.get('code', '') == response_clean:
                    logger.warning(f"Domain {response_clean} found in {class_name} instead of {sdtm_class}")
                    return response_clean, 0.7
        
        # Default based on KB domain keywords
        if sdtm_class in self.domains_by_class:
            field_label_lower = field_data.get('label', '').lower()
            best_match = None
            best_score = 0
            
            # Score each domain based on keyword matches
            for domain in self.domains_by_class[sdtm_class]:
                domain_code = domain.get('code', '')
                domain_name = domain.get('name', '').lower()
                domain_desc = domain.get('description', '').lower()
                
                # Calculate match score
                score = 0
                # Check domain name words
                for word in domain_name.split():
                    if len(word) > 3 and word in field_label_lower:
                        score += 2
                
                # Check description words
                for word in domain_desc.split()[:20]:  # First 20 words of description
                    if len(word) > 4 and word in field_label_lower:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = domain_code
            
            if best_match:
                return best_match, min(0.6, best_score * 0.1)
        
        return None, 0.0
    
    def _get_domain_descriptions_for_class(self, sdtm_class: str) -> str:
        """Get domain descriptions for the given class from KB"""
        
        # Always get domains from KB
        if sdtm_class in self.domains_by_class:
            domains = self.domains_by_class[sdtm_class]
            descriptions = f"{sdtm_class} Class Domains:\n"
            
            # Show all domains for the class
            for domain in domains:  # Show ALL domains
                code = domain.get('code', '')
                name = domain.get('name', '')
                desc = domain.get('description', '')
                descriptions += f"- {code} ({name}) – {desc}\n"
                
            return descriptions
        
        return f"No domains found for class {sdtm_class} in knowledge base"
    
    def _get_all_domain_descriptions(self) -> str:
        """Get descriptions for ALL domains from KB, organized by class"""
        descriptions = "SDTM Domains (organized by observation class):\n\n"
        
        # Order classes for consistent presentation
        class_order = ["Special Purpose", "Interventions", "Events", "Findings"]
        
        for sdtm_class in class_order:
            if sdtm_class in self.domains_by_class:
                domains = self.domains_by_class[sdtm_class]
                descriptions += f"**{sdtm_class} Class**:\n"
                
                for domain in domains:
                    code = domain.get('code', '')
                    name = domain.get('name', '')
                    desc = domain.get('description', '')
                    descriptions += f"- {code} ({name}) – {desc}\n"
                
                descriptions += "\n"
        
        return descriptions
    
    def _select_variables_from_kb(self, field_data: Dict[str, Any], pattern: str, domain: str) -> Tuple[str, float]:
        """Select variables from KB based on pattern and domain"""
        
        if not domain:
            return "[NOT SUBMITTED]", 0.0
        
        # Get domain variables from domain_variables (loaded from variables_all.json)
        domain_vars = self.domain_variables.get(domain, {})
        
        # Fallback to proto_define if no variables in domain_variables
        if not domain_vars and domain in self.proto_define.get('datasets', {}):
            domain_vars = self.proto_define['datasets'][domain].get('variables', {})
        
        # Log pattern selection for debug
        logger.debug(f"Selected pattern '{pattern}' for variable selection")
        
        # Build annotation based on pattern
        if pattern == "direct":
            return self._build_direct_annotation(field_data, domain, domain_vars)
        elif pattern == "fixed_value":
            return self._build_fixed_value_annotation(field_data, domain, domain_vars)
        elif pattern == "conditional_population":
            return self._build_conditional_population_annotation(field_data, domain, domain_vars)
        elif pattern == "conditional_assignment":
            return self._build_conditional_assignment_annotation(field_data, domain, domain_vars)
        elif pattern == "supplemental":
            return self._build_supplemental_annotation(field_data, domain)
        elif pattern == "not_submitted":
            return "[NOT SUBMITTED]", 0.9
        else:
            return f"{domain}TERM", 0.5
    
    def _build_direct_annotation(self, field_data: Dict[str, Any], domain: str, domain_vars: Dict) -> Tuple[str, float]:
        """Build direct pattern annotation following Step 3 from instructions"""
        
        # System prompt based on Step 3
        system_prompt = f"""You are an SDTM annotation expert using the DIRECT pattern.

PATTERN FORMULA: <VARIABLE>
Your output MUST be a single SDTM variable name, nothing else.

Within domain {domain}, select the ONE variable that best captures this field's data.

Key variable selection rules:
1. Topic Variables (what is being recorded):
   - Interventions: Use --TRT (e.g., CMTRT, EXTRT)
   - Events: Use --TERM (e.g., AETERM, MHTERM)  
   - Findings: Use --TESTCD or --TEST (e.g., VSTESTCD, LBTESTCD)

2. Date/Time Variables:
   - Use --DTC for dates (e.g., AESTDTC, CMDTC)
   - Use --STDTC for start dates
   - Use --ENDTC for end dates

3. Result Variables (Findings only):
   - Use --ORRES for original results
   - Use --STRESC for standardized character results

4. Qualifier Variables:
   - Use specific qualifiers like --OCCUR, --STAT, --REASND

IMPORTANT: Return ONLY the variable name (e.g., "CMTRT" or "AESTDTC").
Do NOT include formulas, explanations, or multiple variables."""
        
        # Get ALL variables for the domain (not filtered)
        all_variables_desc = self._get_all_variables_for_domain(domain, domain_vars)
        
        # Check for controlled terminology
        ct_info = ""
        if field_data.get('options'):
            ct_info = self._get_ct_info_for_field(field_data, domain, domain_vars)
        
        prompt = f"""Select the SDTM variable for this field:

Field Label: {field_data.get('label', '')}
Field Type: {field_data.get('input_type', '')}
Has Units: {field_data.get('has_units', False)}
Options: {field_data.get('options', [])}
Domain: {domain}

{all_variables_desc}
{ct_info}

Apply the variable selection criteria and return only the variable name."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Variable Selection (direct) for field: {field_data.get('label', '')} (Domain: {domain}, Pattern: direct)"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract variable
        response_clean = response.strip().upper()
        if response_clean in domain_vars:
            return response_clean, 0.9
        
        # Try without domain prefix
        if not response_clean.startswith(domain):
            prefixed = f"{domain}{response_clean}"
            if prefixed in domain_vars:
                return prefixed, 0.9
        
        # Common defaults based on domain and field type
        if self._is_date_field(field_data):
            # Look for DTC variables
            for var in domain_vars:
                if var.endswith('DTC'):
                    return var, 0.7
        
        # Default topic variables
        if f"{domain}TERM" in domain_vars:
            return f"{domain}TERM", 0.7
        elif f"{domain}TRT" in domain_vars:
            return f"{domain}TRT", 0.7
        
        return list(domain_vars.keys())[0] if domain_vars else f"{domain}TERM", 0.5
    
    def _build_conditional_population_annotation(self, field_data: Dict[str, Any], domain: str, domain_vars: Dict) -> Tuple[str, float]:
        """Build conditional population (findings) pattern annotation following instructions"""
        
        # System prompt based on findings mapping from instructions
        system_prompt = f"""You are an SDTM annotation expert using the CONDITIONAL_POPULATION pattern.

PATTERN FORMULA: <VARIABLE> when <TESTCD> = "<VALUE>"
For units: <VARIABLE1> / <VARIABLE2> when <TESTCD> = "<VALUE>"

This pattern is for Findings domains ({domain}) where:
- {domain}TESTCD identifies what was measured
- {domain}ORRES contains the result value
- {domain}ORRESU contains the units (if applicable)

Your task: Identify the correct TEST CODE for this measurement.

IMPORTANT: Return ONLY the test code value (e.g., "SYSBP", "HEIGHT", "WEIGHT").
Do NOT return the full formula - just the test code that goes in quotes."""
        
        # Get ALL variables for the domain first
        all_variables_desc = self._get_all_variables_for_domain(domain, domain_vars)
        
        # Get test codes from KB if available
        test_code_info = ""
        testcd_var = f"{domain}TESTCD"
        
        if hasattr(self, '_ct_data') and testcd_var in self._ct_data:
            test_codes = []
            field_label_lower = field_data.get('label', '').lower()
            
            # Find relevant test codes
            for code, term_data in self._ct_data[testcd_var].items():
                label = term_data.get('value', '').lower()
                # Score relevance
                if any(word in label for word in field_label_lower.split() if len(word) > 3):
                    test_codes.append({'code': code, 'label': term_data.get('value', ''), 'score': 2})
                elif any(word in field_label_lower for word in label.split() if len(word) > 3):
                    test_codes.append({'code': code, 'label': term_data.get('value', ''), 'score': 1})
            
            # Sort by relevance and take top matches
            test_codes.sort(key=lambda x: x['score'], reverse=True)
            if test_codes:
                test_code_info = "\nRelevant test codes from controlled terminology:\n"
                for tc in test_codes[:10]:
                    test_code_info += f"- {tc['code']}: {tc['label']}\n"
        
        prompt = f"""Select the test code for this findings field:

Field Label: {field_data.get('label', '')}
Domain: {domain}
Has Units: {field_data.get('has_units', False)}

{all_variables_desc}
{test_code_info}

Analyze the field label and return only the appropriate test code."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Variable Selection (findings) for field: {field_data.get('label', '')} (Domain: {domain}, Pattern: conditional_population)"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract test code
        testcd = response.strip().upper()
        
        # Validate test code format (should be uppercase, no spaces)
        if ' ' in testcd or not testcd.isalnum():
            # Try to extract just the code part
            parts = testcd.split()
            testcd = parts[0] if parts else "UNKNOWN"
        
        # Build annotation
        if field_data.get('has_units') and f"{domain}ORRESU" in domain_vars:
            annotation = f"{domain}ORRES / {domain}ORRESU when {testcd_var} = \"{testcd}\""
        else:
            annotation = f"{domain}ORRES when {testcd_var} = \"{testcd}\""
        
        return annotation, 0.85
    
    def _build_fixed_value_annotation(self, field_data: Dict[str, Any], domain: str, domain_vars: Dict) -> Tuple[str, float]:
        """Build fixed value pattern annotation"""
        
        # Get ALL variables for the domain
        all_variables_desc = self._get_all_variables_for_domain(domain, domain_vars)
        
        system_prompt = f"""You are an SDTM annotation expert using the FIXED_VALUE pattern.

PATTERN FORMULA: <VARIABLE> = "<VALUE>"

This pattern is for fields that always map to a constant value in SDTM.

Rules:
1. The VARIABLE must exist in domain {domain}
2. The VALUE should use controlled terminology when available
3. Common examples:
   - Checkbox checked: AESER = "Y"
   - Disposition status: DSTERM = "COMPLETED"
   - Occurrence flag: CMOCCUR = "Y"

Your output MUST follow the exact format: VARIABLE = "VALUE"
Example outputs: AESER = "Y", DSTERM = "RANDOMIZED", CMOCCUR = "N"

IMPORTANT: Include the equals sign and quotes around the value."""
        
        prompt = f"""Determine the fixed value for this field:

Field: {field_data.get('label', '')}
Domain: {domain}
Type: {field_data.get('input_type', '')}

{all_variables_desc}

Return the appropriate fixed value assignment."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Variable Selection (fixed_value) for field: {field_data.get('label', '')} (Domain: {domain}, Pattern: fixed_value)"
        response = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        
        # Clean up response
        if '=' in response and '"' in response:
            return response.strip(), 0.8
        
        # Default
        return f"{domain}TERM = \"UNKNOWN\"", 0.5
    
    def _build_conditional_assignment_annotation(self, field_data: Dict[str, Any], domain: str, domain_vars: Dict) -> Tuple[str, float]:
        """Build conditional assignment pattern annotation"""
        
        # Get ALL variables for the domain
        all_variables_desc = self._get_all_variables_for_domain(domain, domain_vars)
        
        system_prompt = f"""You are an SDTM annotation expert using the CONDITIONAL_ASSIGNMENT pattern.

PATTERN FORMULA: <VARIABLE> = "<VALUE>" when <CONDITION>

This pattern maps different field values to different SDTM values.

CRITICAL RULES for domain {domain}:
1. VARIABLE must exist in the domain
2. VALUE must use controlled terminology
3. CONDITION references the CRF field value

Special rules for IE domain:
- For inclusion criteria (INC):
  - IESTRESC = "N" when response = "Yes" (criteria MET)
  - IESTRESC = "Y" when response = "No" (criteria NOT MET)
- For exclusion criteria (EX):
  - IESTRESC = "Y" when response = "Yes" (criteria NOT MET)
  - IESTRESC = "N" when response = "No" (criteria MET)

Your output MUST follow the exact format: VARIABLE = "VALUE" when CONDITION
Example: IESTRESC = "Y" when response = "No"

IMPORTANT: Include equals sign, quotes, and 'when' keyword."""
        
        # Add specific CT info for common IE variables
        ct_info = ""
        if domain == "IE" and "IESTRESC" in domain_vars:
            ct_info = "\nControlled Terminology for IESTRESC:\n"
            ct_info += "- Y: Criteria Not Met\n"
            ct_info += "- N: Criteria Met\n"
        
        prompt = f"""Build conditional assignment for this field:

Field: {field_data.get('label', '')}
Domain: {domain}
Options: {field_data.get('options', [])}

{all_variables_desc}
{ct_info}

Create the appropriate conditional assignment using the controlled terminology values."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Variable Selection (conditional_assignment) for field: {field_data.get('label', '')} (Domain: {domain}, Pattern: conditional_assignment)"
        response = self._query_llm_with_messages(messages, max_tokens=100, debug_context=debug_ctx)
        
        if 'when' in response:
            return response.strip(), 0.8
        
        # For IE domain, use standard pattern
        if domain == "IE":
            return 'IESTRESC = "Y" when response = "No"', 0.8
        
        return f"{domain}TERM when CONDITION", 0.5
    
    def _build_supplemental_annotation(self, field_data: Dict[str, Any], domain: str) -> Tuple[str, float]:
        """Build supplemental pattern annotation following Step 5 from instructions"""
        
        # System prompt based on Step 5
        system_prompt = f"""You are an SDTM annotation expert using the SUPPLEMENTAL pattern.

PATTERN FORMULA: <QNAM> in SUPP<DOMAIN>

This pattern is for non-standard fields stored in supplemental qualifier datasets.

RULES:
1. QNAM = Qualifier Name (max 8 characters, uppercase)
2. DOMAIN = The parent domain ({domain})
3. Common QNAM patterns:
   - "Other, specify": Use [VAR]OTH (e.g., RACEOTH, AEOTH)
   - "Describe/Details": Use [VAR]DET or [VAR]TXT
   - "Reason for": Use [VAR]RSN
   - "Comments": Use [VAR]COM

Example outputs:
- RACEOTH in SUPPDM
- AEOTH in SUPPAE
- CMREAS in SUPPCM

Your output MUST follow the exact format: QNAM in SUPP{domain}
IMPORTANT: Do NOT include domain prefix in QNAM."""
        
        prompt = f"""Generate the SUPP annotation for this field:

Field Label: {field_data.get('label', '')}
Domain: {domain}
Field Type: {field_data.get('input_type', '')}

Common QNAM patterns:
- "Other, specify" → [VAR]OTH (e.g., RACEOTH)
- "Describe" or "Details" → [VAR]DET or [VAR]TXT  
- "Reason for" → [VAR]RSN
- "Comments" → [VAR]COM

Create a meaningful QNAM (max 8 characters) and return the annotation."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        debug_ctx = f"Variable Selection (supplemental) for field: {field_data.get('label', '')} (Domain: {domain})"
        response = self._query_llm_with_messages(messages, max_tokens=50, debug_context=debug_ctx)
        
        # Extract QNAM and format
        if 'in SUPP' in response:
            return response.strip(), 0.9
        
        # Fallback: Generate QNAM from field label
        label_words = field_data.get('label', '').upper().split()
        stop_words = {'THE', 'A', 'AN', 'OF', 'IF', 'YES', 'NO', 'PLEASE', 'PROVIDE'}
        
        # Check for common patterns
        if 'other' in field_data.get('label', '').lower() and 'specify' in field_data.get('label', '').lower():
            # Extract the variable name before "other"
            for i, word in enumerate(label_words):
                if word == 'OTHER' and i > 0:
                    qnam = label_words[i-1][:6] + 'OTH'
                    return f"{qnam} in SUPP{domain}", 0.8
        
        # Default QNAM generation
        qnam_words = [w for w in label_words if w not in stop_words and len(w) > 2][:2]
        if not qnam_words:
            qnam_words = label_words[:2]
        
        qnam = ''.join(w[:4] for w in qnam_words)[:8].upper()
        
        return f"{qnam} in SUPP{domain}", 0.7
    
    def save_debug_json(self, output_path: str, results: Dict[str, Any] = None):
        """Save debug conversations to JSON file with full prompts and responses
        
        Args:
            output_path: Path to save the debug JSON
            results: Optional annotation results to include (can be list or dict)
        """
        if not self.debug_mode or self.debug_conversations is None:
            logger.warning("Debug mode is not enabled. No debug data to save.")
            return
        
        debug_data = {
            "metadata": {
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "kb_path": str(self.kb_path),
                "total_llm_calls": len(self.debug_conversations),
                "use_all_patterns": self.use_all_patterns
            },
            "llm_conversations": self.debug_conversations
        }
        
        # Handle different result formats
        if results:
            if isinstance(results, dict) and "annotations" in results:
                # Handle annotate-file output format
                fields_debug = []
                for annotation in results.get("annotations", []):
                    field_debug = {
                        "question_id": annotation.get("qid"),
                        "question_text": annotation.get("text", ""),
                        "final_annotation": annotation.get("annotation", ""),
                        "domain": annotation.get("domain", ""),
                        "pattern": annotation.get("pattern", ""),
                        "confidence": annotation.get("confidence", 0),
                        "valid": annotation.get("valid", False)
                    }
                    fields_debug.append(field_debug)
                
                debug_data["annotated_fields"] = fields_debug
                debug_data["summary"] = results.get("summary", {})
            elif isinstance(results, list):
                # Handle legacy format
                fields_debug = []
                for result in results:
                    field_info = result.get('field', {})
                    annotation_info = result.get('annotation', {})
                    
                    field_debug = {
                        "field_label": field_info.get('label', ''),
                        "form_name": field_info.get('form_name', ''),
                        "final_annotation": annotation_info.get('annotation', ''),
                        "domain": annotation_info.get('domain', ''),
                        "pattern": annotation_info.get('pattern', ''),
                        "class": annotation_info.get('sdtm_class', ''),
                        "confidence": annotation_info.get('confidence', 0)
                    }
                    fields_debug.append(field_debug)
                
                debug_data["annotated_fields"] = fields_debug
        
        # Save to file with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Debug JSON saved to: {output_path} (Total LLM calls: {len(self.debug_conversations)})")
    
    def _validate_with_proto_define(self, annotation: str, domain: str, pattern: str) -> Tuple[bool, str]:
        """Validate annotation using proto_define"""
        
        try:
            # Use enhanced validator if available
            if hasattr(self, 'enhanced_validator') and self.enhanced_validator:
                result = self.enhanced_validator.validate_annotation(annotation, domain)
                return result['valid'], result.get('message', '')
            
            # Basic validation
            if not domain or domain not in self.proto_define.get('datasets', {}):
                return False, f"Domain {domain} not found in proto_define"
            
            domain_info = self.proto_define['datasets'][domain]
            domain_vars = domain_info.get('variables', {})
            
            # Extract variables from annotation
            import re
            var_pattern = r'\b([A-Z]+[A-Z0-9]*)\b'
            found_vars = re.findall(var_pattern, annotation)
            
            # Check if variables exist
            invalid_vars = []
            for var in found_vars:
                if var not in domain_vars and not var.startswith('SUPP'):
                    invalid_vars.append(var)
            
            if invalid_vars:
                return False, f"Variables not found in {domain}: {', '.join(invalid_vars)}"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return True, f"Validation error: {str(e)}"
    
    def _query_llm(self, prompt: str, max_tokens: int = 512, debug_context: str = None) -> str:
        """Query the LLM and return response"""
        
        # Format as chat
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return self._query_llm_with_messages(messages, max_tokens, debug_context)
    
    def _query_llm_with_messages(self, messages: List[Dict[str, str]], max_tokens: int = 512, debug_context: str = None) -> str:
        """Query the LLM with messages and return response"""
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.debug(f"LLM Response: {response[:200]}...")
        
        # Capture debug conversation if enabled
        if self.debug_mode and self.debug_conversations is not None and debug_context:
            self.debug_conversations.append({
                "context": debug_context,
                "messages": messages,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        
        return response
    
    def _get_cache_key(self, field_data: Dict[str, Any]) -> str:
        """Generate cache key from field data"""
        label = field_data.get('label', '').lower()
        form = field_data.get('form_name', '').lower()
        section = field_data.get('section', '').lower()
        return f"{form}|{section}|{label}"
    
    def _convert_to_structured_result(self, legacy_result: Dict[str, Any]) -> AnnotationResult:
        """Convert legacy result to structured AnnotationResult"""
        annotation_str = legacy_result.get('annotation', '[NOT SUBMITTED]')
        pattern = legacy_result.get('pattern', 'unknown')
        domain = legacy_result.get('domain')
        
        # Parse annotation string to create mappings
        mappings = []
        
        if annotation_str == '[NOT SUBMITTED]':
            mappings.append(SDTMMapping(variable='[NOT SUBMITTED]'))
        else:
            # Simple parsing - can be enhanced
            parts = annotation_str.split(' / ')
            for part in parts:
                if 'when' in part:
                    var_part, condition = part.split(' when ')
                    mappings.append(SDTMMapping(variable=var_part.strip(), condition=condition.strip()))
                elif '=' in part:
                    var_part, value = part.split('=', 1)
                    mappings.append(SDTMMapping(variable=var_part.strip(), value=value.strip().strip('"')))
                else:
                    mappings.append(SDTMMapping(variable=part.strip()))
        
        return AnnotationResult(
            pattern=pattern,
            domain=domain,
            mappings=mappings,
            confidence=legacy_result.get('confidence', 0.0),
            validation_status='valid' if legacy_result.get('valid', True) else 'invalid',
            validation_message=legacy_result.get('validation_message', '')
        )
    
    # Helper methods for KB loading
    def _load_form_class_mappings_from_kb(self) -> Dict[str, str]:
        """Load form/section to class mappings from KB domains"""
        mappings = {}
        
        # Build mappings from domain metadata in KB
        if hasattr(self, 'domains_by_class') and self.domains_by_class:
            for sdtm_class, domains in self.domains_by_class.items():
                for domain in domains:
                    domain_name = domain.get('name', '').lower()
                    domain_code = domain.get('code', '').lower()
                    
                    # Add variations of the domain name
                    if domain_name:
                        mappings[domain_name] = sdtm_class
                        # Add singular/plural variations
                        if domain_name.endswith('s'):
                            mappings[domain_name[:-1]] = sdtm_class
                        else:
                            mappings[domain_name + 's'] = sdtm_class
                    
                    # Add domain code
                    if domain_code:
                        mappings[domain_code] = sdtm_class
        
        # Add some generic mappings that might not be in domain names
        mappings.update({
            "demographics": "Special Purpose",
            "demography": "Special Purpose", 
            "eligibility": "Findings",
            "questionnaire": "Findings",
            "dosing": "Interventions"
        })
        
        return mappings
    
    def _get_testcd_mappings_from_kb(self, domain: str) -> Dict[str, str]:
        """Get test code mappings from KB controlled terminology"""
        mappings = {}
        
        # Look for domain-specific test codes in CT
        testcd_codelist = f"{domain}TESTCD"
        
        if hasattr(self, '_ct_data') and testcd_codelist in self._ct_data:
            ct_terms = self._ct_data[testcd_codelist]
            for code, term_data in ct_terms.items():
                # Create mapping from label keywords to test codes
                label = term_data.get('value', '').upper()
                if label:
                    # Add main label
                    mappings[label] = code
                    # Add simplified version (first word)
                    first_word = label.split()[0] if label.split() else ''
                    if first_word:
                        mappings[first_word] = code
        
        return mappings
    
    def _get_findings_domain_mappings_from_kb(self) -> Dict[str, List[str]]:
        """Get findings domain mappings from KB"""
        mappings = {}
        
        # Build mappings from test codes in CT
        if hasattr(self, '_ct_data'):
            # VS domain patterns
            if 'VSTESTCD' in self._ct_data:
                vs_patterns = []
                for code, term_data in self._ct_data['VSTESTCD'].items():
                    label = term_data.get('value', '').lower()
                    if label:
                        vs_patterns.append(label)
                mappings['VS'] = vs_patterns
            
            # LB domain patterns  
            if 'LBTESTCD' in self._ct_data:
                lb_patterns = []
                for code, term_data in self._ct_data['LBTESTCD'].items():
                    label = term_data.get('value', '').lower()
                    if label:
                        lb_patterns.append(label)
                mappings['LB'] = lb_patterns
            
            # EG domain patterns
            if 'EGTESTCD' in self._ct_data:
                eg_patterns = []
                for code, term_data in self._ct_data['EGTESTCD'].items():
                    label = term_data.get('value', '').lower()
                    if label:
                        eg_patterns.append(label)
                mappings['EG'] = eg_patterns
        
        return mappings
    
    def _get_parent_domain_mappings_from_kb(self) -> Dict[str, str]:
        """Get parent domain mappings for SUPP from KB"""
        mappings = {}
        
        # Build from domain metadata
        if hasattr(self, 'proto_define') and 'datasets' in self.proto_define:
            for domain_code, domain_info in self.proto_define['datasets'].items():
                domain_label = domain_info.get('label', '').lower()
                
                # Add various forms of the domain label
                if domain_label:
                    mappings[domain_label] = domain_code
                    # Add keywords from the label
                    for word in domain_label.split():
                        if len(word) > 3:  # Skip short words
                            mappings[word] = domain_code
        
        return mappings
    
    def _variable_has_ct(self, variable_name: str, domain: str) -> bool:
        """Check if a variable has controlled terminology"""
        # Check direct mapping
        if variable_name in self.ct_by_variable:
            return True
        
        # Check common patterns
        if 'NY' in variable_name or variable_name.endswith('YN'):
            return True
            
        # Check TESTCD pattern
        if variable_name.endswith('TESTCD'):
            testcd_ct_name = f"{domain} Test Code"
            for ct_name in self.ct_by_variable:
                if testcd_ct_name.lower() in ct_name.lower():
                    return True
        
        return False
    
    def _get_ct_for_variable(self, variable_name: str, domain: str) -> str:
        """Get controlled terminology for a specific variable"""
        ct_info = ""
        
        # Check if variable has CT
        if variable_name in self.ct_by_variable:
            ct_data = self.ct_by_variable[variable_name]
            if 'terms' in ct_data:
                ct_info = f"\nControlled Terminology for {variable_name}:\n"
                ct_info += f"Definition: {ct_data.get('definition', '')}\n"
                ct_info += f"Extensible: {ct_data.get('extensible', 'true')}\n"
                ct_info += "Terms:\n"
                for term in ct_data['terms'][:10]:  # Show first 10 terms
                    ct_info += f"- {term['code']}: {term['value']}\n"
                if len(ct_data['terms']) > 10:
                    ct_info += f"... and {len(ct_data['terms']) - 10} more terms\n"
        
        # Also check for common CT patterns
        # For Yes/No fields
        if 'NY' in variable_name or variable_name.endswith('YN'):
            ct_info += "\nControlled Terminology (Yes/No):\n"
            ct_info += "- Y: Yes\n"
            ct_info += "- N: No\n"
        
        # For TESTCD variables
        if variable_name.endswith('TESTCD'):
            # Look for domain-specific test codes
            testcd_ct_name = f"{domain} Test Code"
            for ct_name, ct_data in self.ct_by_variable.items():
                if testcd_ct_name.lower() in ct_name.lower():
                    ct_info += f"\nControlled Terminology for {variable_name}:\n"
                    ct_info += f"Definition: {ct_data.get('definition', '')}\n"
                    ct_info += "Test Codes:\n"
                    for term in ct_data.get('terms', [])[:10]:
                        ct_info += f"- {term['code']}: {term['value']}\n"
                    if len(ct_data.get('terms', [])) > 10:
                        ct_info += f"... and {len(ct_data['terms']) - 10} more test codes\n"
                    break
        
        return ct_info
    
    def _get_all_patterns_description(self) -> str:
        """Get descriptions of all available annotation patterns"""
        desc = "Available Annotation Patterns:\n\n"
        
        for pattern_key, pattern_info in self.patterns.items():
            desc += f"**{pattern_key}**:\n"
            desc += f"- Description: {pattern_info.get('description', '')}\n"
            desc += f"- Formula: {pattern_info.get('formula', '')}\n"
            if pattern_info.get('keywords'):
                desc += f"- Keywords: {', '.join(pattern_info.get('keywords', []))}\n"
            desc += "\n"
        
        return desc
    
    def _get_all_variables_for_domain(self, domain: str, domain_vars) -> str:
        """Get ALL variables for a domain without filtering"""
        if not domain_vars:
            return f"No variables found for domain {domain}"
        
        # Convert to dict if it's a list
        if isinstance(domain_vars, list):
            domain_vars_dict = {var.get('name', ''): var for var in domain_vars}
        else:
            domain_vars_dict = domain_vars
        
        desc = f"All Variables for {domain} Domain:\n\n"
        
        # Group variables by role
        variables_by_role = {}
        for var_name, var_info in domain_vars_dict.items():
            role = var_info.get('role', 'Other')
            if role not in variables_by_role:
                variables_by_role[role] = []
            variables_by_role[role].append({
                'name': var_name,
                'label': var_info.get('label', ''),
                'type': var_info.get('type', ''),
                'codelist': var_info.get('codelist', ''),
                'has_ct': self._variable_has_ct(var_name, domain)
            })
        
        # Display in a logical order
        role_order = ['Identifier', 'Topic', 'Qualifier', 'Timing', 'Timing Qualifier', 
                     'Result Qualifier', 'Grouping Qualifier', 'Record Qualifier', 'Other']
        
        for role in role_order:
            if role in variables_by_role:
                desc += f"{role} Variables:\n"
                for var in variables_by_role[role]:
                    desc += f"- {var['name']}: {var['label']}"
                    if var['has_ct']:
                        desc += " [CT]"
                    desc += "\n"
                desc += "\n"
        
        # Add section for variables with controlled terminology
        desc += "\nVariables with Controlled Terminology:\n"
        ct_found = False
        for var_name in domain_vars:
            ct_info = self._get_ct_for_variable(var_name, domain)
            if ct_info:
                desc += ct_info
                ct_found = True
        
        if not ct_found:
            desc += "No controlled terminology found for this domain's variables.\n"
        
        return desc
    
    def _get_variable_descriptions_for_mapping(self, domain: str, domain_vars, field_data: Dict[str, Any]) -> str:
        """Get variable descriptions relevant for mapping"""
        
        # Convert to dict if it's a list
        if isinstance(domain_vars, list):
            domain_vars_dict = {var.get('name', ''): var for var in domain_vars}
        else:
            domain_vars_dict = domain_vars
        
        # Categorize variables by role
        topic_vars = []
        timing_vars = []
        qualifier_vars = []
        result_vars = []
        
        for var_name, var_info in domain_vars_dict.items():
            role = var_info.get('role', '')
            label = var_info.get('label', '')
            
            if role == 'Topic':
                topic_vars.append(f"{var_name}: {label}")
            elif role in ['Timing', 'Timing Qualifier'] or var_name.endswith('DTC'):
                timing_vars.append(f"{var_name}: {label}")
            elif role in ['Result Qualifier', 'Grouping Qualifier', 'Record Qualifier'] or 'result' in label.lower():
                result_vars.append(f"{var_name}: {label}")
            elif role == 'Qualifier':
                qualifier_vars.append(f"{var_name}: {label}")
        
        # Build description based on what's relevant
        desc = "Available Variables:\n\n"
        
        if topic_vars:
            desc += "Topic Variables (main subject of observation):\n"
            for var in topic_vars:  # Show ALL variables
                desc += f"- {var}\n"
            desc += "\n"
        
        if self._is_date_field(field_data) and timing_vars:
            desc += "Timing Variables (dates/times):\n"
            for var in timing_vars:  # Show ALL variables
                desc += f"- {var}\n"
            desc += "\n"
        
        if field_data.get('has_units') and result_vars:
            desc += "Result Variables (measurements/values):\n"
            for var in result_vars:  # Show ALL variables
                desc += f"- {var}\n"
            desc += "\n"
        
        if qualifier_vars:
            desc += "Qualifier Variables (additional details):\n"
            for var in qualifier_vars:  # Show ALL variables
                desc += f"- {var}\n"
        
        return desc
    
    def _is_date_field(self, field_data: Dict[str, Any]) -> bool:
        """Check if field is likely a date field"""
        label = field_data.get('label', '').lower()
        input_type = field_data.get('input_type', '').lower()
        
        date_keywords = ['date', 'time', 'day', 'month', 'year', 'when']
        return any(keyword in label for keyword in date_keywords) or input_type == 'date'
    
    def _get_ct_info_for_field(self, field_data: Dict[str, Any], domain: str, domain_vars) -> str:
        """Get controlled terminology info for field with options from KB"""
        
        # Convert to dict if it's a list
        if isinstance(domain_vars, list):
            domain_vars_dict = {var.get('name', ''): var for var in domain_vars}
        else:
            domain_vars_dict = domain_vars
        
        label_lower = field_data.get('label', '').lower()
        options = field_data.get('options', [])
        
        ct_info = "\nControlled Terminology Considerations:\n"
        ct_found = False
        
        # Check variables in the domain for those with CT
        for var_name, var_info in domain_vars_dict.items():
            if 'codelist' in var_info or 'controlled_terms' in var_info:
                # Check if this variable might be relevant based on label
                var_label_lower = var_info.get('label', '').lower()
                
                # Check for keyword matches
                if any(keyword in label_lower for keyword in var_label_lower.split()):
                    ct_name = var_info.get('codelist', var_info.get('controlled_terms', 'controlled terminology'))
                    ct_info += f"- Consider {var_name} with {ct_name}\n"
                    ct_found = True
        
        # Check for common patterns using KB data
        if hasattr(self, 'common_ct') and self.common_ct:
            # Check for Yes/No pattern
            if len(options) == 2 and any('yes' in opt.lower() and 'no' in opt2.lower() 
                                        for opt in options for opt2 in options if opt != opt2):
                ct_info += "- This appears to be a Yes/No field (NY codelist)\n"
                ct_found = True
            
            # Check if options match any CT values
            for ct_name, ct_values in self.common_ct.items():
                if isinstance(ct_values, list):
                    # Convert options to comparable format
                    option_values = [opt.upper() if isinstance(opt, str) else str(opt) for opt in options]
                    
                    # Check for matches
                    matches = []
                    for ct_item in ct_values:
                        if isinstance(ct_item, dict):
                            ct_code = ct_item.get('code', '').upper()
                            ct_value = ct_item.get('value', '').upper()
                            if ct_code in option_values or ct_value in option_values:
                                matches.append(ct_code)
                    
                    if matches:
                        ct_info += f"- Options match {ct_name} codelist: {', '.join(matches[:3])}\n"
                        ct_found = True
        
        if not ct_found:
            ct_info += "- No standard controlled terminology identified. Consider if this needs a sponsor-defined codelist.\n"
            
        return ct_info


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified SDTM Mapper - Enhanced with all patterns")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process CRF JSON file command
    file_parser = subparsers.add_parser('annotate-file', help='Annotate a CRF JSON file')
    file_parser.add_argument("json_file", help="Path to CRF JSON file")
    file_parser.add_argument("--output", help="Output file for annotations (default: annotations/<filename>_annotated.json)")
    file_parser.add_argument("--kb-path", help="Path to KB directory")
    file_parser.add_argument("--use-all-patterns", action="store_true", default=True, help="Use all-patterns mode")
    file_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    file_parser.add_argument("--debug-output", help="Debug output file (default: annotations/<filename>_debug.json)")
    
    # Process directory command
    dir_parser = subparsers.add_parser('annotate-dir', help='Annotate all CRF JSON files in directory')
    dir_parser.add_argument("directory", help="Directory containing CRF JSON files")
    dir_parser.add_argument("--output-dir", help="Output directory for annotations")
    dir_parser.add_argument("--kb-path", help="Path to KB directory")
    dir_parser.add_argument("--use-all-patterns", action="store_true", default=True, help="Use all-patterns mode")
    dir_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Legacy command for backward compatibility
    legacy_parser = subparsers.add_parser('legacy', help='Legacy page processing')
    legacy_parser.add_argument("page_json", help="Path to CRF page JSON")
    legacy_parser.add_argument("--proto-define", help="Path to proto_define.json")
    legacy_parser.add_argument("--kb-path", help="Path to KB directory")
    legacy_parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="LLM model")
    legacy_parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization")
    legacy_parser.add_argument("--no-4bit", dest="use_4bit", action="store_false", help="Disable 4-bit quantization")
    legacy_parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    legacy_parser.add_argument("--output", default="result.json", help="Output file")
    legacy_parser.add_argument("--debug-output", default="debug.json", help="Debug output file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle different commands
    if args.command == 'annotate-file':
        # Initialize mapper for file annotation
        mapper = UnifiedSDTMMapper(
            kb_path=args.kb_path,
            use_all_patterns=args.use_all_patterns,
            debug_mode=args.debug
        )
        
        # Process file
        results = mapper.process_crf_json_file(Path(args.json_file))
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            # Create default output path in annotations folder
            input_path = Path(args.json_file)
            output_dir = Path("annotations")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_annotated.json"
        
        # Save the results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Annotations saved to {output_path}")
        
        # Print summary
        print(f"\nAnnotation Summary:")
        print(f"Total items: {results['summary']['total_items']}")
        print(f"Annotated: {results['summary']['annotated']}")
        print(f"Not submitted: {results['summary']['not_submitted']}")
        print(f"Errors: {results['summary']['errors']}")
        
        # Save debug JSON if debug mode is enabled
        if args.debug:
            if args.debug_output:
                debug_path = Path(args.debug_output)
            else:
                # Create default debug output path
                input_path = Path(args.json_file)
                output_dir = Path("annotations")
                output_dir.mkdir(exist_ok=True)
                debug_path = output_dir / f"{input_path.stem}_debug.json"
            
            mapper.save_debug_json(str(debug_path), results)
            print(f"\nDebug JSON saved to {debug_path}")
    
    elif args.command == 'annotate-dir':
        # Initialize mapper for directory processing
        mapper = UnifiedSDTMMapper(
            kb_path=args.kb_path,
            use_all_patterns=args.use_all_patterns,
            debug_mode=args.debug
        )
        
        # Process directory
        output_dir = Path(args.output_dir) if args.output_dir else None
        results = mapper.process_crf_json_directory(Path(args.directory), output_dir)
        
        # Print summary
        print(f"\nDirectory Annotation Summary:")
        print(f"Files processed: {results['files_processed']}")
        print(f"Total items: {results['summary']['total_items']}")
        print(f"Annotated: {results['summary']['annotated']}")
        print(f"Not submitted: {results['summary']['not_submitted']}")
        print(f"Errors: {results['summary']['errors']}")
    
    elif args.command == 'legacy':
        # Legacy processing
        mapper = UnifiedSDTMMapper(
            proto_define_path=args.proto_define,
            kb_path=args.kb_path,
            model_name=args.model,
            use_4bit=args.use_4bit,
            use_rag=not args.no_rag,
            debug_mode=args.debug
        )
        
        # Process page
        with open(args.page_json, 'r') as f:
            page_data = json.load(f)
        
        results = []
        for question in page_data.get('questions', []):
            for field in question.get('fields', []):
                result = mapper.annotate_field(field)
                results.append({
                    'field': field,
                    'annotation': result
                })
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.output}")
        
        # Save debug JSON if debug mode is enabled
        if args.debug:
            mapper.save_debug_json(args.debug_output, results)
            print(f"Debug JSON saved to {args.debug_output}")


# Alias for backward compatibility
SDTMMapperKBDriven = UnifiedSDTMMapper

if __name__ == "__main__":
    main()