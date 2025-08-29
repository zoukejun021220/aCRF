"""Core UnifiedSDTMMapper class - orchestrates the mapping process"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..kb_handlers import KBLoader
from ..models import (
    LLMModel, RAGModel,
    SDTMMapping, VLMEntry, SUPPEntry, RelationshipEntry,
    OptionAnnotation, AnnotationResult
)
from ..processors import (
    ClassSelector, DomainSelector, PatternSelector, VariableSelector
)
from ..utils import FileProcessor, DebugManager

# Import validators from parent directory
try:
    from ...proto_define_validator import ProtoDefineValidator
    from ...enhanced_validator import EnhancedSDTMValidator
    from ...rule_based_filter import RuleBasedFilter
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from proto_define_validator import ProtoDefineValidator
    from enhanced_validator import EnhancedSDTMValidator  
    from rule_based_filter import RuleBasedFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedSDTMMapper:
    """Unified SDTM Mapper - KB-driven annotation using LLM"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-14B-Instruct", 
                 proto_define_path: str = None,
                 kb_path: str = None,
                 device: str = "auto",
                 use_4bit: bool = True,
                 debug: bool = False,
                 enable_rag: bool = True,
                 use_cache: bool = True):
        """
        Initialize the unified SDTM mapper
        
        Args:
            model_name: LLM model to use
            proto_define_path: Path to proto_define.json
            kb_path: Path to KB directory
            device: Device to use (auto, cuda, cpu)
            use_4bit: Use 4-bit quantization
            debug: Enable debug mode
            enable_rag: Enable RAG for test code search
            use_cache: Enable caching of annotations
        """
        self.model_name = model_name
        self.device = device
        self.use_4bit = use_4bit
        self.debug_mode = debug
        self.enable_rag = enable_rag
        self.use_cache = use_cache
        
        # Initialize cache
        self.cache = {} if use_cache else None
        
        # Initialize KB loader
        self.kb_loader = KBLoader(kb_path)
        self.proto_define = self.kb_loader.load_proto_define(proto_define_path)
        self.kb_loader.load_kb_resources()
        
        # Extract loaded resources
        self.domains_by_class = self.kb_loader.domains_by_class
        self.class_definitions = self.kb_loader.class_definitions
        self.patterns = self.kb_loader.patterns
        self.domain_variables = self.kb_loader.domain_variables
        
        # Initialize models
        self.llm_model = LLMModel(model_name, device, use_4bit)
        self.llm_model.initialize()
        
        if enable_rag:
            self.rag_model = RAGModel()
            self.rag_model.initialize()
            # Encode test codes if available
            test_codes = self._extract_test_codes()
            if test_codes:
                self.rag_model.encode_test_codes(test_codes)
        else:
            self.rag_model = None
        
        # Initialize processors
        self.class_selector = ClassSelector(self.class_definitions, self.llm_model)
        self.domain_selector = DomainSelector(self.domains_by_class, self.proto_define, self.llm_model)
        self.pattern_selector = PatternSelector(self.patterns, self.llm_model)
        self.variable_selector = VariableSelector(
            self.domain_variables, 
            self.kb_loader.controlled_terms,
            self.proto_define,
            self.llm_model
        )
        
        # Initialize validators
        try:
            self.proto_validator = ProtoDefineValidator(self.proto_define)
            self.enhanced_validator = EnhancedSDTMValidator(kb_path)
            self.rule_filter = RuleBasedFilter()
        except:
            logger.warning("Validators not available")
            self.proto_validator = None
            self.enhanced_validator = None
            self.rule_filter = None
            
        # Initialize utilities
        self.file_processor = FileProcessor()
        self.debug_manager = DebugManager(debug)
        
        # Load QRS instruments if available
        qrs_path = Path(__file__).parent.parent.parent / "kb" / "qrs_instruments.json"
        if qrs_path.exists():
            with open(qrs_path, 'r') as f:
                self.qrs_instruments = json.load(f)
        else:
            self.qrs_instruments = {}
            
        logger.info(f"Mapper initialized with {len(self.proto_define.get('datasets', {}))} domains")
    
    def annotate_field(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main annotation method - uses step-by-step approach
        """
        start_time = time.time()
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(field_data)
            if cache_key in self.cache:
                logger.info("Using cached annotation")
                return self.cache[cache_key]
        
        # Check if operational field (skip if yes)
        if self._is_operational_field(field_data):
            result = {
                "annotation": "skip",
                "pattern": "skip", 
                "reason": "operational field",
                "processing_time": time.time() - start_time
            }
            if self.use_cache:
                self.cache[cache_key] = result
            return result
        
        # Step 1: Select class
        sdtm_class, class_conf = self.class_selector.select_class(field_data)
        self.debug_manager.log_step("Class Selection", sdtm_class, class_conf)
        
        # Step 2: Select domain
        domain, domain_conf = self.domain_selector.select_domain(field_data, sdtm_class)
        self.debug_manager.log_step("Domain Selection", domain, domain_conf)
        
        # Step 3: Select pattern
        pattern, pattern_conf = self.pattern_selector.select_pattern(field_data, domain)
        self.debug_manager.log_step("Pattern Selection", pattern, pattern_conf)
        
        # Step 4: Select variables and build annotation
        annotation, var_conf = self.variable_selector.select_variables(field_data, domain, pattern)
        self.debug_manager.log_step("Variable Selection", annotation, var_conf)
        
        # Validate if validators available
        valid = True
        validation_message = ""
        
        if self.proto_validator and domain != "UNKNOWN":
            valid, validation_message = self._validate_with_proto_define(annotation, domain, pattern)
        
        # Build result
        result = {
            "annotation": annotation,
            "pattern": pattern,
            "domain": domain,
            "class": sdtm_class,
            "valid": valid,
            "validation_message": validation_message,
            "confidence": min(class_conf, domain_conf, pattern_conf, var_conf),
            "processing_time": time.time() - start_time
        }
        
        # Cache result
        if self.use_cache:
            self.cache[cache_key] = result
            
        return result
    
    def annotate_field_enhanced(self, field_data: Dict[str, Any]) -> AnnotationResult:
        """Enhanced annotation returning structured result"""
        # Get basic annotation
        result = self.annotate_field(field_data)
        
        # Convert to structured format
        return self._convert_to_structured_result(result)
        
    def process_crf_json_file(self, json_path: Path) -> Dict[str, Any]:
        """Process a CRF JSON file"""
        return self.file_processor.process_json_file(json_path, self)
        
    def process_crf_json_directory(self, directory: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Process directory of CRF JSON files"""
        return self.file_processor.process_json_directory(directory, output_dir, self)
    
    def save_debug_json(self, output_path: str, results: Dict[str, Any] = None):
        """Save debug information"""
        self.debug_manager.save_debug_json(output_path, results)
    
    def _is_operational_field(self, field_data: Dict[str, Any]) -> bool:
        """Check if field is operational (non-data collection)"""
        label_lower = field_data.get('label', '').lower()
        
        operational_terms = [
            'page', 'form', 'site', 'investigator', 'signature',
            'version', 'protocol', 'visit', 'date of completion'
        ]
        
        return any(term in label_lower for term in operational_terms)
    
    def _get_cache_key(self, field_data: Dict[str, Any]) -> str:
        """Generate cache key for field data"""
        key_parts = [
            field_data.get('label', ''),
            field_data.get('type', ''),
            ','.join(sorted(field_data.get('options', [])))
        ]
        return '|'.join(key_parts)
        
    def _validate_with_proto_define(self, annotation: str, domain: str, pattern: str) -> Tuple[bool, str]:
        """Validate annotation using proto_define"""
        try:
            # Use enhanced validator if available
            if hasattr(self, 'enhanced_validator') and self.enhanced_validator:
                result = self.enhanced_validator.validate_mapping(
                    annotation, domain, pattern
                )
                return result.is_valid, result.message
            
            # Basic validation
            if not domain or domain not in self.proto_define.get('datasets', {}):
                return False, f"Domain {domain} not found in proto_define"
            
            # Extract variables from annotation
            import re
            var_pattern = r'([A-Z]+)='
            variables = re.findall(var_pattern, annotation)
            
            domain_info = self.proto_define['datasets'][domain]
            domain_vars = domain_info.get('variables', {})
            
            invalid_vars = []
            for var in variables:
                if var not in domain_vars and not var.startswith('SUPP'):
                    invalid_vars.append(var)
                    
            if invalid_vars:
                return False, f"Invalid variables for {domain}: {', '.join(invalid_vars)}"
                
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return True, "Validation error"
    
    def _convert_to_structured_result(self, legacy_result: Dict[str, Any]) -> AnnotationResult:
        """Convert legacy result to structured AnnotationResult"""
        # Parse annotation to extract mappings
        annotation_text = legacy_result.get('annotation', '')
        pattern = legacy_result.get('pattern', 'direct')
        domain = legacy_result.get('domain')
        
        mappings = []
        
        # Simple parsing - would be more sophisticated in production
        if '=' in annotation_text:
            for part in annotation_text.split(','):
                if '=' in part:
                    var, val = part.strip().split('=', 1)
                    mappings.append(SDTMMapping(
                        variable=var.strip(),
                        value=val.strip().strip('"'),
                        target="input" if "@VALUE" in val else "fixed"
                    ))
        
        return AnnotationResult(
            pattern=pattern,
            domain=domain,
            mappings=mappings,
            confidence=legacy_result.get('confidence', 0.0),
            validation_status="valid" if legacy_result.get('valid', False) else "invalid",
            validation_message=legacy_result.get('validation_message', '')
        )
    
    def _extract_test_codes(self) -> Dict[str, Dict]:
        """Extract test codes from KB for RAG"""
        test_codes = {}
        
        # Extract from controlled terms
        for ct_name, ct_values in self.kb_loader.all_controlled_terms.items():
            if 'TEST' in ct_name.upper() or 'CD' in ct_name.upper():
                for value in ct_values:
                    if isinstance(value, dict) and 'code' in value:
                        test_codes[value['code']] = {
                            'name': value.get('preferred_term', ''),
                            'definition': value.get('definition', ''),
                            'domain': ct_name.split('_')[0] if '_' in ct_name else 'UNKNOWN'
                        }
        
        return test_codes