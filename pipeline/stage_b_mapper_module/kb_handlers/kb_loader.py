"""Knowledge Base loading utilities"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class KBLoader:
    """Handles loading of knowledge base resources"""
    
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = self._find_kb_path(kb_path)
        self.proto_define = {}
        self.domains_by_class = {}
        self.class_definitions = {}
        self.patterns = {}
        self.controlled_terms = {}
        self.common_ct_terms = {}
        self.cdisc_ct = {}
        self.all_controlled_terms = {}
        self.domain_variables = {}
        
    def _find_kb_path(self, kb_path: Optional[str]) -> Optional[Path]:
        """Find the KB directory"""
        if kb_path:
            return Path(kb_path)
            
        # Try common locations
        possible_kb_paths = [
            Path(__file__).parent.parent.parent.parent / "kb" / "sdtmig_v3_4_complete",
            Path(__file__).parent.parent.parent.parent / "kb" / "cdisc_integrated",
            Path(__file__).parent.parent / "kb"
        ]
        
        for kb_dir in possible_kb_paths:
            if kb_dir.exists():
                logger.info(f"Found KB at: {kb_dir}")
                return kb_dir
                
        logger.warning("No KB directory found")
        return None
    
    def load_proto_define(self, proto_define_path: Optional[str] = None) -> Dict:
        """Load proto_define.json"""
        if proto_define_path:
            proto_path = Path(proto_define_path)
        else:
            # Try common locations
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / "kb" / "sdtmig_v3_4_complete" / "proto_define.json",
                Path(__file__).parent.parent / "kb" / "proto_define.json",
                Path(__file__).parent / "proto_define.json"
            ]
            proto_path = None
            for path in possible_paths:
                if path.exists():
                    proto_path = path
                    break
                    
        if proto_path and proto_path.exists():
            logger.info(f"Loading proto_define from: {proto_path}")
            with open(proto_path, 'r') as f:
                self.proto_define = json.load(f)
        else:
            logger.error(f"proto_define.json not found")
            self.proto_define = {"datasets": {}}
            
        return self.proto_define
    
    def load_kb_resources(self) -> None:
        """Load all KB resources"""
        if not self.kb_path:
            return
            
        # Load domains_by_class.json
        domains_path = self.kb_path / "domains_by_class.json"
        if domains_path.exists():
            with open(domains_path, 'r') as f:
                self.domains_by_class = json.load(f)
            logger.info(f"Loaded {len(self.domains_by_class)} domain classes")
        
        # Load class_definitions.json
        class_def_path = self.kb_path / "class_definitions.json"
        if class_def_path.exists():
            with open(class_def_path, 'r') as f:
                class_data = json.load(f)
                self.class_definitions = class_data.get('sdtm_classes', {})
            logger.info(f"Loaded {len(self.class_definitions)} class definitions")
        
        # Load pattern_definitions.json
        pattern_path = self.kb_path / "pattern_definitions.json"
        if pattern_path.exists():
            with open(pattern_path, 'r') as f:
                self.patterns = json.load(f)
            logger.info(f"Loaded pattern definitions")
        
        # Load controlled terminology
        self._load_controlled_terminology()
        
        # Load variables_all.json
        self._load_variables()
    
    def _load_controlled_terminology(self):
        """Load controlled terminology from various sources"""
        # Try cdisc_ct.json first
        ct_path = self.kb_path / "cdisc_ct.json"
        if ct_path.exists():
            with open(ct_path, 'r') as f:
                self.cdisc_ct = json.load(f)
            logger.info(f"Loaded cdisc_ct.json")
        
        # Load common_ct_terms_complete.json
        ct_path = self.kb_path / "common_ct_terms_complete.json"
        if not ct_path.exists():
            ct_path = self.kb_path / "common_ct_terms.json"
            
        if ct_path.exists():
            with open(ct_path, 'r') as f:
                self.common_ct_terms = json.load(f)
            logger.info(f"Loaded common CT terms")
        
        # Load cdisc_all_controlled_terms.json
        all_ct_path = self.kb_path / "cdisc_all_controlled_terms.json"
        if all_ct_path.exists():
            with open(all_ct_path, 'r') as f:
                self.all_controlled_terms = json.load(f)
            logger.info(f"Loaded all controlled terms")
    
    def _load_variables(self):
        """Load variables from variables_all.json"""
        variables_path = self.kb_path / "variables_all.json"
        if variables_path.exists():
            with open(variables_path, 'r') as f:
                variables_data = json.load(f)

            # Support both dict and list formats
            if isinstance(variables_data, dict):
                for var_name, var_info in variables_data.items():
                    domains = var_info.get('domains', [])
                    for domain in domains:
                        if domain not in self.domain_variables:
                            self.domain_variables[domain] = {}
                        self.domain_variables[domain][var_name] = var_info
            elif isinstance(variables_data, list):
                for rec in variables_data:
                    domain = rec.get('domain') or rec.get('DOMAIN')
                    var_name = rec.get('name') or rec.get('VAR_NAME') or rec.get('name_short')
                    if not domain or not var_name:
                        continue
                    if domain not in self.domain_variables:
                        self.domain_variables[domain] = {}
                    self.domain_variables[domain][var_name] = rec

            logger.info(f"Loaded variables for {len(self.domain_variables)} domains")
