"""Build indices from CDISC KB data"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

from .schemas import (
    DomainDocument, VariableDocument, CTDocument,
    CTRelationship, UnitMapping
)

logger = logging.getLogger(__name__)


class CDISCIndexBuilder:
    """Build three separate indices for domains, variables, and CT"""
    
    def __init__(self, kb_path: Path):
        self.kb_path = Path(kb_path)
        self.domain_docs: List[DomainDocument] = []
        self.variable_docs: List[VariableDocument] = []
        self.ct_docs: List[CTDocument] = []
        self.ct_relationships: List[CTRelationship] = []
        self.unit_mappings: List[UnitMapping] = []
        
        # Paired codelists (TEST/TESTCD patterns)
        self.paired_codelists = {}
        
    def build_all_indices(self) -> Tuple[List[DomainDocument], List[VariableDocument], List[CTDocument]]:
        """Build all three indices from KB data"""
        logger.info("Building CDISC indices...")
        
        # Load proto_define for domain and variable info
        proto_define = self._load_proto_define()
        
        # Build domain index
        self._build_domain_index(proto_define)
        
        # Build variable index
        self._build_variable_index(proto_define)
        
        # Build CT index
        self._build_ct_index()
        
        # Build relationships
        self._build_relationships()
        
        logger.info(f"Built indices: {len(self.domain_docs)} domains, "
                   f"{len(self.variable_docs)} variables, {len(self.ct_docs)} CT terms")
        
        return self.domain_docs, self.variable_docs, self.ct_docs
    
    def _load_proto_define(self) -> Dict:
        """Load proto_define.json"""
        proto_path = self.kb_path / "proto_define.json"
        if proto_path.exists():
            with open(proto_path, 'r') as f:
                return json.load(f)
        return {"datasets": {}}
    
    def _build_domain_index(self, proto_define: Dict):
        """Build domain documents"""
        # Load additional domain info
        domains_by_class = self._load_domains_by_class()
        class_definitions = self._load_class_definitions()
        
        # Create domain documents
        for domain_code, domain_info in proto_define.get("datasets", {}).items():
            # Find domain class
            domain_class = "SPECIAL PURPOSE"  # default
            for class_name, domains in domains_by_class.items():
                if domain_code in domains:
                    domain_class = class_name
                    break
            
            doc = DomainDocument(
                domain_code=domain_code,
                domain_name=domain_info.get("name", domain_code),
                domain_class=domain_class,
                definition=domain_info.get("description", "")
            )
            self.domain_docs.append(doc)
    
    def _build_variable_index(self, proto_define: Dict):
        """Build variable documents"""
        # Also load variables_all.json if available
        variables_all = self._load_variables_all()
        
        # Track which variables we've added
        added_vars = set()
        
        # First, process proto_define variables
        for domain_code, domain_info in proto_define.get("datasets", {}).items():
            for var_name, var_info in domain_info.get("variables", {}).items():
                var_key = f"{domain_code}.{var_name}"
                if var_key in added_vars:
                    continue
                    
                doc = VariableDocument(
                    var_name=var_name,
                    label=var_info.get("label", ""),
                    role=var_info.get("role", ""),
                    domain_code=domain_code,
                    definition=var_info.get("description", var_info.get("label", "")),
                    codelist_name=var_info.get("codelist"),
                    codelist_code=var_info.get("codelist_code"),
                    data_type=var_info.get("type"),
                    core=var_info.get("core")
                )
                self.variable_docs.append(doc)
                added_vars.add(var_key)
        
        # Then add from variables_all.json
        for var_name, var_info in variables_all.items():
            for domain in var_info.get("domains", []):
                var_key = f"{domain}.{var_name}"
                if var_key in added_vars:
                    continue
                    
                # Extract codelist info properly
                codelist_info = var_info.get("codelist")
                codelist_name = None
                if isinstance(codelist_info, dict):
                    codelist_name = codelist_info.get("code") or codelist_info.get("name")
                elif isinstance(codelist_info, str):
                    codelist_name = codelist_info
                    
                doc = VariableDocument(
                    var_name=var_name,
                    label=var_info.get("label", ""),
                    role=var_info.get("role", ""),
                    domain_code=domain,
                    definition=var_info.get("definition", var_info.get("label", "")),
                    codelist_name=codelist_name,
                    data_type=var_info.get("type"),
                    core=var_info.get("core")
                )
                self.variable_docs.append(doc)
                added_vars.add(var_key)
    
    def _build_ct_index(self):
        """Build CT documents from various sources"""
        # Load main CDISC CT file
        cdisc_ct_path = self.kb_path / "cdisc_ct.json"
        if cdisc_ct_path.exists():
            with open(cdisc_ct_path, 'r') as f:
                ct_data = json.load(f)
                
                # Process codelists array
                for cl_entry in ct_data.get('codelists', []):
                    codelist_info = cl_entry.get('codelist', {})
                    codelist_code = codelist_info.get('ncitCode', '')
                    codelist_name = codelist_info.get('shortName', '')
                    
                    # Process terms
                    for term in cl_entry.get('terms', []):
                        doc = CTDocument(
                            codelist_name=codelist_code,  # Use NCI code as primary ID
                            codelist_code=codelist_code,
                            submission_value=term.get('submissionValue', ''),
                            preferred_term=term.get('preferredTerm', ''),
                            definition=term.get('definition', ''),
                            synonyms=term.get('synonyms', []),
                            nci_c_code=term.get('ncitCode')
                        )
                        # Create embedding text that includes codelist name for better matching
                        doc.embedding_text = f"CT Term: {doc.submission_value} ({doc.preferred_term}). " \
                                           f"Codelist: {codelist_name} ({codelist_code}). " \
                                           f"Definition: {doc.definition}"
                        self.ct_docs.append(doc)
        
        # Also load from other sources
        ct_sources = [
            self.kb_path / "cdisc_all_controlled_terms.json",
            self.kb_path / "common_ct_terms_complete.json"
        ]
        
        all_ct_data = {}
        
        for ct_path in ct_sources:
            if ct_path.exists():
                with open(ct_path, 'r') as f:
                    ct_data = json.load(f)
                    
                    # Handle different formats
                    if isinstance(ct_data, dict):
                        # Check if it has metadata wrapper
                        if "metadata" in ct_data and "codelists" in ct_data:
                            # Extract codelists from wrapper
                            ct_data = ct_data["codelists"]
                        
                        for codelist_name, terms in ct_data.items():
                            if codelist_name == "metadata":
                                continue
                            if codelist_name not in all_ct_data:
                                all_ct_data[codelist_name] = []
                            if isinstance(terms, list):
                                all_ct_data[codelist_name].extend(terms)
        
        # Build CT documents from other sources
        for codelist_name, terms in all_ct_data.items():
            for term in terms:
                if isinstance(term, dict):
                    doc = CTDocument(
                        codelist_name=codelist_name,
                        codelist_code=term.get("codelist_code", codelist_name),
                        submission_value=term.get("code", term.get("submission_value", "")),
                        preferred_term=term.get("preferred_term", term.get("decode", "")),
                        definition=term.get("definition", ""),
                        synonyms=term.get("synonyms", []),
                        nci_c_code=term.get("nci_c_code", term.get("c_code"))
                    )
                    # Add embedding text
                    doc.embedding_text = f"CT Term: {doc.submission_value} ({doc.preferred_term}). " \
                                       f"Codelist: {codelist_name}. Definition: {doc.definition}"
                    self.ct_docs.append(doc)
    
    def _build_relationships(self):
        """Build variable-codelist relationships"""
        # Map variables to their codelists
        for var_doc in self.variable_docs:
            if var_doc.codelist_name:
                rel = CTRelationship(
                    variable_name=var_doc.var_name,
                    domain_code=var_doc.domain_code,
                    codelist_name=var_doc.codelist_name,
                    codelist_code=var_doc.codelist_code or var_doc.codelist_name
                )
                self.ct_relationships.append(rel)
                
        # Also check proto_define for additional relationships
        proto_define = self._load_proto_define()
        for domain_code, domain_info in proto_define.get("datasets", {}).items():
            for var_name, var_info in domain_info.get("variables", {}).items():
                codelist = var_info.get("codelist")
                if codelist and not any(r.variable_name == var_name and r.domain_code == domain_code 
                                      for r in self.ct_relationships):
                    rel = CTRelationship(
                        variable_name=var_name,
                        domain_code=domain_code,
                        codelist_name=codelist,
                        codelist_code=var_info.get("codelist_code", codelist)
                    )
                    self.ct_relationships.append(rel)
    
    def _load_domains_by_class(self) -> Dict:
        """Load domains_by_class.json"""
        path = self.kb_path / "domains_by_class.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_class_definitions(self) -> Dict:
        """Load class_definitions.json"""
        path = self.kb_path / "class_definitions.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("sdtm_classes", {})
        return {}
    
    def _load_variables_all(self) -> Dict:
        """Load variables_all.json"""
        path = self.kb_path / "variables_all.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    # Convert list to dict format grouped by variable name
                    variables_dict = {}
                    for var in data:
                        var_name = var.get("name", "")
                        if var_name not in variables_dict:
                            variables_dict[var_name] = {
                                "name": var_name,
                                "label": var.get("label", ""),
                                "type": var.get("type", ""),
                                "role": var.get("role", ""),
                                "core": var.get("core", ""),
                                "definition": var.get("description", var.get("label", "")),
                                "codelist": var.get("codelist"),
                                "domains": []
                            }
                        variables_dict[var_name]["domains"].append(var.get("domain", ""))
                    return variables_dict
                else:
                    return data
        return {}
    
    def get_ct_relationships(self) -> Dict[Tuple[str, str], List[str]]:
        """Get CT relationships as lookup dict"""
        relationships = defaultdict(list)
        for rel in self.ct_relationships:
            key = (rel.variable_name, rel.domain_code)
            relationships[key].append(rel.codelist_name)
        return dict(relationships)