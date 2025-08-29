#!/usr/bin/env python3
"""
Proto-Define Validator - Uses Proto-Define to validate SDTM annotations
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class ProtoDefineValidator:
    def __init__(self, proto_define_path: str):
        """Initialize validator with proto-define JSON"""
        with open(proto_define_path, 'r') as f:
            self.proto_define = json.load(f)
        
        # Cache frequently accessed data
        self.datasets = self.proto_define.get("datasets", {})
        self.variables = self.proto_define.get("variables", {})
        self.codelists = self.proto_define.get("codelists", {})
        self.vlm = self.proto_define.get("value_level_metadata", {})
        self.rules = self.proto_define.get("annotation_rules", {})
        
        # Build quick lookup structures
        self._build_lookups()
    
    def _build_lookups(self):
        """Build efficient lookup structures"""
        # Variable to codelist mapping
        self.var_to_codelist = {}
        for domain, vars_dict in self.variables.items():
            for var_name, var_info in vars_dict.items():
                if "codelist_ref" in var_info:
                    cl_ref = var_info["codelist_ref"]
                    if isinstance(cl_ref, dict):
                        cl_key = cl_ref.get("code") or cl_ref.get("name")
                    else:
                        cl_key = cl_ref
                    if cl_key:
                        self.var_to_codelist[f"{domain}.{var_name}"] = cl_key
        
        # Codelist value lookup
        self.codelist_values = {}
        for cl_name, cl_data in self.codelists.items():
            values = set()
            for term in cl_data.get("terms", []):
                values.add(term.get("submission_value", ""))
            self.codelist_values[cl_name] = values
    
    def validate_domain(self, domain: str) -> Tuple[bool, str]:
        """Validate that domain exists in proto-define"""
        if domain in self.datasets:
            return True, ""
        return False, f"Domain '{domain}' not found in proto-define"
    
    def validate_variable(self, domain: str, variable: str) -> Tuple[bool, str]:
        """Validate that variable exists for the given domain"""
        if domain not in self.variables:
            return False, f"Domain '{domain}' has no variables defined"
        
        if variable not in self.variables[domain]:
            return False, f"Variable '{variable}' not found in domain '{domain}'"
        
        return True, ""
    
    def validate_codelist_value(self, domain: str, variable: str, value: str) -> Tuple[bool, str]:
        """Validate that value is in the variable's codelist"""
        var_key = f"{domain}.{variable}"
        
        # Check if variable has a codelist
        if var_key not in self.var_to_codelist:
            # No codelist constraint - value is valid
            return True, ""
        
        cl_key = self.var_to_codelist[var_key]
        
        # Look up codelist values
        if cl_key in self.codelist_values:
            valid_values = self.codelist_values[cl_key]
            if value in valid_values:
                return True, ""
            else:
                return False, f"Value '{value}' not in codelist '{cl_key}' for {domain}.{variable}"
        
        # Codelist not found - warn but don't fail
        return True, f"Warning: Codelist '{cl_key}' not found"
    
    def validate_findings_annotation(self, annotation: str, domain: str) -> Tuple[bool, str]:
        """Validate Findings domain annotation format"""
        # Expected pattern: VSORRES [/ VSORRESU] when VSTESTCD = VALUE
        pattern = r'^([A-Z]+ORRES)(?:\s*/\s*([A-Z]+ORRESU))?\s+when\s+([A-Z]+TESTCD)\s*=\s*([A-Z0-9_]+)$'
        match = re.match(pattern, annotation)
        
        if not match:
            return False, f"Findings annotation doesn't match required pattern: {{VAR}}ORRES [/ {{VAR}}ORRESU] when {{VAR}}TESTCD = {{VALUE}}"
        
        orres_var = match.group(1)
        orresu_var = match.group(2)
        testcd_var = match.group(3)
        testcd_value = match.group(4)
        
        # Validate variables exist
        valid, msg = self.validate_variable(domain, orres_var)
        if not valid:
            return False, msg
        
        valid, msg = self.validate_variable(domain, testcd_var)
        if not valid:
            return False, msg
        
        if orresu_var:
            valid, msg = self.validate_variable(domain, orresu_var)
            if not valid:
                return False, msg
        
        # Validate TESTCD value
        valid, msg = self.validate_codelist_value(domain, testcd_var, testcd_value)
        if not valid:
            return False, msg
        
        # Check VLM if available
        if domain in self.vlm:
            vlm_found = False
            for vlm_entry in self.vlm[domain]:
                where_clause = vlm_entry.get("where", {})
                if testcd_var in where_clause and where_clause[testcd_var] == testcd_value:
                    vlm_found = True
                    break
            
            if not vlm_found:
                return True, f"Warning: No VLM entry found for {testcd_var}={testcd_value}"
        
        return True, ""
    
    def validate_supplemental_annotation(self, annotation: str, domain: str) -> Tuple[bool, str]:
        """Validate supplemental qualifier annotation format"""
        # Expected pattern: QNAM in SUPP{DOMAIN}
        pattern = r'^([A-Z0-9_]+)\s+in\s+SUPP([A-Z]+)$'
        match = re.match(pattern, annotation)
        
        if not match:
            return False, "Supplemental annotation doesn't match pattern: QNAM in SUPP{DOMAIN}"
        
        qnam = match.group(1)
        supp_domain = match.group(2)
        
        if supp_domain != domain:
            return False, f"SUPP domain mismatch: expected SUPP{domain}, got SUPP{supp_domain}"
        
        # QNAM should be <= 8 characters per SDTM rules
        if len(qnam) > 8:
            return False, f"QNAM '{qnam}' exceeds 8 character limit"
        
        return True, ""
    
    def validate_annotation(self, annotation: str, domain: str = None) -> Tuple[bool, str]:
        """Validate a complete annotation string"""
        # Handle special cases
        if annotation == "[NOT SUBMITTED]":
            return True, ""
        
        # Parse multiple variables (separated by " / ")
        if " / " in annotation and " when " not in annotation:
            # Multiple simple variables
            variables = annotation.split(" / ")
            for var in variables:
                if domain:
                    valid, msg = self.validate_variable(domain, var.strip())
                    if not valid:
                        return False, msg
            return True, ""
        
        # Check if it's a Findings annotation
        if " when " in annotation and "TESTCD" in annotation:
            if not domain:
                return False, "Domain required for Findings validation"
            return self.validate_findings_annotation(annotation, domain)
        
        # Check if it's a supplemental annotation
        if " in SUPP" in annotation:
            if not domain:
                return False, "Domain required for supplemental validation"
            return self.validate_supplemental_annotation(annotation, domain)
        
        # Simple variable annotation
        if domain:
            return self.validate_variable(domain, annotation.strip())
        
        return False, "Unable to parse annotation format"
    
    def get_domain_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a domain"""
        return self.datasets.get(domain)
    
    def get_variable_info(self, domain: str, variable: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a variable"""
        if domain in self.variables:
            return self.variables[domain].get(variable)
        return None
    
    def get_valid_domains(self, domain_class: str = None) -> List[str]:
        """Get list of valid domains, optionally filtered by class"""
        if domain_class:
            return [d for d, info in self.datasets.items() 
                    if info.get("class") == domain_class]
        return list(self.datasets.keys())
    
    def get_valid_variables(self, domain: str, role: str = None) -> List[str]:
        """Get list of valid variables for a domain, optionally filtered by role"""
        if domain not in self.variables:
            return []
        
        if role:
            return [v for v, info in self.variables[domain].items() 
                    if info.get("role") == role]
        return list(self.variables[domain].keys())
    
    def get_codelist_values(self, codelist_key: str) -> List[str]:
        """Get valid values for a codelist"""
        if codelist_key in self.codelist_values:
            return list(self.codelist_values[codelist_key])
        return []
    
    def suggest_correction(self, value: str, valid_values: List[str], threshold: float = 0.8) -> Optional[str]:
        """Suggest a correction for an invalid value using fuzzy matching"""
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0
        
        for valid in valid_values:
            score = SequenceMatcher(None, value.upper(), valid.upper()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = valid
        
        return best_match


def main():
    """Example usage of the validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SDTM annotations using Proto-Define")
    parser.add_argument("--proto-define", required=True, help="Path to proto_define.json")
    parser.add_argument("--annotation", required=True, help="Annotation string to validate")
    parser.add_argument("--domain", help="Domain context for validation")
    
    args = parser.parse_args()
    
    validator = ProtoDefineValidator(args.proto_define)
    
    valid, message = validator.validate_annotation(args.annotation, args.domain)
    
    if valid:
        print(f"✓ Valid annotation: {args.annotation}")
        if message:  # Warning message
            print(f"  ⚠ {message}")
    else:
        print(f"✗ Invalid annotation: {args.annotation}")
        print(f"  Error: {message}")
        
        # Try to suggest corrections for simple variables
        if args.domain and " " not in args.annotation:
            valid_vars = validator.get_valid_variables(args.domain)
            suggestion = validator.suggest_correction(args.annotation, valid_vars)
            if suggestion:
                print(f"  Did you mean: {suggestion}?")


if __name__ == "__main__":
    main()