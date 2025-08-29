#!/usr/bin/env python3
"""
Enhanced SDTM Validator with comprehensive variable-by-variable validation
Validates roles, relationships, controlled terms, and pattern requirements
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    
    def to_tuple(self) -> Tuple[bool, str]:
        """Convert to simple tuple for backward compatibility"""
        if self.errors:
            return False, "; ".join(self.errors)
        elif self.warnings:
            return True, "Warnings: " + "; ".join(self.warnings)
        else:
            return True, "Valid"


class EnhancedSDTMValidator:
    """Enhanced validator that checks roles, relationships, and controlled terms"""
    
    def __init__(self, proto_define_path: str, ct_path: Optional[str] = None):
        """Initialize with proto_define and controlled terminology"""
        # Load proto_define
        with open(proto_define_path, 'r') as f:
            self.proto_define = json.load(f)
        
        # Load controlled terminology if provided
        self.ct_data = {}
        if ct_path and Path(ct_path).exists():
            with open(ct_path, 'r') as f:
                ct_raw = json.load(f)
                # Build codelist lookup
                for cl_info in ct_raw.get("codelists", []):
                    codelist = cl_info.get("codelist", {})
                    cl_code = codelist.get("ncitCode")
                    if cl_code:
                        self.ct_data[cl_code] = {
                            "name": codelist.get("shortName"),
                            "extensible": codelist.get("extensible", "false") == "true",
                            "terms": {
                                term["submissionValue"]: term 
                                for term in cl_info.get("terms", [])
                            }
                        }
        
        # Pattern requirements
        self.pattern_requirements = {
            "findings": {
                "required_roles": {
                    "TESTCD": "Topic",
                    "ORRES": "Result Qualifier",
                    "ORRESU": "Variable Qualifier"
                },
                "required_vars": ["TESTCD", "ORRES"],
                "optional_vars": ["ORRESU", "TEST", "STRESC", "STRESN"],
                "structure": "{domain}ORRES [/ {domain}ORRESU] when {domain}TESTCD = {value}"
            },
            "plain": {
                "allowed_roles": ["Topic", "Result Qualifier", "Grouping Qualifier", "Synonym Qualifier"],
                "forbidden_roles": ["Identifier"],
                "structure": "{variable} or {var1} / {var2}"
            },
            "conditional": {
                "required_roles": {
                    "TESTCD": "Topic",
                    "ORRES": "Result Qualifier"
                },
                "structure": "{variable} when {condition}"
            },
            "supplemental": {
                "structure": "{QNAM} in SUPP{domain}"
            },
            "not_submitted": {
                "structure": "[NOT SUBMITTED]"
            }
        }
        
        # Build lookup structures
        self._build_lookups()
    
    def _build_lookups(self):
        """Build efficient lookup structures"""
        # Variable to codelist mapping
        self.var_codelists = {}
        for domain, vars_dict in self.proto_define.get("variables", {}).items():
            for var_name, var_info in vars_dict.items():
                if "codelist_ref" in var_info:
                    cl_ref = var_info["codelist_ref"]
                    if isinstance(cl_ref, dict):
                        cl_code = cl_ref.get("code")
                        if cl_code:
                            self.var_codelists[f"{domain}.{var_name}"] = cl_code
    
    def validate_annotation(self, annotation: str, domain: str = None) -> ValidationResult:
        """Main validation entry point with comprehensive checks"""
        errors = []
        warnings = []
        info = {}
        
        # Special cases
        if annotation == "[NOT SUBMITTED]":
            return ValidationResult(True, [], [], {"pattern": "not_submitted"})
        
        # Detect pattern
        pattern = self._detect_pattern(annotation)
        info["pattern"] = pattern
        
        # Pattern-specific validation
        if pattern == "findings":
            return self._validate_findings_pattern(annotation, domain)
        elif pattern == "plain":
            return self._validate_plain_pattern(annotation, domain)
        elif pattern == "conditional":
            return self._validate_conditional_pattern(annotation, domain)
        elif pattern == "supplemental":
            return self._validate_supplemental_pattern(annotation, domain)
        else:
            errors.append(f"Unknown annotation pattern: {annotation}")
            return ValidationResult(False, errors, warnings, info)
    
    def _detect_pattern(self, annotation: str) -> str:
        """Detect which pattern the annotation uses"""
        if " when " in annotation:
            # Check if it's IE domain conditional pattern
            if "IEORRES when IETESTCD" in annotation:
                return "conditional"
            # Check if it's findings pattern (has ORRES with optional ORRESU before 'when')
            elif re.match(r'^[A-Z]+ORRES(?:\s*/\s*[A-Z]+ORRESU)?\s+when\s+[A-Z]+TESTCD\s*=', annotation):
                return "findings"
            else:
                return "conditional"
        elif " in SUPP" in annotation:
            return "supplemental"
        elif " / " in annotation and " when " not in annotation:
            return "plain"
        elif annotation.isupper() and len(annotation.split()) == 1:
            return "plain"
        else:
            return "unknown"
    
    def _validate_findings_pattern(self, annotation: str, domain: str) -> ValidationResult:
        """Validate findings pattern with comprehensive checks"""
        errors = []
        warnings = []
        info = {"pattern": "findings"}
        
        if not domain:
            errors.append("Domain required for findings pattern validation")
            return ValidationResult(False, errors, warnings, info)
        
        # Parse findings annotation
        pattern = r'^([A-Z]+ORRES)(?:\s*/\s*([A-Z]+ORRESU))?\s+when\s+([A-Z]+TESTCD)\s*=\s*([A-Z0-9_]+)$'
        match = re.match(pattern, annotation)
        
        if not match:
            errors.append(f"Invalid findings pattern format. Expected: {{domain}}ORRES [/ {{domain}}ORRESU] when {{domain}}TESTCD = {{value}}")
            return ValidationResult(False, errors, warnings, info)
        
        orres_var = match.group(1)
        orresu_var = match.group(2)
        testcd_var = match.group(3)
        testcd_value = match.group(4)
        
        info["variables"] = {
            "orres": orres_var,
            "orresu": orresu_var,
            "testcd": testcd_var,
            "testcd_value": testcd_value
        }
        
        # Check variable existence and roles
        domain_vars = self.proto_define.get("variables", {}).get(domain, {})
        
        # Validate ORRES
        if orres_var not in domain_vars:
            errors.append(f"Variable {orres_var} not found in domain {domain}")
        else:
            orres_info = domain_vars[orres_var]
            if orres_info.get("role") != "Result Qualifier":
                errors.append(f"{orres_var} has role '{orres_info.get('role')}' but should be 'Result Qualifier' for findings pattern")
            if orres_info.get("core") not in ["Req", "Exp"]:
                warnings.append(f"{orres_var} has core '{orres_info.get('core')}' - expected Required or Expected")
        
        # Validate ORRESU if present
        if orresu_var:
            if orresu_var not in domain_vars:
                errors.append(f"Variable {orresu_var} not found in domain {domain}")
            else:
                orresu_info = domain_vars[orresu_var]
                if orresu_info.get("role") != "Variable Qualifier":
                    errors.append(f"{orresu_var} has role '{orresu_info.get('role')}' but should be 'Variable Qualifier'")
        
        # Validate TESTCD
        if testcd_var not in domain_vars:
            errors.append(f"Variable {testcd_var} not found in domain {domain}")
        else:
            testcd_info = domain_vars[testcd_var]
            if testcd_info.get("role") != "Topic":
                errors.append(f"{testcd_var} has role '{testcd_info.get('role')}' but should be 'Topic' for findings pattern")
            if testcd_info.get("core") != "Req":
                warnings.append(f"{testcd_var} has core '{testcd_info.get('core')}' - should be Required")
        
        # Validate test code value against controlled terms if available
        testcd_codelist = self.var_codelists.get(f"{domain}.{testcd_var}")
        if testcd_codelist and testcd_codelist in self.ct_data:
            ct_info = self.ct_data[testcd_codelist]
            if testcd_value not in ct_info["terms"]:
                if ct_info["extensible"]:
                    warnings.append(f"Test code '{testcd_value}' not in standard codelist {ct_info['name']} (extensible)")
                else:
                    errors.append(f"Test code '{testcd_value}' not in codelist {ct_info['name']} (non-extensible)")
            else:
                info["testcd_term"] = ct_info["terms"][testcd_value]
        
        # Check for corresponding TEST variable
        test_var = f"{domain}TEST"
        if test_var not in domain_vars:
            warnings.append(f"Missing {test_var} variable - typically paired with {testcd_var}")
        
        return ValidationResult(len(errors) == 0, errors, warnings, info)
    
    def _validate_plain_pattern(self, annotation: str, domain: str) -> ValidationResult:
        """Validate plain pattern variables"""
        errors = []
        warnings = []
        info = {"pattern": "plain"}
        
        if not domain:
            warnings.append("Domain not provided for plain pattern validation")
            return ValidationResult(True, errors, warnings, info)
        
        # Parse variables
        variables = [v.strip() for v in annotation.split(" / ")]
        info["variables"] = variables
        
        domain_vars = self.proto_define.get("variables", {}).get(domain, {})
        
        for var in variables:
            if var not in domain_vars:
                errors.append(f"Variable {var} not found in domain {domain}")
            else:
                var_info = domain_vars[var]
                role = var_info.get("role")
                
                # Check forbidden roles
                if role == "Identifier":
                    errors.append(f"{var} has Identifier role - should not be used in plain pattern")
                
                # Check if variable has controlled terms
                var_key = f"{domain}.{var}"
                if var_key in self.var_codelists:
                    cl_code = self.var_codelists[var_key]
                    if cl_code in self.ct_data:
                        info[f"{var}_codelist"] = self.ct_data[cl_code]["name"]
        
        return ValidationResult(len(errors) == 0, errors, warnings, info)
    
    def _validate_conditional_pattern(self, annotation: str, domain: str) -> ValidationResult:
        """Validate conditional pattern (e.g., IE domain)"""
        errors = []
        warnings = []
        info = {"pattern": "conditional"}
        
        if not domain:
            errors.append("Domain required for conditional pattern validation")
            return ValidationResult(False, errors, warnings, info)
        
        # Special handling for IE domain
        if domain == "IE":
            # Expected: IEORRES when IETESTCD = {code}
            pattern = r'^(IEORRES)\s+when\s+(IETESTCD)\s*=\s*([A-Z0-9]+)$'
            match = re.match(pattern, annotation)
            
            if not match:
                errors.append("IE domain should use format: IEORRES when IETESTCD = {criterion_code}")
                return ValidationResult(False, errors, warnings, info)
            
            orres_var = match.group(1)
            testcd_var = match.group(2)
            criterion_code = match.group(3)
            
            info["variables"] = {
                "orres": orres_var,
                "testcd": testcd_var,
                "criterion_code": criterion_code
            }
            
            domain_vars = self.proto_define.get("variables", {}).get(domain, {})
            
            # Validate IEORRES
            if orres_var not in domain_vars:
                errors.append(f"{orres_var} not found in domain {domain}")
            else:
                orres_info = domain_vars[orres_var]
                if orres_info.get("role") != "Result Qualifier":
                    errors.append(f"{orres_var} should have 'Result Qualifier' role for IE domain")
                
                # Check codelist for IEORRES (should be Y/N/U/NA)
                orres_codelist = self.var_codelists.get(f"{domain}.{orres_var}")
                if orres_codelist == "C66742":  # No Yes Response codelist
                    info["orres_codelist"] = "No Yes Response (Y/N/U/NA)"
                else:
                    warnings.append(f"{orres_var} should use 'No Yes Response' codelist")
            
            # Validate IETESTCD
            if testcd_var not in domain_vars:
                errors.append(f"{testcd_var} not found in domain {domain}")
            else:
                testcd_info = domain_vars[testcd_var]
                if testcd_info.get("role") != "Topic":
                    errors.append(f"{testcd_var} should have 'Topic' role")
            
            # Validate criterion code format
            if not re.match(r'^(INC|EXC?)\d+$', criterion_code):
                warnings.append(f"Criterion code '{criterion_code}' doesn't follow standard pattern (INC# or EX#)")
            
            # Check for IETEST variable
            if "IETEST" not in domain_vars:
                warnings.append("IETEST variable missing - should contain criterion description")
            
            # Check for IECAT variable
            if "IECAT" in domain_vars:
                iecat_info = domain_vars["IECAT"]
                if iecat_info.get("role") != "Grouping Qualifier":
                    warnings.append("IECAT should have 'Grouping Qualifier' role")
        
        return ValidationResult(len(errors) == 0, errors, warnings, info)
    
    def _validate_supplemental_pattern(self, annotation: str, domain: str) -> ValidationResult:
        """Validate supplemental pattern"""
        errors = []
        warnings = []
        info = {"pattern": "supplemental"}
        
        # Parse supplemental annotation
        pattern = r'^([A-Z]+)\s+in\s+SUPP([A-Z]+)$'
        match = re.match(pattern, annotation)
        
        if not match:
            errors.append("Invalid supplemental pattern. Expected: {QNAM} in SUPP{DOMAIN}")
            return ValidationResult(False, errors, warnings, info)
        
        qnam = match.group(1)
        supp_domain = match.group(2)
        
        info["qnam"] = qnam
        info["supp_domain"] = supp_domain
        
        # Validate domain consistency
        if domain and supp_domain != domain:
            errors.append(f"SUPP domain mismatch: SUPP{supp_domain} but validating for domain {domain}")
        
        # Common QNAM patterns
        common_qnams = ["AEOTH", "CMOTH", "MHOTH", "AEOTHSP", "CMOTHSP"]
        if qnam not in common_qnams:
            info["qnam_type"] = "sponsor-defined"
        else:
            info["qnam_type"] = "standard"
        
        return ValidationResult(len(errors) == 0, errors, warnings, info)
    
    def validate_variable_relationships(self, domain: str, variables: List[str]) -> ValidationResult:
        """Validate relationships between variables"""
        errors = []
        warnings = []
        info = {}
        
        domain_vars = self.proto_define.get("variables", {}).get(domain, {})
        
        # Check for required variable pairs
        if f"{domain}TESTCD" in variables and f"{domain}TEST" not in variables:
            warnings.append(f"{domain}TEST should accompany {domain}TESTCD")
        
        if f"{domain}ORRES" in variables:
            # Check for standardized results
            if f"{domain}STRESC" not in variables:
                warnings.append(f"{domain}STRESC (standardized character result) typically accompanies {domain}ORRES")
            if f"{domain}STRESN" not in domain_vars:
                info["numeric_results"] = "Not applicable - domain doesn't support numeric results"
        
        return ValidationResult(len(errors) == 0, errors, warnings, info)


def main():
    """Test the enhanced validator"""
    import sys
    
    # Initialize validator
    validator = EnhancedSDTMValidator(
        proto_define_path="kb/sdtmig_v3_4_complete/proto_define.json",
        ct_path="kb/sdtmig_v3_4_complete/cdisc_ct.json"
    )
    
    # Test cases
    test_cases = [
        ("IEORRES when IETESTCD = INC1", "IE"),
        ("VSORRES / VSORRESU when VSTESTCD = SYSBP", "VS"),
        ("AETERM", "AE"),
        ("AEOTH in SUPPAE", "AE"),
        ("[NOT SUBMITTED]", None)
    ]
    
    for annotation, domain in test_cases:
        print(f"\nValidating: {annotation} (Domain: {domain})")
        print("-" * 60)
        
        result = validator.validate_annotation(annotation, domain)
        
        print(f"Valid: {result.valid}")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        print(f"Info: {json.dumps(result.info, indent=2)}")


if __name__ == "__main__":
    main()