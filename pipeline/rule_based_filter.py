#!/usr/bin/env python3
"""
Rule-based filtering for SDTM mapper
Shows LLM only valid options based on pattern requirements
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

class RuleBasedFilter:
    """Filter options based on SDTM pattern rules"""
    
    def __init__(self, proto_define: Dict[str, Any]):
        self.proto_define = proto_define
        
        # Pattern-specific domain filtering rules
        self.pattern_domain_rules = {
            "findings": {
                "allowed_classes": ["Findings"],
                "required_variables": ["TESTCD", "ORRES"],
                "forbidden_domains": []
            },
            "conditional": {
                "allowed_classes": ["Findings", "Events"],  
                "preferred_domains": ["IE"],  # For inclusion/exclusion
                "required_variables": ["ORRES", "TESTCD"] 
            },
            "plain": {
                "allowed_classes": ["Events", "Interventions", "Special Purpose", "Findings"],
                "forbidden_variables": ["TESTCD", "ORRES", "ORRESU"],  # These require findings pattern
                "preferred_roles": ["Topic", "Result Qualifier", "Grouping Qualifier"]
            },
            "supplemental": {
                "allowed_classes": ["Relationship"],
                "domain_suffix": "SUPP"
            },
            "not_submitted": {
                "allowed_classes": [],  # No domain needed
            }
        }
        
        # Form/section to domain hints
        self.context_domain_hints = {
            "inclusion criteria": ["IE"],
            "exclusion criteria": ["IE"],
            "adverse events": ["AE"],
            "vital signs": ["VS"],
            "laboratory": ["LB"],
            "medical history": ["MH"],
            "concomitant medications": ["CM"],
            "demographics": ["DM"],
            "physical exam": ["PE"],
            "ecg": ["EG"],
            "disposition": ["DS"],
            "quality of life": ["QS", "RS", "FT"],
            "qol": ["QS", "RS", "FT"],
            "questionnaire": ["QS", "RS"],
            "edema": ["VS", "PE"],
            "circumference": ["VS"],
            "anticoagulation": ["CM", "EC", "EX"]
        }
        
        # Unit to domain mappings
        self.unit_domain_hints = {
            "cm": ["VS", "PE"],  # circumference measurements
            "mmhg": ["VS", "EG"],  # blood pressure
            "bpm": ["VS", "EG"],  # heart rate
            "kg": ["VS"],  # weight
            "g": ["VS", "LB"],
            "mg": ["LB", "EX", "EC"],
            "°c": ["VS"],  # temperature
            "°f": ["VS"],
            "%": ["LB", "VS", "EG"]
        }
        
        # Instrument to domain mappings
        self.instrument_domain_map = {
            "eq-5d": "QS",
            "eq5d": "QS",
            "villalta": "RS",  # SDTM v3.3+ moved from QS to RS
            "rvcss": "RS",
            "revised venous clinical severity": "RS",
            "phq-9": "QS",
            "gad-7": "QS",
            "sf-36": "QS",
            "moca": "QS",
            "mmse": "QS"
        }
    
    def filter_domains_for_pattern(self, pattern: str, sdtm_class: str, 
                                  field_data: Dict[str, Any]) -> List[Tuple[str, Dict, float]]:
        """
        Filter domains based on pattern rules and return with priority scores
        Returns: List of (domain_code, domain_info, priority_score)
        """
        all_domains = []
        rules = self.pattern_domain_rules.get(pattern, {})
        
        # Get all domains for the class
        for domain_code, domain_info in self.proto_define.get("datasets", {}).items():
            if domain_info.get("class") == sdtm_class:
                all_domains.append((domain_code, domain_info))
        
        filtered_domains = []
        
        for domain_code, domain_info in all_domains:
            priority = 0.5  # Base priority
            
            # Check if domain has required variables for pattern
            if "required_variables" in rules:
                domain_vars = self.proto_define.get("variables", {}).get(domain_code, {})
                has_required = all(
                    f"{domain_code}{var_suffix}" in domain_vars 
                    for var_suffix in rules["required_variables"]
                )
                if not has_required:
                    continue  # Skip domains without required variables
                priority += 0.2
            
            # Check forbidden domains
            if domain_code in rules.get("forbidden_domains", []):
                continue
            
            # Boost priority for preferred domains
            if domain_code in rules.get("preferred_domains", []):
                priority += 0.3
            
            # Context-based boosting
            form_section = f"{field_data.get('form_name', '')} {field_data.get('section', '')}".lower()
            for context_key, hint_domains in self.context_domain_hints.items():
                if context_key in form_section and domain_code in hint_domains:
                    priority += 0.4
                    break
            
            filtered_domains.append((domain_code, domain_info, priority))
        
        # Sort by priority
        filtered_domains.sort(key=lambda x: x[2], reverse=True)
        
        return filtered_domains
    
    def filter_variables_for_pattern(self, pattern: str, domain: str, 
                                   field_data: Dict[str, Any]) -> List[Tuple[str, Dict, float]]:
        """
        Filter variables based on pattern rules
        Returns: List of (variable_name, variable_info, priority_score)
        """
        domain_vars = self.proto_define.get("variables", {}).get(domain, {})
        rules = self.pattern_domain_rules.get(pattern, {})
        
        filtered_vars = []
        
        # Special handling for findings pattern
        if pattern == "findings":
            # Must include TESTCD, ORRES, and optionally ORRESU
            testcd = f"{domain}TESTCD"
            orres = f"{domain}ORRES"
            orresu = f"{domain}ORRESU"
            test = f"{domain}TEST"
            
            result = []
            if testcd in domain_vars:
                result.append((testcd, domain_vars[testcd], 1.0))
            if test in domain_vars:
                result.append((test, domain_vars[test], 0.9))
            if orres in domain_vars:
                result.append((orres, domain_vars[orres], 1.0))
            if orresu in domain_vars and field_data.get('has_units'):
                result.append((orresu, domain_vars[orresu], 0.8))
                
            return result
        
        # Special handling for conditional pattern (IE domain)
        if pattern == "conditional" and domain == "IE":
            # Only show IEORRES, IETESTCD, IETEST
            relevant_vars = ["IEORRES", "IETESTCD", "IETEST", "IECAT"]
            result = []
            for var in relevant_vars:
                if var in domain_vars:
                    priority = 1.0 if var in ["IEORRES", "IETESTCD"] else 0.7
                    result.append((var, domain_vars[var], priority))
            return result
        
        # For plain pattern, filter by role and avoid findings variables
        if pattern == "plain":
            forbidden_suffixes = ["TESTCD", "ORRES", "ORRESU", "STRESC", "STRESN"]
            preferred_roles = rules.get("preferred_roles", [])
            
            for var_name, var_info in domain_vars.items():
                # Skip identifier variables
                if var_info.get("role") == "Identifier":
                    continue
                    
                # Skip findings pattern variables
                if any(var_name.endswith(suffix) for suffix in forbidden_suffixes):
                    continue
                
                # Calculate priority
                priority = 0.5
                if var_info.get("role") in preferred_roles:
                    priority += 0.3
                if var_info.get("core") == "Req":
                    priority += 0.2
                elif var_info.get("core") == "Exp":
                    priority += 0.1
                
                filtered_vars.append((var_name, var_info, priority))
        
        # Sort by priority
        filtered_vars.sort(key=lambda x: x[2], reverse=True)
        
        return filtered_vars
    
    def infer_value_shape(self, field_data: Dict[str, Any]) -> str:
        """Infer the value shape/type from field data"""
        control_type = field_data.get('control_type', '').lower()
        label = field_data.get('label', '').lower()
        has_units = field_data.get('has_units', False)
        options = field_data.get('options', [])
        
        # Check for numeric with units
        if has_units or any(unit in label for unit in ['cm', 'mg', 'ml', 'mmhg', 'kg', '°c']):
            return 'NUM_UNIT'
        
        # Check for dates
        if control_type == 'date' or any(d in label for d in ['date', 'day', 'time']):
            return 'DATE'
        
        # Check for yes/no
        if options and set(opt.lower() for opt in options) == {'yes', 'no'}:
            return 'YN'
        
        # Check for ordinal scales
        if options:
            if len(options) == 5 and all(opt.isdigit() or opt in ['0','1','2','3','4','5'] for opt in options):
                return 'ORDINAL_5'
            elif len(options) == 4:
                return 'ORDINAL_4'
        
        # Check for numeric without units
        if control_type in ['number', 'integer', 'decimal']:
            return 'NUMERIC'
        
        # Default to text
        return 'TEXT'
    
    def gate_domains_hard(self, field_data: Dict[str, Any]) -> Optional[List[str]]:
        """Hard gate domains based on deterministic rules"""
        shape = self.infer_value_shape(field_data)
        form = (field_data.get('form_name', '') + ' ' + field_data.get('section', '')).lower()
        label = field_data.get('label', '').lower()
        
        # Check for known instruments first
        for instrument, domain in self.instrument_domain_map.items():
            if instrument in label or instrument in form:
                return [domain]
        
        # Check for IE criteria patterns
        import re
        if re.search(r'\((inc|ex)\d+\)', label) or 'inclusion' in form or 'exclusion' in form:
            return ['IE']
        
        # Units-based gating
        if shape == 'NUM_UNIT':
            # Check specific units
            for unit_pattern, domains in self.unit_domain_hints.items():
                if unit_pattern in label:
                    return domains
            # Default findings domains for numeric with units
            return ['VS', 'LB', 'EG', 'PE']
        
        # Form/section based hard gates
        if 'demograph' in form:
            return ['DM']
        if 'adverse event' in form or ' ae ' in form or 'aes' in form:
            return ['AE']
        if 'medical history' in form:
            return ['MH']
        if 'concomitant med' in form or 'conmed' in form:
            return ['CM']
        if 'disposition' in form:
            return ['DS']
        if 'anticoag' in form:
            return ['CM', 'EC', 'EX']
        
        # QOL/questionnaire gating
        if 'quality of life' in form or 'qol' in form or 'questionnaire' in form:
            return ['QS', 'RS', 'FT']
        
        # No hard gate - return None to use normal filtering
        return None
    
    def get_valid_testcodes(self, domain: str, field_label: str) -> List[Dict[str, Any]]:
        """Get valid test codes for a domain with intelligent filtering"""
        # Get test codes from proto_define or CT
        testcode_metadata = self.proto_define.get("testcode_metadata", {}).get(domain, [])
        
        if not testcode_metadata:
            return []
        
        # Filter based on field label
        label_lower = field_label.lower()
        filtered_codes = []
        
        # Enhanced keyword matching for common measurements
        keywords = {
            "systolic": ["SYSBP"],
            "diastolic": ["DIABP"],
            "pulse": ["PULSE", "HR"],
            "heart rate": ["HR", "PULSE"],
            "temperature": ["TEMP"],
            "weight": ["WEIGHT", "WT"],
            "height": ["HEIGHT", "HT"],
            "hemoglobin": ["HGB", "HB"],
            "glucose": ["GLUC", "GLUCOSE"],
            "circumference": ["CIRC"],
            "mid-calf": ["MIDCALF", "CALFCIRC"],
            "mobility": ["MOBILITY", "MOB"],
            "self-care": ["SELFCARE", "SELFCAR"],
            "pain": ["PAIN"],
            "cramps": ["CRAMP"],
            "edema": ["EDEMA"],
            "ulcer": ["ULCER"]
        }
        
        for code_info in testcode_metadata:
            code = code_info.get("code", "")
            test_label = code_info.get("label", "").lower()
            definition = code_info.get("definition", "").lower()
            
            # Calculate relevance score
            score = 0
            
            # Direct match in label
            if code.lower() in label_lower or test_label in label_lower:
                score += 1.0
            
            # Check each keyword
            for keyword, codes in keywords.items():
                if keyword in label_lower and code in codes:
                    score += 0.8
                elif keyword in label_lower and keyword in test_label:
                    score += 0.6
            
            # Partial word matching
            label_words = set(label_lower.split())
            test_words = set(test_label.split())
            overlap = len(label_words & test_words)
            if overlap > 0:
                score += overlap * 0.3
            
            if score > 0:
                filtered_codes.append({
                    "code": code,
                    "label": code_info.get("label"),
                    "definition": code_info.get("definition"),
                    "score": score
                })
        
        # Sort by relevance
        filtered_codes.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top matches
        return filtered_codes[:5]


def integrate_rule_filter(mapper_instance):
    """Integrate rule-based filtering into the mapper"""
    # Add the filter to the mapper instance
    mapper_instance.rule_filter = RuleBasedFilter(mapper_instance.proto_define)
    
    # Override domain selection method
    original_select_domain = mapper_instance._select_domain_with_class_constraint
    
    def filtered_select_domain(field_data, pattern, sdtm_class):
        if not mapper_instance.rule_filter or not sdtm_class:
            return original_select_domain(field_data, pattern, sdtm_class)
        
        # Get filtered domains
        filtered_domains = mapper_instance.rule_filter.filter_domains_for_pattern(
            pattern, sdtm_class, field_data
        )
        
        if not filtered_domains:
            return original_select_domain(field_data, pattern, sdtm_class)
        
        # Build prompt with only valid domains
        prompt = f"""You are an SDTM expert. Select the most appropriate domain for this {pattern} pattern.

Field Information:
- Label: {field_data.get('label', '')}
- Form: {field_data.get('form_name', '')}
- Section: {field_data.get('section', '')}
- Pattern: {pattern}

VALID DOMAINS FOR {pattern.upper()} PATTERN:
"""
        
        for domain_code, domain_info, priority in filtered_domains[:10]:
            prompt += f"\n{domain_code} - {domain_info.get('label', '')}"
            prompt += f"\n  Definition: {domain_info.get('description', '')}"
            if priority > 0.7:
                prompt += f"\n  **RECOMMENDED** based on form/section context"
            prompt += "\n"
        
        prompt += f"\n\nSelect ONE domain code from the VALID options above:"
        
        response = mapper_instance._query_llm(prompt, max_tokens=10)
        selected_domain = response.upper().strip()
        
        # Validate selection is from filtered list
        valid_domains = [d[0] for d in filtered_domains]
        if selected_domain in valid_domains:
            # Find priority
            for d, _, p in filtered_domains:
                if d == selected_domain:
                    confidence = min(0.95, 0.7 + p * 0.3)
                    break
        else:
            # Use top recommendation
            selected_domain = filtered_domains[0][0]
            confidence = 0.7
        
        return selected_domain, confidence
    
    mapper_instance._select_domain_with_class_constraint = filtered_select_domain


if __name__ == "__main__":
    # Test the filter
    with open("kb/sdtmig_v3_4_complete/proto_define.json", 'r') as f:
        proto_define = json.load(f)
    
    filter = RuleBasedFilter(proto_define)
    
    # Test filtering domains for findings pattern
    print("Domains valid for FINDINGS pattern (Findings class):")
    findings_domains = filter.filter_domains_for_pattern(
        "findings", "Findings", {"form_name": "Vital Signs"}
    )
    for domain, info, priority in findings_domains[:5]:
        print(f"  {domain} - {info.get('label')} (priority: {priority:.2f})")
    
    # Test filtering variables for IE conditional
    print("\n\nVariables valid for CONDITIONAL pattern in IE domain:")
    ie_vars = filter.filter_variables_for_pattern(
        "conditional", "IE", {"label": "Age >= 18 (INC1)"}
    )
    for var, info, priority in ie_vars:
        print(f"  {var} - {info.get('label')} (priority: {priority:.2f})")