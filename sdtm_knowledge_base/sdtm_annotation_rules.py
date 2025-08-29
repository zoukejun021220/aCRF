#!/usr/bin/env python3
"""
SDTM Annotation Rules Module
Provides structured rules and patterns for SDTM-MSG v2.0 compliant CRF annotation
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AnnotationType(Enum):
    """Types of SDTM annotations"""
    DOMAIN = "domain"
    VARIABLE = "variable"
    FINDINGS = "findings"
    SUPPLEMENTAL = "supplemental"
    RELREC = "relrec"
    NOT_SUBMITTED = "not_submitted"
    CLARIFYING = "clarifying"


@dataclass
class AnnotationRule:
    """Individual annotation rule"""
    rule_id: str
    category: str
    title: str
    description: str
    pattern: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    validation: Optional[str] = None
    msg_reference: Optional[str] = None


class SDTMAnnotationRules:
    """Manager for SDTM-MSG v2.0 annotation rules"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.patterns = self._initialize_patterns()
        self.validation_checks = self._initialize_validations()
        
    def _initialize_rules(self) -> Dict[str, List[AnnotationRule]]:
        """Initialize all annotation rules grouped by category"""
        
        rules = {
            "general_principles": [
                AnnotationRule(
                    rule_id="GP001",
                    category="general_principles",
                    title="Annotate for tabulation only",
                    description="Include only annotations for SDTM tabulation datasets. Exclude sponsor-internal operational/collection system variables that are not part of the regulatory submission.",
                    msg_reference="Section 3.1"
                ),
                AnnotationRule(
                    rule_id="GP002",
                    category="general_principles",
                    title="Searchable text requirement",
                    description="Annotations must be text (not pictures); they must remain searchable even if the PDF is flattened.",
                    msg_reference="Section 3.1"
                ),
                AnnotationRule(
                    rule_id="GP003",
                    category="general_principles",
                    title="Planned but uncollected data",
                    description="If a data item was planned but no data were ultimately collected, keep the annotation. The absence of data is indicated in Define-XML using HasNoData.",
                    msg_reference="Section 3.1"
                ),
            ],
            
            "formatting": [
                AnnotationRule(
                    rule_id="FMT001",
                    category="formatting",
                    title="Domain annotation format",
                    description="Domain annotations use black, bold text with format: DOM (Domain Label)",
                    pattern="^[A-Z]{2}\\s\\([^)]+\\)$",
                    examples=["DM (Demographics)", "VS (Vital Signs)", "AE (Adverse Events)"],
                    msg_reference="Section 3.1.2"
                ),
                AnnotationRule(
                    rule_id="FMT002",
                    category="formatting",
                    title="Variable annotation format",
                    description="Variable annotations use black, regular text. Variables and dataset codes are UPPERCASE.",
                    examples=["BRTHDTC", "VSORRES", "AEACN01"],
                    msg_reference="Section 3.1.2"
                ),
                AnnotationRule(
                    rule_id="FMT003",
                    category="formatting",
                    title="Multiple variables with slashes",
                    description="If one field maps to multiple variables, separate variable names with a forward slash.",
                    pattern="^[A-Z]+(/[A-Z]+)*$",
                    examples=["VSORRES/VSORRESU", "LBORRES/LBORRESU/LBORNRLO/LBORNRHI"],
                    msg_reference="Section 3.1.2"
                ),
                AnnotationRule(
                    rule_id="FMT004",
                    category="formatting",
                    title="Not submitted annotation",
                    description="If a prompt or field is collected but not intended for SDTM, annotate it as [NOT SUBMITTED].",
                    pattern="\\[NOT SUBMITTED\\]",
                    examples=["[NOT SUBMITTED]"],
                    msg_reference="Section 3.1.2"
                ),
            ],
            
            "findings_patterns": [
                AnnotationRule(
                    rule_id="FND001",
                    category="findings_patterns",
                    title="Findings Pattern A (combined)",
                    description="For vertical findings domains, use combined pattern: --ORRES when --TESTCD = <VALUE>",
                    pattern="^[A-Z]+\\swhen\\s[A-Z]+\\s=\\s[A-Z0-9]+$",
                    examples=[
                        "VSORRES when VSTESTCD = TEMP",
                        "LBORRES/LBORRESU when LBTESTCD = GLUC",
                        "QSORRES when QSTESTCD = PHQ01"
                    ],
                    msg_reference="Section 3.1.3"
                ),
                AnnotationRule(
                    rule_id="FND002",
                    category="findings_patterns",
                    title="Findings Pattern B (separate)",
                    description="For crowded pages, annotate --TESTCD = <VALUE> near test label and --ORRES near result box separately.",
                    examples=[
                        "VSTESTCD = TEMP",
                        "VSORRES/VSORRESU"
                    ],
                    msg_reference="Section 3.1.3"
                ),
            ],
            
            "supplemental": [
                AnnotationRule(
                    rule_id="SUPP001",
                    category="supplemental",
                    title="Supplemental qualifier pattern",
                    description="Annotate supplemental qualifiers with pattern: <QNAM> in SUPP<DOMAIN>",
                    pattern="^[A-Z]+\\sin\\sSUPP[A-Z]{2}$",
                    examples=[
                        "RACEOTH in SUPPDM",
                        "AEACN01 in SUPPAE",
                        "CMONGO in SUPPCM"
                    ],
                    msg_reference="Section 3.1.3"
                ),
            ],
            
            "relationships": [
                AnnotationRule(
                    rule_id="REL001",
                    category="relationships",
                    title="RELREC pattern",
                    description="When a form captures a linking identifier showing a relationship between records",
                    pattern="^RELREC\\swhen\\s[A-Z]+\\s=\\s[A-Z]+\\.[A-Z]+$",
                    examples=[
                        "RELREC when DDLNKID = AE.AELNKID",
                        "RELREC when MHLNKID = AE.AETERM",
                        "RELREC when EXLNKID = EC.ECDOSE"
                    ],
                    msg_reference="Section 3.1.3"
                ),
            ],
            
            "navigation": [
                AnnotationRule(
                    rule_id="NAV001",
                    category="navigation",
                    title="Dual bookmarking requirement",
                    description="Create bookmarks by Visits (chronology) and by Forms (topics). Running Records for non-visit pages.",
                    msg_reference="Sections 3.2, 3.3"
                ),
                AnnotationRule(
                    rule_id="NAV002",
                    category="navigation",
                    title="Unique forms strategy",
                    description="Default: annotate unique forms only. Provide one annotated instance of each unique form and represent repeats via bookmarks.",
                    msg_reference="Section 3.1.1"
                ),
            ]
        }
        
        return rules
    
    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for validation"""
        
        return {
            "domain_annotation": r"^[A-Z]{2}\s\([^)]+\)$",
            "variable_simple": r"^[A-Z]+$",
            "variable_multiple": r"^[A-Z]+(/[A-Z]+)*$",
            "findings_combined": r"^[A-Z]+(/[A-Z]+)*\swhen\s[A-Z]+\s=\s[A-Z0-9]+$",
            "findings_testcd": r"^[A-Z]+\s=\s[A-Z0-9]+$",
            "supplemental": r"^[A-Z]+\sin\sSUPP[A-Z]{2}$",
            "relrec": r"^RELREC\swhen\s[A-Z]+\s=\s[A-Z]+\.[A-Z]+$",
            "not_submitted": r"\[NOT SUBMITTED\]"
        }
    
    def _initialize_validations(self) -> List[Dict[str, any]]:
        """Initialize validation checks"""
        
        return [
            {
                "check_id": "VAL001",
                "description": "Every Findings variable that can vary by test has a when clause referencing --TESTCD",
                "applies_to": ["findings_patterns"],
                "validation_func": self._validate_findings_when_clause
            },
            {
                "check_id": "VAL002",
                "description": "Supplemental data uses correct format with QNAM and SUPP<DOMAIN>",
                "applies_to": ["supplemental"],
                "validation_func": self._validate_supplemental_format
            },
            {
                "check_id": "VAL003",
                "description": "Variables and dataset codes are UPPERCASE",
                "applies_to": ["formatting"],
                "validation_func": self._validate_uppercase
            },
            {
                "check_id": "VAL004",
                "description": "Domain annotations follow correct format",
                "applies_to": ["formatting"],
                "validation_func": self._validate_domain_format
            },
        ]
    
    def get_rules_for_context(self, context: str) -> List[AnnotationRule]:
        """Get relevant rules for a specific context"""
        
        relevant_rules = []
        
        # Map contexts to rule categories
        context_mapping = {
            "findings": ["findings_patterns", "formatting"],
            "demographics": ["general_principles", "formatting", "supplemental"],
            "adverse_events": ["general_principles", "formatting", "relationships"],
            "questionnaire": ["findings_patterns", "formatting"],
            "supplemental": ["supplemental", "formatting"],
            "navigation": ["navigation"],
        }
        
        categories = context_mapping.get(context.lower(), ["general_principles", "formatting"])
        
        for category in categories:
            if category in self.rules:
                relevant_rules.extend(self.rules[category])
        
        return relevant_rules
    
    def get_pattern_for_type(self, annotation_type: AnnotationType) -> Optional[str]:
        """Get regex pattern for specific annotation type"""
        
        pattern_mapping = {
            AnnotationType.DOMAIN: self.patterns["domain_annotation"],
            AnnotationType.VARIABLE: self.patterns["variable_simple"],
            AnnotationType.FINDINGS: self.patterns["findings_combined"],
            AnnotationType.SUPPLEMENTAL: self.patterns["supplemental"],
            AnnotationType.RELREC: self.patterns["relrec"],
            AnnotationType.NOT_SUBMITTED: self.patterns["not_submitted"],
        }
        
        return pattern_mapping.get(annotation_type)
    
    def format_annotation(self, annotation_type: AnnotationType, **kwargs) -> str:
        """Format annotation according to SDTM-MSG v2.0 rules"""
        
        if annotation_type == AnnotationType.DOMAIN:
            return f"{kwargs['domain']} ({kwargs['label']})"
            
        elif annotation_type == AnnotationType.VARIABLE:
            variables = kwargs.get('variables', [])
            if isinstance(variables, str):
                variables = [variables]
            return "/".join(variables)
            
        elif annotation_type == AnnotationType.FINDINGS:
            variables = kwargs.get('variables', [])
            if isinstance(variables, str):
                variables = [variables]
            testcd_var = kwargs['testcd_var']
            testcd_value = kwargs['testcd_value']
            return f"{'/'.join(variables)} when {testcd_var} = {testcd_value}"
            
        elif annotation_type == AnnotationType.SUPPLEMENTAL:
            return f"{kwargs['qnam']} in SUPP{kwargs['domain']}"
            
        elif annotation_type == AnnotationType.RELREC:
            return f"RELREC when {kwargs['link_var']} = {kwargs['related_domain']}.{kwargs['related_var']}"
            
        elif annotation_type == AnnotationType.NOT_SUBMITTED:
            return "[NOT SUBMITTED]"
            
        else:
            return ""
    
    def get_rules_summary(self) -> Dict[str, str]:
        """Get concise summary of key rules for LLM context"""
        
        return {
            "core_principles": [
                "Annotate SDTM tabulation data only - exclude operational fields",
                "Keep annotations for planned but uncollected data",
                "Annotations must be searchable text",
            ],
            "formatting_rules": [
                "Domain: DOM (Domain Label) - black bold text",
                "Variables: UPPERCASE - black regular text",
                "Multiple variables: separate with forward slash (/)",
                "Not submitted: mark as [NOT SUBMITTED]",
            ],
            "findings_pattern": [
                "Pattern A (compact): VSORRES/VSORRESU when VSTESTCD = TEMP",
                "Pattern B (separate): VSTESTCD = TEMP ... VSORRES/VSORRESU",
            ],
            "special_cases": [
                "Supplemental: RACEOTH in SUPPDM",
                "Relationships: RELREC when DDLNKID = AE.AELNKID",
                "Clarifying annotations: use dashed borders (non-collected)",
            ],
            "navigation": [
                "Dual bookmarks required: by Visits and by Forms",
                "Running Records for non-visit pages",
                "Unique forms with bookmarks for repeats",
            ]
        }
    
    def _validate_findings_when_clause(self, annotation: str) -> bool:
        """Validate findings annotation has proper when clause"""
        import re
        
        findings_vars = ["ORRES", "ORRESU", "STRESC", "STRESN", "ORNRLO", "ORNRHI"]
        
        # Check if annotation contains findings variables
        has_findings_var = any(var in annotation.upper() for var in findings_vars)
        
        if has_findings_var:
            # Must have "when" clause
            return "when" in annotation and "TESTCD" in annotation
        
        return True
    
    def _validate_supplemental_format(self, annotation: str) -> bool:
        """Validate supplemental qualifier format"""
        import re
        
        if "SUPP" in annotation:
            return bool(re.match(self.patterns["supplemental"], annotation))
        
        return True
    
    def _validate_uppercase(self, annotation: str) -> bool:
        """Validate variables are uppercase"""
        import re
        
        # Extract variable names (sequences of letters that should be uppercase)
        var_pattern = r'\b[A-Z][A-Z0-9]{1,7}\b'
        
        # Skip domain labels in parentheses
        if "(" in annotation and ")" in annotation:
            # Remove content in parentheses for checking
            annotation_check = re.sub(r'\([^)]+\)', '', annotation)
        else:
            annotation_check = annotation
        
        # Find all potential variable names
        potential_vars = re.findall(r'\b[A-Za-z][A-Za-z0-9]{1,7}\b', annotation_check)
        
        # Check if they are uppercase (excluding common words)
        exclude_words = {"when", "in", "NOT", "SUBMITTED"}
        
        for var in potential_vars:
            if var not in exclude_words and var != var.upper():
                return False
        
        return True
    
    def _validate_domain_format(self, annotation: str) -> bool:
        """Validate domain annotation format"""
        import re
        
        # Check if it's a domain annotation
        if "(" in annotation and ")" in annotation:
            return bool(re.match(self.patterns["domain_annotation"], annotation))
        
        return True


def create_annotation_rules() -> SDTMAnnotationRules:
    """Factory function to create annotation rules instance"""
    return SDTMAnnotationRules()