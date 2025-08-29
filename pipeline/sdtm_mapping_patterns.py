#!/usr/bin/env python3
"""
SDTM Mapping Patterns based on CDISC aCRF annotation rules
Implements the semantic patterns from Section 3 of SDTM aCRF specification
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re


class MappingPatternType(Enum):
    """Types of SDTM mapping patterns"""
    PLAIN = "plain"  # Direct variable mapping
    FINDINGS = "findings"  # Result with test context
    SUPPLEMENTAL = "supplemental"  # SUPP qualifiers
    RELREC = "relrec"  # Cross-domain relationships
    NOT_SUBMITTED = "not_submitted"  # Collected but not in SDTM
    CLARIFICATION = "clarification"  # Non-collected, for clarity


@dataclass
class MappingPattern:
    """Base class for all mapping patterns"""
    pattern_type: MappingPatternType
    confidence: float = 0.0
    reasoning: str = ""


class PlainMapping(MappingPattern):
    """Direct variable mapping: VARIABLE or VAR1 / VAR2 / VAR3"""
    def __init__(self, variables: List[str], confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.PLAIN, confidence=confidence, reasoning=reasoning)
        self.variables = variables if isinstance(variables, list) else [variables]
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        return " / ".join(self.variables)


class FindingsMapping(MappingPattern):
    """Findings pattern: --ORRES [/ --ORRESU] when --TESTCD = VALUE"""
    def __init__(self, domain: str, result_vars: List[str], testcd_value: str, confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.FINDINGS, confidence=confidence, reasoning=reasoning)
        self.domain = domain
        self.result_vars = result_vars
        self.testcd_value = testcd_value
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        results = " / ".join(self.result_vars)
        return f"{results} when {self.domain}TESTCD = {self.testcd_value}"


class SupplementalMapping(MappingPattern):
    """Supplemental pattern: QNAM in SUPPDOMAIN"""
    def __init__(self, qnam: str, parent_domain: str, confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.SUPPLEMENTAL, confidence=confidence, reasoning=reasoning)
        self.qnam = qnam
        self.parent_domain = parent_domain
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        return f"{self.qnam} in SUPP{self.parent_domain}"


class RelrecMapping(MappingPattern):
    """RELREC pattern: RELREC when collected_var = related_domain.var"""
    def __init__(self, collected_var: str, related_domain: str, related_var: str, confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.RELREC, confidence=confidence, reasoning=reasoning)
        self.collected_var = collected_var
        self.related_domain = related_domain
        self.related_var = related_var
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        return f"RELREC when {self.collected_var} = {self.related_domain}.{self.related_var}"


class NotSubmittedMapping(MappingPattern):
    """Collected but not submitted: VARIABLE [NOT SUBMITTED]"""
    def __init__(self, variable: str, confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.NOT_SUBMITTED, confidence=confidence, reasoning=reasoning)
        self.variable = variable
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        return f"{self.variable} [NOT SUBMITTED]"


class ClarificationMapping(MappingPattern):
    """Non-collected clarification: VARIABLE [clarification; not Collected]"""
    def __init__(self, variable: str, confidence: float = 0.0, reasoning: str = ""):
        super().__init__(pattern_type=MappingPatternType.CLARIFICATION, confidence=confidence, reasoning=reasoning)
        self.variable = variable
        
    def to_annotation(self) -> str:
        """Generate aCRF annotation string"""
        return f"{self.variable} [clarification; not Collected]"


class SDTMMappingRules:
    """Rules engine for determining appropriate SDTM mapping patterns"""
    
    # Variable groupings that should be annotated together
    VARIABLE_GROUPS = {
        # Date and time pairs
        "datetime_pairs": [
            ("DTC", "TM"),  # Date/Time of Collection + Time
            ("STDTC", "STTM"),  # Start Date/Time + Start Time
            ("ENDTC", "ENTM"),  # End Date/Time + End Time
        ],
        # Value and unit pairs
        "value_unit_pairs": [
            ("DOSE", "DOSU"),  # Dose + Dose Units
            ("ORRES", "ORRESU"),  # Original Result + Original Units
            ("STRESC", "STRESU"),  # Standardized Result Character + Units
            ("STRESN", "STRESU"),  # Standardized Result Numeric + Units
            ("AGE", "AGEU"),  # Age + Age Units
            ("DUR", "DURU"),  # Duration + Duration Units
        ],
        # Result sets for findings
        "result_sets": [
            ("ORRES", "ORRESU", "STRESC", "STRESN", "STRESU"),  # Complete result set
            ("ORRES", "STRESC", "STRESN"),  # Result set without units
        ],
        # Coded term sets
        "coded_sets": [
            ("TERM", "DECOD"),  # Reported Term + Dictionary Decoded
            ("TERM", "DECOD", "BODSYS"),  # Term + Decoded + Body System
        ]
    }
    
    # Common findings domains and their standard variables
    FINDINGS_DOMAINS = {
        "VS": {  # Vital Signs
            "tests": ["TEMP", "SYSBP", "DIABP", "HR", "RESP", "HEIGHT", "WEIGHT"],
            "result_vars": ["VSORRES", "VSORRESU", "VSSTRESC", "VSSTRESN", "VSSTRESU"]
        },
        "LB": {  # Laboratory
            "tests": ["HGB", "WBC", "PLAT", "SODIUM", "GLUC"],
            "result_vars": ["LBORRES", "LBORRESU", "LBSTRESC", "LBSTRESN", "LBSTRESU"]
        },
        "QS": {  # Questionnaires
            "tests": [],  # Various questionnaire items
            "result_vars": ["QSORRES", "QSSTRESC", "QSSTRESN", "QSSTRESU"]
        },
        "EG": {  # ECG
            "tests": ["QTCF", "QT", "RR", "PR"],
            "result_vars": ["EGORRES", "EGORRESU", "EGSTRESC", "EGSTRESN", "EGSTRESU"]
        },
        "CV": {  # Cardiovascular
            "tests": [],
            "result_vars": ["CVORRES", "CVORRESU", "CVSTRESC", "CVSTRESN", "CVSTRESU"]
        }
    }
    
    # Common supplemental qualifiers by domain
    COMMON_SUPP = {
        "DM": ["RACEOTH", "ETHNIC", "RACESPEC"],
        "AE": ["AEACNOTH", "AERLDEV", "AERELNST"],
        "CM": ["CMINDC", "CMDOSFRQ", "CMROUTE"],
        "MH": ["MHONGO", "MHDECOD", "MHBODSYS"]
    }
    
    @classmethod
    def determine_pattern(cls, question: Dict, domain: str, variable: str, 
                         context: Dict) -> MappingPattern:
        """Determine the appropriate mapping pattern for a question"""
        
        # Check if it's a findings domain
        if domain in cls.FINDINGS_DOMAINS:
            return cls._create_findings_pattern(question, domain, variable, context)
            
        # Check if it's a supplemental qualifier
        if cls._is_supplemental(variable, domain):
            return cls._create_supplemental_pattern(variable, domain, context)
            
        # Check if it's a relationship
        if cls._is_relrec(question, variable):
            return cls._create_relrec_pattern(question, variable, context)
            
        # Check if it's not submitted
        if cls._is_not_submitted(question, context):
            return NotSubmittedMapping(variable=variable, confidence=0.9)
            
        # Check if it's a clarification
        if cls._is_clarification(variable, context):
            return ClarificationMapping(variable=variable, confidence=0.8)
            
        # Default to plain mapping
        return cls._create_plain_mapping(variable, question, context)
        
    @classmethod
    def _create_findings_pattern(cls, question: Dict, domain: str, 
                                variable: str, context: Dict) -> FindingsMapping:
        """Create a findings pattern mapping"""
        
        # Determine test code from question
        testcd_value = cls._extract_testcd(question, domain, context)
        
        # Determine result variables based on variable type
        result_vars = []
        if variable.endswith("ORRES"):
            result_vars = [variable]
            # Add units if likely present
            if question.get("type") in ["field", "numeric"]:
                result_vars.append(variable.replace("ORRES", "ORRESU"))
            # Add standardized results if numeric
            if cls._is_numeric_question(question):
                result_vars.extend([
                    variable.replace("ORRES", "STRESC"),
                    variable.replace("ORRES", "STRESN")
                ])
                
        return FindingsMapping(
            domain=domain,
            result_vars=result_vars,
            testcd_value=testcd_value,
            confidence=0.85,
            reasoning=f"Findings pattern for {domain} domain"
        )
        
    @classmethod
    def _create_supplemental_pattern(cls, variable: str, domain: str, 
                                    context: Dict) -> SupplementalMapping:
        """Create a supplemental qualifier pattern"""
        
        # Extract QNAM from variable or context
        qnam = variable  # Often the variable itself is the QNAM
        
        # Check common patterns
        if domain == "DM" and "race" in variable.lower() and "other" in variable.lower():
            qnam = "RACEOTH"
        elif domain == "AE" and "other" in variable.lower():
            qnam = "AEACNOTH"
            
        return SupplementalMapping(
            qnam=qnam,
            parent_domain=domain,
            confidence=0.8,
            reasoning=f"Supplemental qualifier for {domain}"
        )
        
    @classmethod
    def _create_relrec_pattern(cls, question: Dict, variable: str, 
                              context: Dict) -> RelrecMapping:
        """Create a RELREC pattern mapping"""
        
        # Extract link information from question
        # This is simplified - real implementation would be more sophisticated
        collected_var = variable
        
        # Common patterns
        if "LNKID" in variable:
            # Death details linked to AE
            if "death" in question['text'].lower() or variable == "DDLNKID":
                return RelrecMapping(
                    collected_var="DDLNKID",
                    related_domain="AE",
                    related_var="AELNKID",
                    confidence=0.8,
                    reasoning="Death details linked to adverse event"
                )
            # Generic AE link pattern
            elif "adverse" in question['text'].lower() or variable.endswith("AELNKID"):
                return RelrecMapping(
                    collected_var=variable,
                    related_domain="AE",
                    related_var="AELNKID",
                    confidence=0.8,
                    reasoning="Related to adverse event"
                )
                
        return RelrecMapping(
            collected_var=collected_var,
            related_domain="UNKNOWN",
            related_var="UNKNOWN",
            confidence=0.5,
            reasoning="Relationship pattern detected"
        )
        
    @classmethod
    def _create_plain_mapping(cls, variable: str, question: Dict, 
                             context: Dict) -> PlainMapping:
        """Create a plain mapping pattern with multiple variable detection"""
        
        variables = cls._detect_multiple_variables(variable, question, context)
                
        return PlainMapping(
            variables=variables,
            confidence=0.9,
            reasoning=cls._get_multiple_var_reason(variables) if len(variables) > 1 else "Direct variable mapping"
        )
    
    @classmethod
    def _detect_multiple_variables(cls, primary_var: str, question: Dict, 
                                  context: Dict) -> List[str]:
        """Detect if multiple variables should be annotated together"""
        
        variables = [primary_var]
        question_text = question.get('text', '').lower()
        
        # Check for date/time pairs
        if primary_var.endswith("DTC"):
            # Check if time is also collected
            if any(time_word in question_text for time_word in ["time", "hour", "minute"]):
                # Handle both TM and TTM patterns
                if "STTM" in primary_var or "STDTC" in primary_var:
                    time_var = primary_var.replace("STDTC", "STTM")
                elif "ENTM" in primary_var or "ENDTC" in primary_var:
                    time_var = primary_var.replace("ENDTC", "ENTM")
                else:
                    time_var = primary_var.replace("DTC", "TM")
                variables.append(time_var)
                
        # Check for value/unit pairs
        if cls._needs_units(primary_var):
            unit_var = cls._get_unit_variable(primary_var)
            if unit_var:
                variables.append(unit_var)
                
        # Check for start/end pairs
        if "start" in question_text and "end" in question_text:
            if primary_var.endswith("STDTC"):
                end_var = primary_var.replace("STDTC", "ENDTC")
                variables.append(end_var)
                
        # Check for coded term sets
        if primary_var.endswith("TERM"):
            if "coded" in question_text or "dictionary" in question_text:
                variables.append(primary_var.replace("TERM", "DECOD"))
                if "body system" in question_text:
                    variables.append(primary_var.replace("TERM", "BODSYS"))
                    
        return variables
    
    @classmethod
    def _get_unit_variable(cls, value_var: str) -> Optional[str]:
        """Get the corresponding unit variable"""
        
        unit_mappings = {
            "ORRES": "ORRESU",
            "STRESC": "STRESU", 
            "STRESN": "STRESU",
            "DOSE": "DOSU",
            "AGE": "AGEU",
            "DUR": "DURU"
        }
        
        for pattern, unit_suffix in unit_mappings.items():
            if value_var.endswith(pattern):
                return value_var.replace(pattern, unit_suffix)
                
        return None
    
    @classmethod
    def _get_multiple_var_reason(cls, variables: List[str]) -> str:
        """Get reasoning for multiple variable annotation"""
        
        var_str = " ".join(variables).upper()
        
        if "DTC" in var_str and "TM" in var_str:
            return "Date and time collection mapped together"
        elif any(suffix in var_str for suffix in ["ORRESU", "DOSU", "STRESU", "AGEU"]):
            return "Value with units mapped together"
        elif "STDTC" in var_str and "ENDTC" in var_str:
            return "Start and end dates mapped together"
        elif "TERM" in var_str and "DECOD" in var_str:
            return "Verbatim term with coded value"
        else:
            return "Related variables mapped together"
        
    @classmethod
    def _is_supplemental(cls, variable: str, domain: str) -> bool:
        """Check if variable is a supplemental qualifier"""
        
        # Check known supplemental patterns
        if domain in cls.COMMON_SUPP:
            return variable in cls.COMMON_SUPP[domain]
            
        # Check common patterns
        return any(pattern in variable.upper() for pattern in ["OTH", "SPEC"])
        
    @classmethod
    def _is_relrec(cls, question: Dict, variable: str) -> bool:
        """Check if this represents a relationship"""
        return "LNKID" in variable or "link" in question.get('text', '').lower()
        
    @classmethod
    def _is_not_submitted(cls, question: Dict, context: Dict) -> bool:
        """Check if collected but not submitted to SDTM"""
        
        # Check for operational/sponsor fields
        operational_keywords = ["crf page", "visit date collected", "data entry", 
                              "monitor", "signature", "initials"]
        text_lower = question.get('text', '').lower()
        
        return any(kw in text_lower for kw in operational_keywords)
        
    @classmethod
    def _is_clarification(cls, variable: str, context: Dict) -> bool:
        """Check if this is a clarification variable (not collected)"""
        
        # Common clarification variables
        clarification_vars = ["REPNUM", "TPTNUM", "GRPID", "SPID"]
        
        return any(cv in variable for cv in clarification_vars)
        
    @classmethod
    def _extract_testcd(cls, question: Dict, domain: str, context: Dict) -> str:
        """Extract test code from question text"""
        
        text = question.get('text', '').upper()
        
        # Check known tests for the domain
        if domain in cls.FINDINGS_DOMAINS:
            for test in cls.FINDINGS_DOMAINS[domain]["tests"]:
                if test in text:
                    return test
                    
        # Extract from common patterns
        if "TEMPERATURE" in text:
            return "TEMP"
        elif "SYSTOLIC" in text:
            return "SYSBP"
        elif "DIASTOLIC" in text:
            return "DIABP"
        elif "HEART RATE" in text or "PULSE" in text:
            return "HR"
            
        # Default to extracted abbreviation
        return "UNKNOWN"
        
    @classmethod
    def _is_numeric_question(cls, question: Dict) -> bool:
        """Check if question expects numeric response"""
        
        numeric_indicators = ["numeric", "number", "integer", "float", "decimal"]
        type_lower = question.get('type', '').lower()
        text_lower = question.get('text', '').lower()
        
        return (any(ind in type_lower for ind in numeric_indicators) or
                any(ind in text_lower for ind in ["rate", "pressure", "count", "score"]))
                
    @classmethod
    def _needs_units(cls, variable: str) -> bool:
        """Check if variable typically has associated units"""
        
        units_patterns = ["ORRES", "STRES", "DOSE", "AGE", "DUR", "HEIGHT", "WEIGHT"]
        return any(pattern in variable.upper() for pattern in units_patterns)


def format_annotation_output(page_data: Dict, mappings: List[Dict]) -> Dict:
    """Format mappings according to aCRF annotation specification"""
    
    # Extract domains present on page
    domains_on_page = set()
    annotations = []
    
    for mapping in mappings:
        domain = mapping.get("domain", {}).get("choice", "")
        if domain and domain != "UNKNOWN":
            domains_on_page.add(domain)
            
        # Create annotation based on pattern
        pattern = mapping.get("pattern")
        if pattern:
            pattern_obj = create_pattern_from_dict(pattern)
            annotation = {
                "question_id": mapping.get("question_id"),
                "question_text": mapping.get("question_text"),
                "annotation": pattern_obj.to_annotation() if pattern_obj else "",
                "confidence": pattern.get("confidence", 0.0)
            }
            annotations.append(annotation)
            
    # Format output
    output = {
        "page_id": page_data.get("page_id"),
        "domains_on_page": sorted(list(domains_on_page)),
        "annotations": annotations
    }
    
    return output


def create_pattern_from_dict(pattern_dict: Dict) -> Optional[MappingPattern]:
    """Create pattern object from dictionary representation"""
    
    pattern_type = pattern_dict.get("type")
    
    if pattern_type == "plain":
        return PlainMapping(
            variables=pattern_dict.get("variables", []),
            confidence=pattern_dict.get("confidence", 0.0),
            reasoning=pattern_dict.get("reasoning", "")
        )
    elif pattern_type == "findings":
        return FindingsMapping(
            domain=pattern_dict.get("domain", ""),
            result_vars=pattern_dict.get("result_vars", []),
            testcd_value=pattern_dict.get("testcd_value", ""),
            confidence=pattern_dict.get("confidence", 0.0),
            reasoning=pattern_dict.get("reasoning", "")
        )
    elif pattern_type == "supplemental":
        return SupplementalMapping(
            qnam=pattern_dict.get("qnam", ""),
            parent_domain=pattern_dict.get("parent_domain", ""),
            confidence=pattern_dict.get("confidence", 0.0),
            reasoning=pattern_dict.get("reasoning", "")
        )
    # Add other pattern types as needed
    
    return None


# Example usage
if __name__ == "__main__":
    print("SDTM-MSG v2.0 Annotation Patterns Examples")
    print("=" * 60)
    
    # Example 1: Vital Signs with units (Findings pattern)
    vs_question = {
        "text": "Systolic Blood Pressure (mmHg)",
        "type": "numeric",
        "item_id": "vs_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=vs_question,
        domain="VS",
        variable="VSORRES",
        context={"form_type": "vital_signs"}
    )
    
    print(f"\n1. Findings Pattern:")
    print(f"   Question: {vs_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: VSORRES / VSORRESU / VSSTRESC / VSSTRESN when VSTESTCD = SYSBP")
    
    # Example 2: Multiple variables - Date/Time
    datetime_question = {
        "text": "Visit Date and Time",
        "type": "datetime",
        "item_id": "sv_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=datetime_question,
        domain="SV",
        variable="SVSTDTC",
        context={}
    )
    
    print(f"\n2. Multiple Variables - Date/Time:")
    print(f"   Question: {datetime_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: SVSTDTC / SVSTTM")
    
    # Example 3: Multiple variables - Dose with units
    dose_question = {
        "text": "Study Drug Dose (mg)",
        "type": "numeric",
        "item_id": "ex_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=dose_question,
        domain="EX",
        variable="EXDOSE",
        context={}
    )
    
    print(f"\n3. Multiple Variables - Value/Unit:")
    print(f"   Question: {dose_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: EXDOSE / EXDOSU")
    
    # Example 4: Supplemental qualifier
    supp_question = {
        "text": "Race - Other, specify:",
        "type": "text",
        "item_id": "dm_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=supp_question,
        domain="DM",
        variable="RACEOTH",
        context={}
    )
    
    print(f"\n4. Supplemental Qualifier:")
    print(f"   Question: {supp_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: RACEOTH in SUPPDM")
    
    # Example 5: RELREC pattern
    relrec_question = {
        "text": "Related Adverse Event ID",
        "type": "text",
        "item_id": "dd_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=relrec_question,
        domain="DD",
        variable="DDLNKID",
        context={}
    )
    
    print(f"\n5. RELREC Pattern:")
    print(f"   Question: {relrec_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: RELREC when DDLNKID = AELNKID")
    
    # Example 6: NOT SUBMITTED
    not_submitted_question = {
        "text": "CRF Page Number",
        "type": "text",
        "item_id": "op_001"
    }
    
    pattern = SDTMMappingRules.determine_pattern(
        question=not_submitted_question,
        domain="DM",
        variable="CRFPG",
        context={}
    )
    
    print(f"\n6. Not Submitted:")
    print(f"   Question: {not_submitted_question['text']}")
    print(f"   Pattern type: {pattern.pattern_type.value}")
    print(f"   Annotation: {pattern.to_annotation()}")
    print(f"   Expected: CRFPG [NOT SUBMITTED]")