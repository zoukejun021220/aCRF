#!/usr/bin/env python3
"""
Create instruction tuning dataset for SDTM mapper
Converts reference annotations into step-by-step instruction-response pairs.

Now supports CLI overrides for KB, reference, CRF, and output directories, so you can
point at an external reference set (e.g. /home/.../reference_with_sections).
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import random
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """Single instruction-response example for fine-tuning"""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any]


class InstructionDatasetBuilder:
    def __init__(self, kb_path: str):
        """Initialize with knowledge base path"""
        self.kb_path = Path(kb_path)
        self.load_kb_resources()
        
    def load_kb_resources(self):
        """Load KB resources for building prompts"""
        # Load class definitions
        class_def_path = self.kb_path / "class_definitions.json"
        if class_def_path.exists():
            with open(class_def_path) as f:
                class_data = json.load(f)
                self.class_definitions = class_data.get('sdtm_classes', {})
                self.class_hierarchy = class_data.get('decision_hierarchy', {})
        
        # Load domains by class
        domains_path = self.kb_path / "domains_by_class.json"
        if domains_path.exists():
            with open(domains_path) as f:
                self.domains_by_class = json.load(f)
        
        # Load proto_define
        proto_path = self.kb_path / "proto_define.json"
        if proto_path.exists():
            with open(proto_path) as f:
                self.proto_define = json.load(f)
                self.datasets = self.proto_define.get("datasets", {})
                self.variables = self.proto_define.get("variables", {})
        
        # Load pattern definitions
        patterns_path = self.kb_path / "pattern_definitions.json"
        if patterns_path.exists():
            with open(patterns_path) as f:
                self.patterns = json.load(f)
                self.annotation_patterns = self.patterns.get("annotation_patterns", {})
        
        # Load controlled terminology
        ct_path = self.kb_path / "cdisc_ct.json"
        if ct_path.exists():
            with open(ct_path) as f:
                self.ct_data = json.load(f)

    # ===== Helpers to mirror unified mapper prompts =====
    def _get_all_domain_descriptions(self) -> str:
        """Format domain descriptions exactly like UnifiedSDTMMapper."""
        descriptions = "SDTM Domains (organized by observation class):\n\n"
        class_order = ["Special Purpose", "Interventions", "Events", "Findings"]
        for sdtm_class in class_order:
            if sdtm_class in getattr(self, "domains_by_class", {}):
                domains = self.domains_by_class[sdtm_class]
                descriptions += f"**{sdtm_class} Class**:\n"
                for domain in domains:
                    code = domain.get('code', '')
                    name = domain.get('name', '')
                    desc = domain.get('description', '')
                    descriptions += f"- {code} ({name}) – {desc}\n"
                descriptions += "\n"
        return descriptions

    def _get_all_patterns_description(self) -> str:
        desc = "Available Annotation Patterns:\n\n"
        for key, info in getattr(self, "annotation_patterns", {}).items():
            desc += f"**{key}**:\n"
            if isinstance(info, dict):
                desc += f"- Description: {info.get('description', '')}\n"
                desc += f"- Formula: {info.get('formula', '')}\n"
                if info.get('keywords'):
                    desc += f"- Keywords: {', '.join(info.get('keywords', []))}\n"
            desc += "\n"
        return desc

    def create_step_by_step_examples(self, crf_item: Dict, annotation: Dict) -> List[InstructionExample]:
        """Create step-by-step instruction examples for a single CRF item"""
        examples = []
        
        # Extract field data
        field_data = self.build_field_data(crf_item, annotation)
        
        # Extract and validate pattern
        pattern = self.extract_pattern_from_annotation(annotation)
        if not pattern or pattern not in ["plain", "direct", "conditional", "findings", "supplemental", "not_submitted"]:
            logger.debug(f"Skipping non-standard pattern: {annotation.get('pattern', '')}")
            return examples
        
        # Step 1: Domain Selection
        domain = annotation.get("domain")
        if domain and domain != "Multiple":
            domain_example = self.create_domain_selection_example(field_data, domain)
            if domain_example:
                # Propagate reference confidence into metadata
                if domain_example and 'confidence' in annotation:
                    domain_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                examples.append(domain_example)
            
            # Step 2: Pattern Selection
            pattern = self.extract_pattern_from_annotation(annotation)
            if pattern:
                pattern_example = self.create_pattern_selection_example(field_data, domain, pattern)
                if pattern_example:
                    if 'confidence' in annotation:
                        pattern_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                    examples.append(pattern_example)
                
                # Step 3: Variable/Value Selection based on pattern
                if pattern == "findings":
                    var_example = self.create_findings_variable_example(field_data, domain, annotation)
                    if var_example:
                        if 'confidence' in annotation:
                            var_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.append(var_example)
                elif pattern == "direct":
                    var_example = self.create_direct_variable_example(field_data, domain, annotation)
                    if var_example:
                        if 'confidence' in annotation:
                            var_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.append(var_example)
                elif pattern == "conditional":
                    var_example = self.create_conditional_variable_example(field_data, domain, annotation)
                    if var_example:
                        if 'confidence' in annotation:
                            var_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.append(var_example)
                elif pattern == "supplemental":
                    var_example = self.create_supplemental_variable_example(field_data, domain, annotation)
                    if var_example:
                        if 'confidence' in annotation:
                            var_example.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.append(var_example)
                # Additional slot examples for complex patterns
                # variable_with_ct: add CT value selection prompts if we can parse VAR = VALUE pairs
                if "variable_with_ct" in annotation.get("pattern", ""):
                    slot_examples = self.create_variable_with_ct_examples(field_data, domain, annotation)
                    if slot_examples:
                        for ex in slot_examples:
                            if 'confidence' in annotation:
                                ex.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.extend(slot_examples)
                # conditional population: add condition var + CT selection if detectable
                if ("conditional" in annotation.get("pattern", "")) or (" when " in annotation.get("annotation", "")):
                    cond_examples = self.create_conditional_population_examples(field_data, domain, annotation)
                    if cond_examples:
                        for ex in cond_examples:
                            if 'confidence' in annotation:
                                ex.metadata['confidence'] = annotation.get('confidence', 1.0)
                        examples.extend(cond_examples)
        
        return examples

    def build_field_data(self, crf_item: Dict, annotation: Dict) -> Dict[str, Any]:
        """Build field data from CRF item"""
        # Find question and inputs
        question_text = crf_item.get("text", "")
        form = crf_item.get("form", "")
        section = crf_item.get("section", "")
        
        # Extract options if this is a checkbox/radio field
        options = []
        has_units = False
        
        # Simple heuristic for units
        if any(unit in question_text.lower() for unit in ["cm", "mg", "ml", "mmhg", "°c", "°f"]):
            has_units = True
        
        # Check if it's yes/no from annotation
        if "Option 'Yes'" in annotation.get("annotation", "") or "Option 'No'" in annotation.get("annotation", ""):
            options = ["Yes", "No"]
        
        return {
            "label": question_text,
            "form_name": form,
            "section": section,
            "options": options,
            "has_units": has_units,
            "control_type": "checkbox" if options else "text"
        }

    def create_domain_selection_example(self, field_data: Dict, target_domain: str) -> Optional[InstructionExample]:
        """Create domain selection example aligned with unified mapper prompts."""
        all_domains_text = self._get_all_domain_descriptions()
        system_prompt = (
            "You are an SDTM domain selection expert. Select the most appropriate SDTM domain for the given CRF field.\n\n"
            "IMPORTANT: You must select from the available domains listed below. Consider:\n"
            "- The field content and medical context\n"
            "- The form name and section (especially for Inclusion/Exclusion criteria → IE domain)\n"
            "- Whether it's demographic, event, intervention, or finding data\n\n"
            "Return ONLY the domain code (e.g., DM, AE, VS, IE, etc.)."
        )
        user_prompt = (
            f"Select the SDTM domain for this CRF field:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Form: {field_data.get('form_name','')}\n"
            f"Section: {field_data.get('section','')}\n"
            f"Field Type: {field_data.get('control_type','')}\n"
            f"Options: {field_data.get('options', [])}\n\n"
            f"{all_domains_text}\n"
            f"Based on the field information above, which SDTM domain should this field map to?\n"
            f"Return only the domain code."
        )
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=target_domain,
            metadata={
                "step": "domain_selection",
                "field_label": field_data['label']
            }
        )

    def create_pattern_selection_example(self, field_data: Dict, domain: str, target_pattern: str) -> Optional[InstructionExample]:
        """Create pattern selection example aligned with unified mapper prompts."""
        # Map reference pattern to a name unified mapper accepts reliably
        mapping = {
            "direct": "direct",
            "findings": "test measurement",  # will map to test_measurement
            "conditional": "direct",          # conservative fallback
            "supplemental": "supplemental",
            "not_submitted": "not_submitted",
        }
        standardized = mapping.get(target_pattern, target_pattern)

        pattern_descriptions = self._get_all_patterns_description()
        system_prompt = (
            f"You are an SDTM annotation pattern expert. The domain {domain} has been selected.\n"
            f"Now select the appropriate annotation pattern based on the field characteristics.\n\n"
            f"Consider:\n"
            f"- Field type (text, radio, checkbox, date)\n"
            f"- Whether it has \"other, specify\" text\n"
            f"- Whether it's a measurement/test (for findings domains)\n"
            f"- Whether it maps to multiple domains\n"
            f"- Whether it needs supplemental qualifiers\n\n"
            f"{pattern_descriptions}\n"
            f"Return ONLY the pattern name from the list above."
        )
        user_prompt = (
            f"Select the annotation pattern for this field in domain {domain}:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Field Type: {field_data.get('control_type','')}\n"
            f"Options: {field_data.get('options', [])}\n"
            f"Has Units: {field_data.get('has_units', False)}\n\n"
            f"Context:\n"
            f"- Contains \"other specify\"? {'yes' if ('other' in field_data.get('label','').lower() and 'specify' in field_data.get('label','').lower()) else 'no'}\n"
            f"- Is measurement/test? {'yes' if domain in ['VS','LB','EG','PE','QS'] else 'no'}\n"
            f"- Is checkbox with multiple options? {'yes' if field_data.get('control_type') == 'checkbox' else 'no'}\n\n"
            f"Which pattern should be used?"
        )
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=standardized,
            metadata={
                "step": "pattern_selection",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def create_direct_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create direct variable selection example using unified mapper prompt."""
        # Build prompt listing variables by role
        domain_vars = self.variables.get(domain, {})
        by_role = {}
        for var_name, var_info in domain_vars.items():
            by_role.setdefault(var_info.get('role', 'Other'), []).append((var_name, var_info))

        system_prompt = (
            f"Select the appropriate SDTM variable from {domain} domain.\n\n"
            f"Pattern: Direct mapping <Domain>.<Variable>\n"
            f"You must select ONE variable from the list below.\n\n"
            f"Available {domain} Variables:"
        )
        for role in ['Identifier', 'Topic', 'Qualifier', 'Timing', 'Result Qualifier', 'Other', 'Grouping Qualifier']:
            if role in by_role:
                system_prompt += f"\n\n{role} Variables:"
                for vname, vinfo in by_role[role]:
                    system_prompt += f"\n- {vname}: {vinfo.get('label','')}"
                    if vinfo.get('codelist'):
                        system_prompt += f" [CT: {vinfo['codelist']}]"
        system_prompt += "\n\nReturn ONLY the variable name."

        user_prompt = (
            f"Field: {field_data.get('label','')}\n"
            f"Type: {field_data.get('control_type','')}\n"
            f"Options: {field_data.get('options', [])}\n\n"
            f"Select the most appropriate variable."
        )

        # Heuristically extract the first variable-like token from annotation text
        ann_text = annotation.get("annotation", "")
        import re
        tokens = re.findall(r"\b([A-Z]{2,}[A-Z0-9]{0,})\b", ann_text)
        output_var = None
        for t in tokens:
            if t not in {"WHEN", "TESTCD", "ORRES", "ORRESU", "DTC"}:
                output_var = t
                break
        if not output_var and domain_vars:
            output_var = next(iter(domain_vars.keys()))
        if not output_var:
            return None

        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=output_var,
            metadata={
                "step": "variable_selection",
                "pattern": "direct",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def create_findings_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create findings example using unified mapper TESTCODE/HAS_UNITS format."""
        ann_text = annotation.get("annotation", "")
        
        # Extract TESTCD value from annotation
        testcd_value = None
        if "TESTCD =" in ann_text:
            testcd_value = ann_text.split("TESTCD =")[1].split()[0].strip()
        
        if not testcd_value:
            return None
        # Build available test codes list to mirror inference guidance (top 15)
        tc_terms = self._ct_terms_for_variable(domain, f"{domain}TESTCD")
        tc_desc = ""
        if tc_terms:
            tc_desc = "Available test codes:\n" + "\n".join(
                [f"- {t.get('submissionValue') or t.get('code')}: {t.get('preferredTerm') or t.get('value','')}" for t in tc_terms[:15]]
            )

        system_prompt = (
            f"You are selecting test codes for a findings domain {domain}.\n"
            f"Select the appropriate test code and specify if units are needed.\n\n"
            f"Return format:\nTESTCODE: <code>\nHAS_UNITS: <yes/no>"
        )
        user_prompt = (
            f"Select test code for this measurement:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Has Units: {field_data.get('has_units', False)}\n\n"
            f"{tc_desc}\n\nWhich test code should be used?"
        )
        has_units_flag = "yes" if field_data.get('has_units', False) else "no"
        output = f"TESTCODE: {testcd_value}\nHAS_UNITS: {has_units_flag}"

        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=output,
            metadata={
                "step": "testcode_selection",
                "pattern": "findings",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def create_conditional_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create variable selection example for conditional pattern (IE domain)"""
        if domain != "IE":
            return None
        
        ann_text = annotation.get("annotation", "")
        
        # Extract IETESTCD value
        testcd_value = None
        if "IETESTCD =" in ann_text:
            testcd_value = ann_text.split("IETESTCD =")[1].split()[0].strip()
        
        if not testcd_value:
            return None
        
        system_prompt = (
            "You are annotating inclusion/exclusion criteria for the IE domain.\n\n"
            "For IE domain:\n"
            "- Each criterion gets a unique IETESTCD (e.g., IN01, IN02 for inclusions, EX01, EX02 for exclusions)\n"
            "- The criterion code usually appears in the field label in parentheses\n"
            "- IETEST contains the criterion description\n"
            "- IEORRES contains the response (Y/N)"
        )
        user_prompt = (
            f"Extract the criterion code for:\n"
            f"Field Label: {field_data.get('label','')}\n\n"
            f"Return the IETESTCD value (e.g., IN01, EX01)."
        )
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=testcd_value,
            metadata={
                "step": "criterion_code_extraction",
                "pattern": "conditional",
                "domain": "IE",
                "field_label": field_data['label']
            }
        )

    # ===== New: variable_with_ct slot examples =====
    def _extract_var_equals_pairs(self, ann_text: str) -> List[Tuple[str, str]]:
        """Extract VAR = VALUE pairs from annotation text."""
        pairs: List[Tuple[str, str]] = []
        for part in ann_text.split('|'):
            part = part.strip()
            if '=' in part:
                var, val = part.split('=', 1)
                var = var.strip().upper().replace('.', '')
                val = val.strip().strip("'\"")
                # Skip WHEN clauses
                if var and not var.startswith('WHEN'):
                    pairs.append((var, val))
        return pairs

    def _ct_terms_for_variable(self, domain: str, variable: str) -> List[Dict[str, str]]:
        """Lookup CT terms for a given domain.variable using KB cdisc_ct.json via variable->codelist mapping."""
        # Get codelist code from proto variables (variables_all.json style)
        terms: List[Dict[str, str]] = []
        try:
            # variables_all.json format
            vars_list = []
            kb_vars_path = self.kb_path / "variables_all.json"
            if kb_vars_path.exists():
                import json as _json
                vars_all = _json.load(open(kb_vars_path))
                vars_list = [v for v in vars_all if v.get('domain') == domain]
                for v in vars_list:
                    if v.get('name') == variable and v.get('codelist'):
                        cl_code = v['codelist'].get('code')
                        # Find codelist in cdisc_ct.json
                        ct_path = self.kb_path / "cdisc_ct.json"
                        if ct_path.exists():
                            ct = _json.load(open(ct_path))
                            for cl in ct.get('codelists', []):
                                c = cl.get('codelist', {})
                                if c.get('conceptId') == cl_code or c.get('href', '').endswith(cl_code) or c.get('code') == cl_code:
                                    return cl.get('terms', [])
        except Exception:
            pass
        return terms

    def create_variable_with_ct_examples(self, field_data: Dict, domain: str, annotation: Dict) -> List[InstructionExample]:
        """Create CT value selection examples for variable_with_ct pattern."""
        out: List[InstructionExample] = []
        ann_text = annotation.get("annotation", "")
        pairs = self._extract_var_equals_pairs(ann_text)
        for var, value in pairs:
            # Build CT selection prompt like unified mapper
            ct_terms = self._ct_terms_for_variable(domain, var)
            system_prompt = (
                f"You are selecting a controlled terminology value for SDTM variable {var}.\n\n"
                f"Variable: {var}\nDomain: {domain}\n\nALL Available Controlled Terminology Values:"
            )
            # Show up to 20 terms
            for i, t in enumerate(ct_terms[:20]):
                code = t.get('submissionValue') or t.get('code') or ''
                term = t.get('preferredTerm') or t.get('value') or ''
                definition = t.get('definition', '')
                system_prompt += f"\n{i+1}. {code}: {term}"
                if definition:
                    system_prompt += f"\n   Definition: {definition[:140]}"
            user_prompt = (
                f"Select CT value for:\nQuestion Text: {field_data.get('label','')}\n"
                f"Variable: {var}\nDomain: {domain}\n\nReturn ONLY the submission value."
            )
            # Try to map provided value to submissionValue if ct available
            target = value
            if ct_terms:
                low = value.lower()
                for t in ct_terms:
                    sub = t.get('submissionValue', '')
                    term = t.get('preferredTerm', t.get('value', ''))
                    if low == sub.lower() or low == term.lower() or low in term.lower():
                        target = sub
                        break
            out.append(InstructionExample(
                instruction=system_prompt,
                input=user_prompt,
                output=target,
                metadata={
                    "step": "ct_value_selection",
                    "pattern": "variable_with_ct",
                    "domain": domain,
                    "variable": var,
                    "field_label": field_data['label']
                }
            ))
        return out

    # ===== New: conditional population slot examples =====
    def create_conditional_population_examples(self, field_data: Dict, domain: str, annotation: Dict) -> List[InstructionExample]:
        """Create slot prompts for conditional population: select condition variable and its CT value."""
        ann_text = annotation.get("annotation", "")
        if " when " not in ann_text:
            return []
        # Parse pattern: <...> when <CONDVAR> = <VALUE>
        cond_part = ann_text.split(" when ", 1)[1]
        cond_var = None
        cond_val = None
        if "=" in cond_part:
            left, right = cond_part.split("=", 1)
            cond_var = left.strip().upper().replace('.', '')
            cond_val = right.strip().split("|")[0].strip().strip("'\"")
        if not cond_var or not cond_val:
            return []
        examples: List[InstructionExample] = []
        # A) Condition variable selection (from domain vars)
        # Build a direct-style variable selection prompt but mention it is condition variable
        # Gather domain vars
        domain_vars = self.variables.get(domain, {})
        by_role = {}
        for var_name, var_info in domain_vars.items():
            by_role.setdefault(var_info.get('role', 'Other'), []).append((var_name, var_info))
        sys_var = (
            f"Select the condition SDTM variable from {domain} domain.\n\n"
            f"You must select ONE variable from the list below.\n\n"
            f"Available {domain} Variables:"
        )
        for role in ['Identifier', 'Topic', 'Qualifier', 'Timing', 'Result Qualifier', 'Other', 'Grouping Qualifier']:
            if role in by_role:
                sys_var += f"\n\n{role} Variables:"
                for vname, vinfo in by_role[role]:
                    sys_var += f"\n- {vname}: {vinfo.get('label','')}"
        sys_var += "\n\nReturn ONLY the variable name."
        user_var = (
            f"Field: {field_data.get('label','')}\n"
            f"Select the condition variable used in the 'when' clause."
        )
        examples.append(InstructionExample(
            instruction=sys_var,
            input=user_var,
            output=cond_var,
            metadata={
                "step": "condition_variable_selection",
                "pattern": "conditional_population",
                "domain": domain,
                "field_label": field_data['label']
            }
        ))
        # B) CT value selection for the condition variable
        ct_terms = self._ct_terms_for_variable(domain, cond_var)
        sys_ct = (
            f"You are selecting a controlled terminology value for SDTM variable {cond_var}.\n\n"
            f"Variable: {cond_var}\nDomain: {domain}\n\nALL Available Controlled Terminology Values:"
        )
        for i, t in enumerate(ct_terms[:20]):
            code = t.get('submissionValue') or t.get('code') or ''
            term = t.get('preferredTerm') or t.get('value') or ''
            definition = t.get('definition', '')
            sys_ct += f"\n{i+1}. {code}: {term}"
            if definition:
                sys_ct += f"\n   Definition: {definition[:140]}"
        usr_ct = (
            f"Select CT value for:\nQuestion Text: {field_data.get('label','')}\n"
            f"Variable: {cond_var}\nDomain: {domain}\n\nReturn ONLY the submission value."
        )
        target_val = cond_val
        if ct_terms:
            low = cond_val.lower()
            for t in ct_terms:
                sub = t.get('submissionValue', '')
                term = t.get('preferredTerm', t.get('value', ''))
                if low == sub.lower() or low == term.lower() or low in term.lower():
                    target_val = sub
                    break
        examples.append(InstructionExample(
            instruction=sys_ct,
            input=usr_ct,
            output=target_val,
            metadata={
                "step": "condition_ct_selection",
                "pattern": "conditional_population",
                "domain": domain,
                "field_label": field_data['label'],
                "variable": cond_var
            }
        ))
        return examples

    def create_supplemental_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create QNAM selection example aligned with unified mapper prompts."""
        ann_text = annotation.get("annotation", "")
        
        # Extract QNAM
        qnam = None
        if " in SUPP" in ann_text:
            qnam = ann_text.split(" in SUPP")[0].strip()
        
        if not qnam:
            return None
        system_prompt = (
            f"You are creating a supplemental qualifier (QNAM) for domain {domain}.\n"
            f"Create an 8-character uppercase identifier based on the field label.\n\n"
            f"Rules:\n"
            f"- Maximum 8 characters\n"
            f"- All uppercase\n"
            f"- No spaces or special characters\n"
            f"- Should be descriptive of the field content\n\n"
            f"Return format:\nQNAM: <identifier>"
        )
        user_prompt = (
            f"Create QNAM for this supplemental field:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Domain: {domain}\n\n"
            f"What should the QNAM be?"
        )
        output = f"QNAM: {qnam}"

        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=output,
            metadata={
                "step": "qnam_selection", 
                "pattern": "supplemental",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def extract_pattern_from_annotation(self, annotation: Dict) -> Optional[str]:
        """Extract pattern type from annotation"""
        pattern = annotation.get("pattern", "")
        
        # Handle comma-separated patterns
        if "," in pattern:
            patterns = [p.strip() for p in pattern.split(",")]
            # Prioritize main patterns
            for p in ["findings", "conditional", "supplemental", "direct", "plain"]:
                for pat in patterns:
                    if p in pat or f"{p}_single" in pat:
                        return p
            # Special handling for conditional_single
            if "conditional_single" in patterns:
                return "conditional"
            # Default to first pattern
            return patterns[0].replace("_single", "")
        
        # Remove _single suffix
        if pattern.endswith("_single"):
            pattern = pattern[:-7]
        
        # Map patterns
        pattern_mapping = {
            "conditional_single": "conditional",
            "direct": "plain",  # Map direct to plain
            "not_submitted": "not_submitted"
        }
        
        return pattern_mapping.get(pattern, pattern) if pattern else None

    def process_reference_files(self, reference_dir: str, crf_dir: str) -> List[InstructionExample]:
        """Process all reference files and create instruction examples"""
        all_examples = []
        reference_path = Path(reference_dir)
        crf_path = Path(crf_dir)
        
        # Process each result file
        for result_file in sorted(reference_path.glob("page_*.json")):
            if "Zone.Identifier" in str(result_file):
                continue
                
            logger.info(f"Processing {result_file.name}")
            
            with open(result_file) as f:
                result_data = json.load(f)
            
            # Try to load corresponding CRF file (optional)
            page_num = result_file.name.split("_")[1].split(".")[0]
            crf_file = crf_path / f"page_{page_num}.json"
            crf_data = None
            text_to_item = {}
            if crf_file.exists():
                try:
                    with open(crf_file) as f:
                        crf_data = json.load(f)
                    for item in crf_data.get("items", []):
                        if item.get("tag") == "<Q>":
                            text = item.get("text", "").strip()
                            if text:
                                text_to_item[text] = item
                except Exception as e:
                    logger.warning(f"Failed to read CRF file {crf_file}: {e}
")
            
            # Process annotations
            for annotation in result_data.get("annotations", []):
                ann_text = annotation.get("text", "").strip()
                
                # Try to find matching CRF item by text, else fallback to reference-only
                crf_item = None
                if text_to_item:
                    if ann_text in text_to_item:
                        crf_item = text_to_item[ann_text]
                    else:
                        for crf_text, item in text_to_item.items():
                            if ann_text in crf_text or crf_text in ann_text:
                                crf_item = item
                                break
                if not crf_item:
                    # Fallback: build minimal CRF-like item from reference
                    crf_item = {
                        "text": ann_text,
                        "form": result_data.get("file", "") or "",
                        "section": result_data.get("summary", {}).get("section", ""),
                        "tag": "<Q>",
                    }
                examples = self.create_step_by_step_examples(crf_item, annotation)
                all_examples.extend(examples)
        
        logger.info(f"Created {len(all_examples)} instruction examples")
        return all_examples

    def save_dataset(self, examples: List[InstructionExample], output_path: str):
        """Save dataset in different formats"""
        output_path = Path(output_path)
        
        # Save as JSON for general use
        json_data = [asdict(ex) for ex in examples]
        with open(output_path / "instruction_dataset.json", "w") as f:
            json.dump(json_data, f, indent=2)
        
        # Save as JSONL for training
        with open(output_path / "instruction_dataset.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")
        
        # Save in Alpaca format
        alpaca_data = []
        for ex in examples:
            alpaca_data.append({
                "instruction": ex.instruction,
                "input": ex.input,
                "output": ex.output
            })
        
        with open(output_path / "alpaca_format.json", "w") as f:
            json.dump(alpaca_data, f, indent=2)
        
        # Save metadata summary
        metadata_summary = {
            "total_examples": len(examples),
            "examples_by_step": {},
            "examples_by_domain": {},
            "examples_by_pattern": {}
        }
        
        for ex in examples:
            step = ex.metadata.get("step", "unknown")
            domain = ex.metadata.get("domain", "unknown")
            pattern = ex.metadata.get("pattern", "unknown")
            
            metadata_summary["examples_by_step"][step] = metadata_summary["examples_by_step"].get(step, 0) + 1
            metadata_summary["examples_by_domain"][domain] = metadata_summary["examples_by_domain"].get(domain, 0) + 1
            metadata_summary["examples_by_pattern"][pattern] = metadata_summary["examples_by_pattern"].get(pattern, 0) + 1
        
        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(metadata_summary, f, indent=2)


def main():
    # Defaults - relative to package directory (works in cloud and locally)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    default_kb = script_dir / "kb" / "sdtmig_v3_4_complete"
    # Prefer local packaged reference if present, else user's Inari path, else repo fallback
    local_ref = script_dir / "data" / "reference"
    inari_ref = Path("/home/kejunzou/Projects/Oss+MinerU ACRF/data/data/sample_crfs/Inari Reference/all_results")
    if local_ref.exists():
        default_reference = local_ref
    elif inari_ref.exists():
        default_reference = inari_ref
    else:
        default_reference = repo_root / "reference_with_sections"
    default_crf = repo_root / "crf_json"
    default_out = script_dir / "data"

    ap = argparse.ArgumentParser(description="Build SDTM instruction-tuning dataset from reference annotations")
    ap.add_argument("--kb-path", default=str(default_kb), help="Knowledge base directory (proto_define, domains_by_class, etc.)")
    ap.add_argument("--reference-dir", default=str(default_reference), help="Directory of reference annotations (e.g., page_*_result.json)")
    ap.add_argument("--crf-dir", default=str(default_crf), help="Directory of CRF page_*.json files")
    ap.add_argument("--output-dir", default=str(default_out), help="Directory to write alpaca/instruction datasets")
    args = ap.parse_args()

    kb_path = Path(args.kb_path)
    reference_dir = Path(args.reference_dir)
    crf_dir = Path(args.crf_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log effective paths
    logger.info(f"KB: {kb_path}")
    logger.info(f"Reference: {reference_dir}")
    logger.info(f"CRF: {crf_dir}")
    logger.info(f"Output: {output_dir}")

    # Build dataset
    builder = InstructionDatasetBuilder(str(kb_path))
    examples = builder.process_reference_files(str(reference_dir), str(crf_dir))

    # Save dataset
    builder.save_dataset(examples, str(output_dir))

    print(f"Dataset created with {len(examples)} examples")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
