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
        
        # Load controlled terminology
        ct_path = self.kb_path / "cdisc_ct.json"
        if ct_path.exists():
            with open(ct_path) as f:
                self.ct_data = json.load(f)

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
                examples.append(domain_example)
            
            # Step 2: Pattern Selection
            pattern = self.extract_pattern_from_annotation(annotation)
            if pattern:
                pattern_example = self.create_pattern_selection_example(field_data, domain, pattern)
                if pattern_example:
                    examples.append(pattern_example)
                
                # Step 3: Variable/Value Selection based on pattern
                if pattern == "findings":
                    var_example = self.create_findings_variable_example(field_data, domain, annotation)
                    if var_example:
                        examples.append(var_example)
                elif pattern == "direct":
                    var_example = self.create_direct_variable_example(field_data, domain, annotation)
                    if var_example:
                        examples.append(var_example)
                elif pattern == "conditional":
                    var_example = self.create_conditional_variable_example(field_data, domain, annotation)
                    if var_example:
                        examples.append(var_example)
                elif pattern == "supplemental":
                    var_example = self.create_supplemental_variable_example(field_data, domain, annotation)
                    if var_example:
                        examples.append(var_example)
        
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
        """Create domain selection instruction example"""
        # Build domain list for the prompt
        all_domains = []
        for class_name, domains in self.domains_by_class.items():
            for domain_info in domains:
                all_domains.append(domain_info)
        
        # Sort domains to put target first for variety
        random.shuffle(all_domains)
        
        # Build system prompt
        system_prompt = """You are helping select the appropriate SDTM domain for CRF fields.
        
Available SDTM Domains:
"""
        for domain_info in all_domains[:20]:  # Limit to top 20 for brevity
            code = domain_info.get('code', 'Unknown')
            label = domain_info.get('label', domain_info.get('name', ''))
            definition = domain_info.get('definition', domain_info.get('description', ''))
            system_prompt += f"\n{code} - {label}"
            if definition:
                system_prompt += f"\n  Definition: {definition[:100]}..."
        
        system_prompt += "\n\nSelect the most appropriate domain code."
        
        # Build user prompt
        user_prompt = f"""Select the SDTM domain for this field:
Field Label: {field_data['label']}
Form: {field_data['form_name']}
Section: {field_data.get('section', 'N/A')}

Return only the domain code."""
        
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
        """Create pattern selection instruction example"""
        # Map annotation patterns to our standard patterns
        pattern_map = {
            "direct": "plain",
            "conditional": "conditional",
            "conditional_single": "conditional",
            "findings": "findings",
            "supplemental": "supplemental",
            "not_submitted": "not_submitted"
        }
        
        standard_pattern = pattern_map.get(target_pattern, target_pattern)
        
        system_prompt = f"""You are selecting the annotation pattern for domain {domain}.

Available patterns:
1. plain - Direct mapping to a single variable or multiple variables
2. findings - For test results with TESTCD (used in Findings class domains)
3. conditional - For criteria that map based on conditions
4. supplemental - For non-standard variables that go to SUPP domain
5. not_submitted - For fields not submitted to SDTM

Select the appropriate pattern based on the field characteristics."""
        
        user_prompt = f"""Select the annotation pattern for:
Domain: {domain}
Field Label: {field_data['label']}
Has Options: {bool(field_data.get('options'))}
Has Units: {field_data.get('has_units', False)}

Return only the pattern name."""
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=standard_pattern,
            metadata={
                "step": "pattern_selection",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def create_direct_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create variable selection example for direct pattern"""
        # Extract variables from annotation
        ann_text = annotation.get("annotation", "")
        
        # Get domain variables
        domain_vars = self.variables.get(domain, {})
        
        # Build system prompt with available variables
        system_prompt = f"""You are selecting SDTM variables for direct mapping in domain {domain}.

Available variables in {domain}:
"""
        for var_name, var_info in list(domain_vars.items())[:15]:  # Limit for brevity
            system_prompt += f"\n{var_name}: {var_info.get('label', '')}"
            if var_info.get('role'):
                system_prompt += f" (Role: {var_info['role']})"
        
        system_prompt += "\n\nSelect the appropriate variable(s) for this field."
        
        user_prompt = f"""Select variables for:
Domain: {domain}  
Field Label: {field_data['label']}
Pattern: direct mapping

Return the variable name(s) separated by '/' if multiple."""
        
        # Extract expected output from annotation
        output = ann_text.split(",")[0].strip() if "," in ann_text else ann_text.strip()
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=output,
            metadata={
                "step": "variable_selection",
                "pattern": "direct",
                "domain": domain,
                "field_label": field_data['label']
            }
        )

    def create_findings_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create variable selection example for findings pattern"""
        ann_text = annotation.get("annotation", "")
        
        # Extract TESTCD value from annotation
        testcd_value = None
        if "TESTCD =" in ann_text:
            testcd_value = ann_text.split("TESTCD =")[1].split()[0].strip()
        
        if not testcd_value:
            return None
        
        system_prompt = f"""You are selecting a test code for findings pattern in domain {domain}.

For findings domains, you need to select:
1. The appropriate TESTCD value that identifies what is being measured
2. Whether units (ORRESU) are needed

Common test codes should be short, uppercase identifiers."""
        
        user_prompt = f"""Select the test code for:
Domain: {domain}
Field Label: {field_data['label']}
Has Units: {field_data.get('has_units', False)}

Return the TESTCD value."""
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=testcd_value,
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
        
        system_prompt = """You are annotating inclusion/exclusion criteria for the IE domain.

For IE domain:
- Each criterion gets a unique IETESTCD (e.g., IN01, IN02 for inclusions, EX01, EX02 for exclusions)
- The criterion code usually appears in the field label in parentheses
- IETEST contains the criterion description
- IEORRES contains the response (Y/N)"""
        
        user_prompt = f"""Extract the criterion code for:
Field Label: {field_data['label']}

Return the IETESTCD value (e.g., IN01, EX01)."""
        
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

    def create_supplemental_variable_example(self, field_data: Dict, domain: str, annotation: Dict) -> Optional[InstructionExample]:
        """Create QNAM selection example for supplemental pattern"""
        ann_text = annotation.get("annotation", "")
        
        # Extract QNAM
        qnam = None
        if " in SUPP" in ann_text:
            qnam = ann_text.split(" in SUPP")[0].strip()
        
        if not qnam:
            return None
        
        system_prompt = f"""You are creating a supplemental qualifier name (QNAM) for domain {domain}.

QNAMs should be:
- 8 characters or less
- Uppercase
- Descriptive of what is being captured
- Common QNAMs include: {domain}OTH, {domain}OTHSP (for 'other specify' fields)"""
        
        user_prompt = f"""Create QNAM for:
Domain: {domain}
Field Label: {field_data['label']}

Return the QNAM (8 chars max)."""
        
        return InstructionExample(
            instruction=system_prompt,
            input=user_prompt,
            output=qnam,
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
            
            # Load corresponding CRF file
            page_num = result_file.name.split("_")[1].split(".")[0]
            crf_file = crf_path / f"page_{page_num}.json"
            
            if not crf_file.exists():
                logger.warning(f"CRF file not found: {crf_file}")
                continue
            
            with open(crf_file) as f:
                crf_data = json.load(f)
            
            # Create mapping of text to CRF items (use text matching instead of qid)
            text_to_item = {}
            for item in crf_data.get("items", []):
                if item.get("tag") == "<Q>":
                    text = item.get("text", "").strip()
                    if text:
                        text_to_item[text] = item
            
            # Process annotations
            for annotation in result_data.get("annotations", []):
                ann_text = annotation.get("text", "").strip()
                
                # Try to find matching CRF item by text
                crf_item = None
                if ann_text in text_to_item:
                    crf_item = text_to_item[ann_text]
                else:
                    # Try partial matching if exact match fails
                    for crf_text, item in text_to_item.items():
                        if ann_text in crf_text or crf_text in ann_text:
                            crf_item = item
                            break
                
                if crf_item:
                    examples = self.create_step_by_step_examples(crf_item, annotation)
                    all_examples.extend(examples)
                else:
                    logger.debug(f"No CRF match for annotation: {ann_text[:50]}...")
        
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
    # Defaults - relative to package directory
    script_dir = Path(__file__).parent
    default_kb = script_dir / "kb" / "sdtmig_v3_4_complete"
    default_reference = script_dir / "data" / "reference"
    default_crf = script_dir / "data" / "crf_json"
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
