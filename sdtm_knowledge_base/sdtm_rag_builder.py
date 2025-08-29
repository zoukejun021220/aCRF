#!/usr/bin/env python3
"""
SDTM RAG Knowledge Base Builder
Structures knowledge base following Domain → Variable → Definition → Codelist → Patterns
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import re
from sdtm_annotation_rules import SDTMAnnotationRules, AnnotationType, create_annotation_rules

logger = logging.getLogger(__name__)


@dataclass
class SDTMVariableEntry:
    """Structured entry for SDTM variable following SDTMIG pattern"""
    domain: str
    variable: str
    full_name: str  # DOMAIN.VARIABLE
    label: str
    definition: str
    role: str
    type: str
    core: str
    codelist: Optional[str] = None
    codelist_values: Optional[List[Dict[str, str]]] = None
    aliases: List[str] = None
    crf_synonyms: List[str] = None
    when_then_patterns: List[Dict[str, str]] = None
    examples: List[str] = None
    context: Optional[str] = None
    
    def __post_init__(self):
        self.full_name = f"{self.domain}.{self.variable}"
        if self.aliases is None:
            self.aliases = []
        if self.crf_synonyms is None:
            self.crf_synonyms = []
        if self.when_then_patterns is None:
            self.when_then_patterns = []
        if self.examples is None:
            self.examples = []


class SDTMRAGBuilder:
    """Build SDTM knowledge base optimized for RAG retrieval"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.kb_entries: Dict[str, SDTMVariableEntry] = {}
        self.domain_index: Dict[str, List[str]] = {}
        self.alias_index: Dict[str, List[str]] = {}
        self.pattern_index: Dict[str, List[str]] = {}
        self.annotation_rules = create_annotation_rules()
        
    def load_sdtm_data(self):
        """Load SDTM data from existing JSON files"""
        
        # Load controlled terminology
        ct_path = self.base_path / "STDM controlled terminology" / "sdtm_controlled_terminology.json"
        with open(ct_path) as f:
            ct_data = json.load(f)
            
        # Load variable definitions
        var_def_path = self.base_path / "sdtm_variable_definitions.json"
        with open(var_def_path) as f:
            var_defs = json.load(f)
            
        # Load domain info
        domain_info_path = self.base_path / "sdtm_domain_info.json"
        with open(domain_info_path) as f:
            domain_info = json.load(f)
            
        # Load annotation instructions if available
        self.annotation_instructions = None
        instructions_path = self.base_path / "annotation_instructions.md"
        if instructions_path.exists():
            with open(instructions_path) as f:
                self.annotation_instructions = f.read()
                logger.info("Loaded annotation instructions")
            
        # Process domains and variables
        for domain_code, domain_data in ct_data.get("domains", {}).items():
            for var_name, var_data in domain_data.get("variables", {}).items():
                self._create_variable_entry(
                    domain_code, var_name, var_data, ct_data, var_defs
                )
                
    def _create_variable_entry(
        self, 
        domain: str, 
        variable: str, 
        var_data: Dict,
        ct_data: Dict,
        var_defs: Dict
    ) -> SDTMVariableEntry:
        """Create structured variable entry"""
        
        full_name = f"{domain}.{variable}"
        
        # Get additional definitions
        additional_def = var_defs.get(full_name, {})
        
        # Create base entry
        entry = SDTMVariableEntry(
            domain=domain,
            variable=variable,
            label=var_data.get("label", ""),
            definition=var_data.get("description", additional_def.get("description", "")),
            role=var_data.get("role", ""),
            type=var_data.get("type", ""),
            core=var_data.get("core", ""),
            codelist=var_data.get("codelist")
        )
        
        # Add CRF synonyms based on common patterns
        entry.crf_synonyms = self._generate_crf_synonyms(entry)
        
        # Add when/then patterns
        entry.when_then_patterns = self._generate_when_then_patterns(entry)
        
        # Store entry
        self.kb_entries[full_name] = entry
        
        # Update indices
        if domain not in self.domain_index:
            self.domain_index[domain] = []
        self.domain_index[domain].append(full_name)
        
        # Index aliases and synonyms
        for synonym in entry.crf_synonyms:
            synonym_lower = synonym.lower()
            if synonym_lower not in self.alias_index:
                self.alias_index[synonym_lower] = []
            self.alias_index[synonym_lower].append(full_name)
            
        return entry
        
    def _generate_crf_synonyms(self, entry: SDTMVariableEntry) -> List[str]:
        """Generate common CRF synonyms for a variable"""
        
        synonyms = []
        
        # Common mappings for specific variables
        synonym_map = {
            "BRTHDTC": ["Date of Birth", "Birth Date", "DOB", "Birthdate"],
            "SEX": ["Gender", "Sex", "M/F", "Male/Female"],
            "RACE": ["Race", "Ethnicity", "Racial Background"],
            "AESEV": ["Severity", "Grade", "Severity/Intensity", "AE Severity"],
            "AESER": ["Serious", "Is Serious?", "Serious AE", "SAE"],
            "AESTDTC": ["Start Date", "Onset Date", "AE Start Date", "Date of Onset"],
            "AEENDTC": ["End Date", "Resolution Date", "AE End Date", "Date Resolved"],
            "AETERM": ["Adverse Event", "AE Term", "Event", "AE Description"],
            "VSTESTCD": ["Vital Sign", "Test Code", "VS Test", "Parameter"],
            "VSORRES": ["Result", "Value", "Original Result", "Measurement"],
            "VSORRESU": ["Unit", "Original Unit", "Unit of Measure"],
            "LBTESTCD": ["Lab Test", "Test Code", "Laboratory Test", "Lab Parameter"],
            "LBORRES": ["Result", "Lab Result", "Original Result", "Lab Value"],
            "LBORRESU": ["Unit", "Lab Unit", "Result Unit"],
            "CMTRT": ["Medication", "Drug Name", "Concomitant Med", "Con Med"],
            "CMDOSE": ["Dose", "Dosage", "Amount"],
            "CMDOSU": ["Dose Unit", "Unit", "Dosage Unit"],
            "CMSTDTC": ["Start Date", "Med Start Date", "Date Started"],
            "CMENDTC": ["End Date", "Med End Date", "Date Stopped"],
            "EXDOSE": ["Dose", "Study Drug Dose", "Treatment Dose"],
            "EXDOSU": ["Unit", "Dose Unit", "Dosage Unit"],
            "DSSTDTC": ["Date", "Disposition Date", "Completion Date"],
            "DSTERM": ["Disposition", "Completion Status", "Study Status"],
            "MHTERM": ["Medical History", "Condition", "Past Medical History", "Medical Condition"],
            "MHSTDTC": ["Start Date", "Onset Date", "Date of Diagnosis"],
            "IETEST": ["Inclusion/Exclusion", "I/E Criteria", "Eligibility Criteria"],
            "IEORRES": ["Met?", "Yes/No", "Criteria Met", "Result"]
        }
        
        # Get synonyms for this variable
        var_key = entry.variable
        if var_key in synonym_map:
            synonyms.extend(synonym_map[var_key])
            
        # Add label variations
        if entry.label:
            synonyms.append(entry.label)
            # Add variations without "Date" suffix for date fields
            if entry.label.endswith(" Date"):
                synonyms.append(entry.label[:-5])
                
        # Domain-specific patterns
        if entry.domain == "VS" and entry.variable.startswith("VS"):
            if "TEMP" in entry.variable:
                synonyms.extend(["Temperature", "Temp", "Body Temperature"])
            elif "PULSE" in entry.variable:
                synonyms.extend(["Pulse", "Heart Rate", "HR", "Pulse Rate"])
            elif "BP" in entry.variable:
                synonyms.extend(["Blood Pressure", "BP", "Systolic/Diastolic"])
            elif "RESP" in entry.variable:
                synonyms.extend(["Respiration", "Respiratory Rate", "RR", "Breathing Rate"])
                
        return list(set(synonyms))  # Remove duplicates
        
    def _generate_when_then_patterns(self, entry: SDTMVariableEntry) -> List[Dict[str, str]]:
        """Generate when/then patterns for conditional variable usage"""
        
        patterns = []
        
        # Test-specific patterns
        if entry.variable == "VSORRES":
            patterns.append({
                "when": "VSTESTCD='TEMP'",
                "then": "Temperature measurement value",
                "example": "36.5"
            })
            patterns.append({
                "when": "VSTESTCD='PULSE'",
                "then": "Pulse rate value",
                "example": "72"
            })
            patterns.append({
                "when": "VSTESTCD='SYSBP'",
                "then": "Systolic blood pressure value",
                "example": "120"
            })
            
        elif entry.variable == "VSORRESU":
            patterns.append({
                "when": "VSTESTCD='TEMP'",
                "then": "Temperature unit",
                "example": "C"
            })
            patterns.append({
                "when": "VSTESTCD='PULSE'",
                "then": "Pulse rate unit",
                "example": "beats/min"
            })
            
        elif entry.variable == "LBORRES":
            patterns.append({
                "when": "LBTESTCD='HGB'",
                "then": "Hemoglobin result",
                "example": "14.5"
            })
            patterns.append({
                "when": "LBTESTCD='GLUC'",
                "then": "Glucose result",
                "example": "95"
            })
            
        # Domain-specific patterns
        elif entry.domain == "AE" and entry.variable in ["AESER", "AESEV"]:
            patterns.append({
                "when": "Adverse Event is captured",
                "then": f"Record {entry.label}",
                "codelist": entry.codelist
            })
            
        return patterns
        
    def build_rag_chunks(self) -> List[Dict[str, Any]]:
        """Build small, typed chunks for RAG retrieval"""
        
        chunks = []
        
        # Add annotation instructions as chunks if available
        if self.annotation_instructions:
            chunks.extend(self._create_instruction_chunks())
            
        # Add annotation rule chunks from the rules module
        chunks.extend(self._create_annotation_rule_chunks())
        
        for full_name, entry in self.kb_entries.items():
            # Create main variable chunk
            main_chunk = {
                "chunk_id": f"{full_name}_main",
                "chunk_type": "variable_definition",
                "domain": entry.domain,
                "variable": entry.variable,
                "full_name": full_name,
                "content": self._format_variable_content(entry),
                "metadata": {
                    "label": entry.label,
                    "role": entry.role,
                    "type": entry.type,
                    "core": entry.core,
                    "codelist": entry.codelist
                },
                "search_text": self._create_search_text(entry)
            }
            chunks.append(main_chunk)
            
            # Create synonym chunks
            for synonym in entry.crf_synonyms:
                syn_chunk = {
                    "chunk_id": f"{full_name}_syn_{synonym.lower().replace(' ', '_')}",
                    "chunk_type": "synonym_mapping",
                    "domain": entry.domain,
                    "variable": entry.variable,
                    "full_name": full_name,
                    "content": f'"{synonym}" maps to {full_name} ({entry.label})',
                    "metadata": {
                        "synonym": synonym,
                        "variable_ref": full_name
                    },
                    "search_text": synonym.lower()
                }
                chunks.append(syn_chunk)
                
            # Create pattern chunks
            for i, pattern in enumerate(entry.when_then_patterns):
                pattern_chunk = {
                    "chunk_id": f"{full_name}_pattern_{i}",
                    "chunk_type": "when_then_pattern",
                    "domain": entry.domain,
                    "variable": entry.variable,
                    "full_name": full_name,
                    "content": f"When {pattern['when']}, then {pattern['then']} (stored in {full_name})",
                    "metadata": {
                        "pattern": pattern,
                        "variable_ref": full_name
                    },
                    "search_text": f"{pattern['when']} {pattern['then']}"
                }
                chunks.append(pattern_chunk)
                
        return chunks
        
    def _format_variable_content(self, entry: SDTMVariableEntry) -> str:
        """Format variable entry as readable content"""
        
        content = f"{entry.full_name}: {entry.label}\n"
        content += f"Definition: {entry.definition}\n"
        content += f"Domain: {entry.domain}\n"
        content += f"Variable: {entry.variable}\n"
        content += f"Role: {entry.role}, Type: {entry.type}, Core: {entry.core}\n"
        
        if entry.codelist:
            content += f"Codelist: {entry.codelist}\n"
            
        if entry.crf_synonyms:
            content += f"Common CRF Terms: {', '.join(entry.crf_synonyms)}\n"
            
        if entry.when_then_patterns:
            content += "\nUsage Patterns:\n"
            for pattern in entry.when_then_patterns:
                content += f"  - When {pattern['when']}: {pattern['then']}\n"
                if 'example' in pattern:
                    content += f"    Example: {pattern['example']}\n"
                    
        return content
        
    def _create_search_text(self, entry: SDTMVariableEntry) -> str:
        """Create searchable text combining all relevant fields"""
        
        search_parts = [
            entry.full_name,
            entry.variable,
            entry.label,
            entry.definition
        ]
        
        # Add synonyms
        search_parts.extend(entry.crf_synonyms)
        
        # Add pattern text
        for pattern in entry.when_then_patterns:
            search_parts.append(pattern.get('when', ''))
            search_parts.append(pattern.get('then', ''))
            
        return ' '.join(filter(None, search_parts)).lower()
        
    def _create_instruction_chunks(self) -> List[Dict[str, Any]]:
        """Create chunks from annotation instructions"""
        
        chunks = []
        
        # Split instructions into sections
        sections = self.annotation_instructions.split('\n## ')
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Get section title and content
            lines = section.strip().split('\n', 1)
            title = lines[0].strip('#').strip()
            content = lines[1] if len(lines) > 1 else ""
            
            # Create instruction chunk
            chunk = {
                "chunk_id": f"instruction_{i}_{title.lower().replace(' ', '_')}",
                "chunk_type": "annotation_instruction",
                "domain": "INSTRUCTION",
                "variable": "INSTRUCTION",
                "full_name": f"INSTRUCTION.{title.upper().replace(' ', '_')}",
                "content": f"Annotation Instruction - {title}:\n{content[:500]}...",
                "metadata": {
                    "section": title,
                    "full_content": content
                },
                "search_text": f"{title} {content}".lower()
            }
            chunks.append(chunk)
            
            # Extract examples as separate chunks
            if "Examples" in title or "example" in content.lower():
                example_chunks = self._extract_example_chunks(title, content)
                chunks.extend(example_chunks)
                
        # Add special chunks for key concepts
        chunks.extend(self._create_concept_chunks())
        
        return chunks
        
    def _create_annotation_rule_chunks(self) -> List[Dict[str, Any]]:
        """Create chunks from annotation rules module"""
        
        chunks = []
        
        # Get rules summary
        rules_summary = self.annotation_rules.get_rules_summary()
        
        # Create chunk for overall rules summary
        summary_chunk = {
            "chunk_id": "rules_summary",
            "chunk_type": "annotation_rules_summary",
            "domain": "RULES",
            "variable": "SUMMARY",
            "full_name": "RULES.SUMMARY",
            "content": json.dumps(rules_summary, indent=2),
            "metadata": {
                "rule_type": "summary",
                "categories": list(rules_summary.keys())
            },
            "search_text": "annotation rules formatting patterns guidelines"
        }
        chunks.append(summary_chunk)
        
        # Create chunks for each rule category
        for category, rules in self.annotation_rules.rules.items():
            for rule in rules:
                rule_chunk = {
                    "chunk_id": f"rule_{rule.rule_id}",
                    "chunk_type": "annotation_rule",
                    "domain": "RULES",
                    "variable": rule.rule_id,
                    "full_name": f"RULES.{rule.rule_id}",
                    "content": f"{rule.title}: {rule.description}",
                    "metadata": {
                        "category": category,
                        "rule_id": rule.rule_id,
                        "pattern": rule.pattern,
                        "examples": rule.examples,
                        "msg_reference": rule.msg_reference
                    },
                    "search_text": f"{rule.title} {rule.description} {' '.join(rule.examples or [])}".lower()
                }
                chunks.append(rule_chunk)
                
        # Create chunks for annotation patterns
        for ann_type in AnnotationType:
            pattern = self.annotation_rules.get_pattern_for_type(ann_type)
            if pattern:
                pattern_chunk = {
                    "chunk_id": f"pattern_{ann_type.value}",
                    "chunk_type": "annotation_pattern",
                    "domain": "PATTERN",
                    "variable": ann_type.value.upper(),
                    "full_name": f"PATTERN.{ann_type.value.upper()}",
                    "content": f"Pattern for {ann_type.value}: {pattern}",
                    "metadata": {
                        "annotation_type": ann_type.value,
                        "regex_pattern": pattern
                    },
                    "search_text": f"{ann_type.value} annotation pattern format"
                }
                chunks.append(pattern_chunk)
                
        return chunks
        
    def _extract_example_chunks(self, section_title: str, content: str) -> List[Dict[str, Any]]:
        """Extract examples as individual chunks"""
        
        chunks = []
        
        # Find JSON examples
        import re
        json_pattern = r'```json\n(.*?)\n```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            try:
                # Parse JSON to extract key information
                example_data = json.loads(match)
                
                chunk = {
                    "chunk_id": f"example_{section_title.lower().replace(' ', '_')}_{i}",
                    "chunk_type": "annotation_example",
                    "domain": example_data.get("domain", "UNKNOWN"),
                    "variable": "/".join(example_data.get("variables", [])),
                    "full_name": f"{example_data.get('domain', 'UNKNOWN')}.{'/'.join(example_data.get('variables', []))}",
                    "content": f"Example: {example_data.get('text_anchor', '')} → {example_data.get('domain', '')}.{'/'.join(example_data.get('variables', []))}",
                    "metadata": {
                        "example": example_data,
                        "section": section_title
                    },
                    "search_text": f"{example_data.get('text_anchor', '')} {example_data.get('domain', '')} {' '.join(example_data.get('variables', []))}"
                }
                chunks.append(chunk)
            except:
                # If JSON parsing fails, still create a text chunk
                pass
                
        return chunks
        
    def _create_concept_chunks(self) -> List[Dict[str, Any]]:
        """Create chunks for key SDTM annotation concepts"""
        
        concepts = [
            {
                "concept": "Findings When Pattern",
                "content": "For Findings domains (VS, LB, QS, etc.), always use 'when' clause: VARIABLE when TESTCD=VALUE",
                "example": "VSORRES/VSORRESU when VSTESTCD=TEMP"
            },
            {
                "concept": "Supplemental Qualifiers",
                "content": "For 'Other, specify' fields, use supplemental qualifiers with QNAM and SUPPxx domain",
                "example": "RACE with QNAM=RACEOTH in SUPPDM"
            },
            {
                "concept": "RELREC Relationships",
                "content": "For cross-domain relationships, use RELREC: RELREC when DOMAIN1.VAR = DOMAIN2.VAR",
                "example": "RELREC when DDLNKID = AE.AELNKID"
            },
            {
                "concept": "Not Submitted",
                "content": "For operational fields not in SDTM, mark not_submitted=true and add [NOT SUBMITTED] in notes",
                "example": "Site use only fields"
            },
            {
                "concept": "Origins",
                "content": "Valid origins: Collected, Derived, Predecessor, Assigned, ePRO Collected",
                "example": "EC to EX mapping uses 'Predecessor'"
            }
        ]
        
        chunks = []
        for i, concept in enumerate(concepts):
            chunk = {
                "chunk_id": f"concept_{i}_{concept['concept'].lower().replace(' ', '_')}",
                "chunk_type": "annotation_concept",
                "domain": "CONCEPT",
                "variable": concept['concept'].upper().replace(' ', '_'),
                "full_name": f"CONCEPT.{concept['concept'].upper().replace(' ', '_')}",
                "content": f"{concept['concept']}: {concept['content']}. Example: {concept['example']}",
                "metadata": concept,
                "search_text": f"{concept['concept']} {concept['content']} {concept['example']}".lower()
            }
            chunks.append(chunk)
            
        return chunks
        
    def save_knowledge_base(self, output_dir: Path):
        """Save the structured knowledge base"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save variable entries
        entries_dict = {k: asdict(v) for k, v in self.kb_entries.items()}
        with open(output_dir / "sdtm_variables_structured.json", "w") as f:
            json.dump(entries_dict, f, indent=2)
            
        # Save indices
        with open(output_dir / "sdtm_domain_index.json", "w") as f:
            json.dump(self.domain_index, f, indent=2)
            
        with open(output_dir / "sdtm_alias_index.json", "w") as f:
            json.dump(self.alias_index, f, indent=2)
            
        # Save RAG chunks
        chunks = self.build_rag_chunks()
        with open(output_dir / "sdtm_rag_chunks.json", "w") as f:
            json.dump(chunks, f, indent=2)
            
        logger.info(f"Saved {len(self.kb_entries)} variables and {len(chunks)} RAG chunks")


def main():
    """Build SDTM RAG knowledge base"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Build SDTM RAG Knowledge Base")
    parser.add_argument("--base-path", default=".", help="Base path for SDTM data")
    parser.add_argument("--output-dir", default="sdtm_rag_kb", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Build knowledge base
    builder = SDTMRAGBuilder(args.base_path)
    logger.info("Loading SDTM data...")
    builder.load_sdtm_data()
    
    logger.info(f"Loaded {len(builder.kb_entries)} SDTM variables")
    logger.info(f"Created {len(builder.alias_index)} alias mappings")
    
    # Save knowledge base
    builder.save_knowledge_base(args.output_dir)
    logger.info(f"Knowledge base saved to {args.output_dir}")


if __name__ == "__main__":
    main()