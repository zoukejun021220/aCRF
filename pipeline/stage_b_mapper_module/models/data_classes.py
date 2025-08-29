"""Data classes for Stage B SDTM Mapper"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Any

@dataclass
class SDTMMapping:
    """Single SDTM variable mapping"""
    variable: str
    value: Optional[str] = None
    condition: Optional[str] = None
    target: str = "input"  # input, question, option, section
    
@dataclass  
class VLMEntry:
    """Value-level metadata entry for findings"""
    domain: str
    testcd: str
    testcd_value: str
    variables: List[str]
    
@dataclass
class SUPPEntry:
    """Supplemental qualifier entry"""
    qnam: str
    qlabel: str
    domain: str
    
@dataclass
class RelationshipEntry:
    """Cross-domain relationship"""
    from_var: str
    to_var: str
    type: str = "RELREC"

@dataclass
class OptionAnnotation:
    """Annotation for a specific option value"""
    option_value: str
    annotation: str
    mappings: List[SDTMMapping]
    
@dataclass
class AnnotationResult:
    """Complete structured annotation result"""
    pattern: str
    domain: Optional[str]
    mappings: List[SDTMMapping]
    vlm: Optional[VLMEntry] = None
    supp: Optional[SUPPEntry] = None  
    relationships: List[RelationshipEntry] = None
    clarifying: List[str] = None
    confidence: float = 0.0
    validation_status: str = "pending"
    validation_message: str = ""
    # New fields for separate option annotations
    question_annotation: Optional[str] = None
    option_annotations: List[OptionAnnotation] = None
    
    def to_json_lines(self) -> str:
        """Convert to JSON lines format for plain text annotation"""
        lines = []
        
        # Primary mapping line
        base_line = {
            "pattern": self.pattern,
            "confidence": self.confidence
        }
        
        if self.domain:
            base_line["domain"] = self.domain
            
        if self.mappings:
            # Add primary mappings
            primary_mappings = []
            for m in self.mappings:
                if m.target == "question":
                    primary_mappings.append({
                        "var": m.variable,
                        "val": m.value,
                        "cond": m.condition,
                        "target": m.target
                    })
            if primary_mappings:
                base_line["mappings"] = primary_mappings
                
        lines.append(base_line)
        
        # Add option-specific annotations
        if self.option_annotations:
            for opt_ann in self.option_annotations:
                opt_line = {
                    "option": opt_ann.option_value,
                    "annotation": opt_ann.annotation,
                    "mappings": [
                        {
                            "var": m.variable,
                            "val": m.value, 
                            "cond": m.condition
                        } for m in opt_ann.mappings
                    ]
                }
                lines.append(opt_line)
                
        # Add supplemental qualifiers
        if self.supp:
            supp_line = {
                "type": "SUPP",
                "qnam": self.supp.qnam,
                "qlabel": self.supp.qlabel,
                "domain": self.supp.domain
            }
            lines.append(supp_line)
            
        # Add VLM entries
        if self.vlm:
            vlm_line = {
                "type": "VLM",
                "domain": self.vlm.domain,
                "testcd": self.vlm.testcd,
                "testcd_value": self.vlm.testcd_value,
                "variables": self.vlm.variables
            }
            lines.append(vlm_line)
            
        # Add relationships
        if self.relationships:
            for rel in self.relationships:
                rel_line = {
                    "type": rel.type,
                    "from_var": rel.from_var,
                    "to_var": rel.to_var
                }
                lines.append(rel_line)
                
        import json
        return "\n".join(json.dumps(line) for line in lines)