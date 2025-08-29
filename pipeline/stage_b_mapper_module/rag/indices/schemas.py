"""Schema definitions for RAG indices"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DomainDocument:
    """Domain index document schema"""
    domain_code: str
    domain_name: str
    domain_class: str  # Interventions/Events/Findings/Special Purpose
    definition: str
    # Derived text for embedding
    embedding_text: str = ""
    
    def __post_init__(self):
        if not self.embedding_text:
            self.embedding_text = (
                f"SDTM Domain: {self.domain_code} - {self.domain_name}. "
                f"Class: {self.domain_class}. Definition: {self.definition}"
            )


@dataclass
class VariableDocument:
    """Variable index document schema"""
    var_name: str
    label: str
    role: str  # Identifier/Topic/Timing/Qualifier/Rule
    domain_code: str
    definition: str
    codelist_name: Optional[str] = None
    codelist_code: Optional[str] = None
    data_type: Optional[str] = None
    core: Optional[str] = None  # Required/Expected/Permissible
    # Derived text for embedding
    embedding_text: str = ""
    
    def __post_init__(self):
        if not self.embedding_text:
            codelist_info = f" Codelist: {self.codelist_name}" if self.codelist_name else ""
            self.embedding_text = (
                f"SDTM Variable: {self.var_name} in {self.domain_code}. "
                f"Role: {self.role}. Label: {self.label}. "
                f"Definition: {self.definition}{codelist_info}"
            )


@dataclass
class CTDocument:
    """Controlled Terminology document schema"""
    codelist_name: str
    codelist_code: str
    submission_value: str
    preferred_term: str
    definition: str
    synonyms: List[str] = field(default_factory=list)
    nci_c_code: Optional[str] = None
    # For paired codelists (e.g., TEST/TESTCD)
    paired_codelist: Optional[str] = None
    paired_value: Optional[str] = None
    # Derived text for embedding
    embedding_text: str = ""
    
    def __post_init__(self):
        if not self.embedding_text:
            synonyms_text = f" Synonyms: {', '.join(self.synonyms)}" if self.synonyms else ""
            nci_text = f" NCI Code: {self.nci_c_code}" if self.nci_c_code else ""
            self.embedding_text = (
                f"CDISC CT Term in {self.codelist_name}: {self.submission_value}. "
                f"Definition: {self.definition}{synonyms_text}{nci_text}"
            )


@dataclass
class CTRelationship:
    """Variable to Codelist relationship"""
    variable_name: str
    domain_code: str
    codelist_name: str
    codelist_code: str
    relationship_type: str = "uses"  # uses, references, etc.


@dataclass
class UnitMapping:
    """Unit to UCUM mapping"""
    unit_text: str
    ucum_code: str
    description: str
    variables: List[str] = field(default_factory=list)  # Variables that use this unit