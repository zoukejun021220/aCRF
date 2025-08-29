#!/usr/bin/env python3
"""
Knowledge Base Builder for SDTM Standards
Builds versioned KB from SDTMIG v3.4 and NCI EVS Controlled Terminology
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Domain:
    """SDTM Domain definition"""
    code: str
    name: str
    class_: str  # Using class_ to avoid Python keyword
    definition: str
    ig_section: str = ""
    
    def to_dict(self):
        d = asdict(self)
        d["class"] = d.pop("class_")
        return d


@dataclass
class Variable:
    """SDTM Variable definition"""
    domain: str
    name: str
    role: str  # Identifier, Topic, Timing, Qualifier, Rule
    label: str
    definition: str
    notes: str = ""
    usage: str = "Required"
    codelist: Optional[str] = None


@dataclass
class ControlledTerm:
    """NCI EVS Controlled Term"""
    codelist_id: str
    codelist_name: str
    code: str
    preferred_term: str
    ncit_code: str = ""
    synonyms: List[str] = None
    definition: str = ""
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


class SDTMKnowledgeBaseBuilder:
    """Builds versioned SDTM knowledge base files"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.versions = {
            "sdtm": "2.0",
            "sdtmig": "3.4",
            "ct_release": None,
            "build_date": datetime.utcnow().isoformat()
        }
        
    def build_domains(self) -> List[Domain]:
        """Build domain definitions from SDTMIG v3.4"""
        logger.info("Building SDTM domains...")
        
        # SDTMIG v3.4 human clinical domains
        # Source: CDISC SDTMIG v3.4 + EVS C66734
        domains = [
            # Special Purpose Domains
            Domain("DM", "Demographics", "Special Purpose", 
                  "Subject demographic characteristics", "6.2"),
            Domain("CO", "Comments", "Special Purpose",
                  "Comments collected on CRFs", "6.4"),
            Domain("SE", "Subject Elements", "Special Purpose",
                  "Timing and order of elements", "6.5"),
            Domain("SV", "Subject Visits", "Special Purpose",
                  "Actual visit dates", "6.6"),
            
            # Interventions Domains
            Domain("CM", "Concomitant/Prior Medications", "Interventions",
                  "Non-study medications", "7.2"),
            Domain("EC", "Exposure as Collected", "Interventions",
                  "Study treatment exposure data", "7.3"),
            Domain("EX", "Exposure", "Interventions",
                  "Protocol-specified study treatment", "7.4"),
            Domain("SU", "Substance Use", "Interventions",
                  "Alcohol, tobacco, caffeine use", "7.5"),
            Domain("PR", "Procedures", "Interventions",
                  "Therapeutic procedures", "7.6"),
            
            # Events Domains
            Domain("AE", "Adverse Events", "Events",
                  "Adverse events experienced", "8.2"),
            Domain("CE", "Clinical Events", "Events",
                  "Clinical events of interest", "8.3"),
            Domain("DS", "Disposition", "Events",
                  "Subject disposition events", "8.4"),
            Domain("DV", "Protocol Deviations", "Events",
                  "Protocol deviations", "8.5"),
            Domain("HO", "Healthcare Encounters", "Events",
                  "Healthcare resource utilization", "8.6"),
            Domain("MH", "Medical History", "Events",
                  "Subject medical history", "8.7"),
            
            # Findings Domains
            Domain("DA", "Drug Accountability", "Findings",
                  "Drug dispensing and return", "9.2"),
            Domain("DD", "Death Details", "Findings",
                  "Details about subject death", "9.3"),
            Domain("EG", "ECG Test Results", "Findings",
                  "Electrocardiogram findings", "9.4"),
            Domain("IE", "Inclusion/Exclusion Criteria", "Findings",
                  "Eligibility criteria", "9.5"),
            Domain("LB", "Laboratory Test Results", "Findings",
                  "Laboratory test findings", "9.6"),
            Domain("PE", "Physical Examination", "Findings",
                  "Physical exam findings", "9.7"),
            Domain("QS", "Questionnaires", "Findings",
                  "Questionnaire responses", "9.8"),
            Domain("SC", "Subject Characteristics", "Findings",
                  "Subject baseline characteristics", "9.9"),
            Domain("VS", "Vital Signs", "Findings",
                  "Vital sign measurements", "9.10"),
            
            # Findings About Domains
            Domain("FA", "Findings About", "Findings About",
                  "Findings about interventions/events", "10.2"),
            Domain("SR", "Skin Response", "Findings About",
                  "Skin test responses", "10.3"),
            
            # Trial Design Domains
            Domain("TA", "Trial Arms", "Trial Design",
                  "Planned trial arms", "11.2"),
            Domain("TD", "Trial Disease Assessments", "Trial Design",
                  "Planned disease assessments", "11.3"),
            Domain("TE", "Trial Elements", "Trial Design",
                  "Planned elements", "11.4"),
            Domain("TI", "Trial Inclusion/Exclusion Criteria", "Trial Design",
                  "Protocol eligibility criteria", "11.5"),
            Domain("TM", "Trial Disease Milestones", "Trial Design",
                  "Disease progression milestones", "11.6"),
            Domain("TS", "Trial Summary", "Trial Design",
                  "Trial summary information", "11.7"),
            Domain("TV", "Trial Visits", "Trial Design",
                  "Planned visits", "11.8"),
            
            # Relationship Domains
            Domain("RELREC", "Related Records", "Relationship",
                  "Record relationships", "12.1"),
            Domain("RELSUB", "Related Subjects", "Relationship",
                  "Subject relationships", "12.2"),
            Domain("RELSPEC", "Related Specimens", "Relationship",
                  "Specimen relationships", "12.3"),
            Domain("SUPPQUAL", "Supplemental Qualifiers", "Relationship",
                  "Additional qualifier variables", "12.4"),
        ]
        
        # Save domains
        domains_file = self.output_dir / "domains.json"
        with open(domains_file, 'w') as f:
            json.dump([d.to_dict() for d in domains], f, indent=2)
            
        logger.info(f"Saved {len(domains)} domains to {domains_file}")
        return domains
        
    def build_variables(self, domains: List[Domain]) -> List[Variable]:
        """Build variable definitions for each domain"""
        logger.info("Building SDTM variables...")
        
        variables = []
        
        # Common identifiers across domains
        common_identifiers = [
            ("STUDYID", "Study Identifier", "Unique study identifier"),
            ("DOMAIN", "Domain Abbreviation", "Domain code"),
            ("USUBJID", "Unique Subject Identifier", "Unique subject identifier"),
            ("SUBJID", "Subject Identifier", "Subject identifier within study"),
        ]
        
        # Add variables by domain
        for domain in domains:
            domain_code = domain.code
            
            # Add common identifiers
            for var_name, label, definition in common_identifiers:
                if var_name == "DOMAIN":
                    # Special handling for DOMAIN variable
                    variables.append(Variable(
                        domain=domain_code,
                        name=var_name,
                        role="Identifier",
                        label=label,
                        definition=f"Value is '{domain_code}'",
                        usage="Required"
                    ))
                else:
                    variables.append(Variable(
                        domain=domain_code,
                        name=var_name,
                        role="Identifier",
                        label=label,
                        definition=definition,
                        usage="Required"
                    ))
            
            # Add domain-specific variables
            if domain_code == "DM":
                dm_vars = [
                    Variable(domain_code, "BRTHDTC", "Date/Time of Birth", "Timing",
                            "Subject's date of birth"),
                    Variable(domain_code, "AGE", "Age", "Timing",
                            "Age at informed consent"),
                    Variable(domain_code, "AGEU", "Age Units", "Timing",
                            "Units for AGE", codelist="C66781"),
                    Variable(domain_code, "SEX", "Sex", "Topic",
                            "Sex of subject", codelist="C66731"),
                    Variable(domain_code, "RACE", "Race", "Topic",
                            "Race of subject", codelist="C74457"),
                    Variable(domain_code, "ETHNIC", "Ethnicity", "Topic",
                            "Ethnicity of subject", codelist="C66790"),
                    Variable(domain_code, "ARMCD", "Planned Arm Code", "Record Qualifier",
                            "Planned arm code"),
                    Variable(domain_code, "ARM", "Description of Planned Arm", "Synonym Qualifier",
                            "Planned arm description"),
                    Variable(domain_code, "COUNTRY", "Country", "Record Qualifier",
                            "Country of investigational site", codelist="ISO3166"),
                ]
                variables.extend(dm_vars)
                
            elif domain_code == "VS":
                vs_vars = [
                    Variable(domain_code, "VSSEQ", "Sequence Number", "Identifier",
                            "Sequence number within domain"),
                    Variable(domain_code, "VSTESTCD", "Vital Signs Test Short Name", "Topic",
                            "Short name of vital sign test", codelist="C66741"),
                    Variable(domain_code, "VSTEST", "Vital Signs Test Name", "Synonym Qualifier",
                            "Name of vital sign test", codelist="C66742"),
                    Variable(domain_code, "VSPOS", "Vital Signs Position", "Record Qualifier",
                            "Position during measurement", codelist="C71148"),
                    Variable(domain_code, "VSORRES", "Result or Finding in Original Units", "Result Qualifier",
                            "Vital sign result as collected"),
                    Variable(domain_code, "VSORRESU", "Original Units", "Variable Qualifier",
                            "Units for VSORRES", codelist="C66770"),
                    Variable(domain_code, "VSSTRESC", "Character Result/Finding", "Result Qualifier",
                            "Standardized result in character format"),
                    Variable(domain_code, "VSSTRESN", "Numeric Result/Finding", "Result Qualifier",
                            "Standardized numeric result"),
                    Variable(domain_code, "VSSTRESU", "Standard Units", "Variable Qualifier",
                            "Units for VSSTRESN", codelist="C66770"),
                    Variable(domain_code, "VSDTC", "Date/Time of Measurements", "Timing",
                            "Date/time vital signs collected"),
                ]
                variables.extend(vs_vars)
                
            elif domain_code == "AE":
                ae_vars = [
                    Variable(domain_code, "AESEQ", "Sequence Number", "Identifier",
                            "Sequence number within domain"),
                    Variable(domain_code, "AETERM", "Reported Term", "Topic",
                            "Adverse event verbatim term"),
                    Variable(domain_code, "AEDECOD", "Dictionary-Derived Term", "Synonym Qualifier",
                            "MedDRA preferred term", codelist="MedDRA"),
                    Variable(domain_code, "AEBODSYS", "Body System or Organ Class", "Result Qualifier",
                            "MedDRA system organ class", codelist="MedDRA"),
                    Variable(domain_code, "AESEV", "Severity/Intensity", "Record Qualifier",
                            "Severity of adverse event", codelist="C66769"),
                    Variable(domain_code, "AESER", "Serious Event", "Record Qualifier",
                            "Serious adverse event flag", codelist="NY"),
                    Variable(domain_code, "AEACN", "Action Taken with Study Treatment", "Record Qualifier",
                            "Action taken", codelist="C66767"),
                    Variable(domain_code, "AEREL", "Causality", "Record Qualifier",
                            "Relationship to study treatment", codelist="C66766"),
                    Variable(domain_code, "AEOUT", "Outcome of Adverse Event", "Record Qualifier",
                            "Outcome of event", codelist="C66768"),
                    Variable(domain_code, "AESTDTC", "Start Date/Time of Adverse Event", "Timing",
                            "Start date/time of AE"),
                    Variable(domain_code, "AEENDTC", "End Date/Time of Adverse Event", "Timing",
                            "End date/time of AE"),
                ]
                variables.extend(ae_vars)
                
            elif domain_code == "LB":
                lb_vars = [
                    Variable(domain_code, "LBSEQ", "Sequence Number", "Identifier",
                            "Sequence number within domain"),
                    Variable(domain_code, "LBTESTCD", "Lab Test or Examination Short Name", "Topic",
                            "Short name of lab test", codelist="C65047"),
                    Variable(domain_code, "LBTEST", "Lab Test or Examination Name", "Synonym Qualifier",
                            "Name of lab test", codelist="C67154"),
                    Variable(domain_code, "LBCAT", "Category for Lab Test", "Grouping Qualifier",
                            "Category of lab test"),
                    Variable(domain_code, "LBORRES", "Result or Finding in Original Units", "Result Qualifier",
                            "Lab result as collected"),
                    Variable(domain_code, "LBORRESU", "Original Units", "Variable Qualifier",
                            "Units for LBORRES", codelist="C66770"),
                    Variable(domain_code, "LBORNRLO", "Reference Range Lower Limit", "Variable Qualifier",
                            "Lower limit of normal range"),
                    Variable(domain_code, "LBORNRHI", "Reference Range Upper Limit", "Variable Qualifier",
                            "Upper limit of normal range"),
                    Variable(domain_code, "LBSTRESC", "Character Result/Finding", "Result Qualifier",
                            "Standardized result in character format"),
                    Variable(domain_code, "LBSTRESN", "Numeric Result/Finding", "Result Qualifier",
                            "Standardized numeric result"),
                    Variable(domain_code, "LBSTRESU", "Standard Units", "Variable Qualifier",
                            "Units for LBSTRESN", codelist="C66770"),
                    Variable(domain_code, "LBDTC", "Date/Time of Specimen Collection", "Timing",
                            "Date/time specimen collected"),
                ]
                variables.extend(lb_vars)
                
            elif domain_code == "CM":
                cm_vars = [
                    Variable(domain_code, "CMSEQ", "Sequence Number", "Identifier",
                            "Sequence number within domain"),
                    Variable(domain_code, "CMTRT", "Reported Name of Drug, Med, or Therapy", "Topic",
                            "Medication verbatim term"),
                    Variable(domain_code, "CMDECOD", "Standardized Medication Name", "Synonym Qualifier",
                            "WHODrug preferred name", codelist="WHODrug"),
                    Variable(domain_code, "CMCAT", "Category for Medication", "Grouping Qualifier",
                            "Category of medication"),
                    Variable(domain_code, "CMINDIC", "Indication", "Record Qualifier",
                            "Reason for medication"),
                    Variable(domain_code, "CMDOSE", "Dose per Administration", "Record Qualifier",
                            "Numeric dose amount"),
                    Variable(domain_code, "CMDOSU", "Dose Units", "Variable Qualifier",
                            "Units for CMDOSE", codelist="C66770"),
                    Variable(domain_code, "CMDOSFRQ", "Dosing Frequency per Interval", "Timing",
                            "Frequency of dosing", codelist="C66728"),
                    Variable(domain_code, "CMROUTE", "Route of Administration", "Record Qualifier",
                            "Route of administration", codelist="C66729"),
                    Variable(domain_code, "CMSTDTC", "Start Date/Time of Medication", "Timing",
                            "Start date/time of medication"),
                    Variable(domain_code, "CMENDTC", "End Date/Time of Medication", "Timing",
                            "End date/time of medication"),
                    Variable(domain_code, "CMONGO", "Ongoing", "Record Qualifier",
                            "Medication ongoing flag", codelist="NY"),
                ]
                variables.extend(cm_vars)
                
            # Add more domain-specific variables as needed...
            
        # Save variables
        variables_file = self.output_dir / "variables.json"
        with open(variables_file, 'w') as f:
            json.dump([asdict(v) for v in variables], f, indent=2)
            
        logger.info(f"Saved {len(variables)} variables to {variables_file}")
        return variables
        
    def build_controlled_terms(self, ct_release_date: str) -> List[ControlledTerm]:
        """Build controlled terminology from NCI EVS"""
        logger.info("Building controlled terminology...")
        
        self.versions["ct_release"] = ct_release_date
        
        # Sample controlled terms (would be loaded from EVS in production)
        controlled_terms = []
        
        # C66731 - Sex codelist
        sex_terms = [
            ControlledTerm("C66731", "Sex", "F", "Female", "C16576"),
            ControlledTerm("C66731", "Sex", "M", "Male", "C20197"),
            ControlledTerm("C66731", "Sex", "U", "Unknown", "C17998"),
            ControlledTerm("C66731", "Sex", "UNDIFFERENTIATED", "Undifferentiated", "C45908"),
        ]
        controlled_terms.extend(sex_terms)
        
        # C66741 - Vital Signs Test Code
        vs_test_codes = [
            ControlledTerm("C66741", "Vital Signs Test Code", "SYSBP", 
                          "Systolic Blood Pressure", "C25298",
                          ["Systolic BP", "SBP"], "Systolic arterial pressure"),
            ControlledTerm("C66741", "Vital Signs Test Code", "DIABP",
                          "Diastolic Blood Pressure", "C25299",
                          ["Diastolic BP", "DBP"], "Diastolic arterial pressure"),
            ControlledTerm("C66741", "Vital Signs Test Code", "PULSE",
                          "Pulse Rate", "C49676",
                          ["Heart Rate", "HR", "Pulse"], "Number of heartbeats per minute"),
            ControlledTerm("C66741", "Vital Signs Test Code", "TEMP",
                          "Temperature", "C25206",
                          ["Body Temperature", "Temp"], "Body temperature measurement"),
            ControlledTerm("C66741", "Vital Signs Test Code", "RESP",
                          "Respiratory Rate", "C49677",
                          ["RR", "Breathing Rate"], "Number of breaths per minute"),
            ControlledTerm("C66741", "Vital Signs Test Code", "HEIGHT",
                          "Height", "C25347",
                          ["Body Height", "Stature"], "Height measurement"),
            ControlledTerm("C66741", "Vital Signs Test Code", "WEIGHT",
                          "Weight", "C25208",
                          ["Body Weight", "Mass"], "Body weight measurement"),
        ]
        controlled_terms.extend(vs_test_codes)
        
        # C66767 - Action Taken with Study Treatment
        aeacn_terms = [
            ControlledTerm("C66767", "Action Taken with Study Treatment",
                          "DOSE NOT CHANGED", "Dose Not Changed", "C48660"),
            ControlledTerm("C66767", "Action Taken with Study Treatment",
                          "DOSE REDUCED", "Dose Reduced", "C49503"),
            ControlledTerm("C66767", "Action Taken with Study Treatment",
                          "DRUG INTERRUPTED", "Drug Interrupted", "C49502"),
            ControlledTerm("C66767", "Action Taken with Study Treatment",
                          "DRUG WITHDRAWN", "Drug Withdrawn", "C49501"),
            ControlledTerm("C66767", "Action Taken with Study Treatment",
                          "NOT APPLICABLE", "Not Applicable", "C48660"),
        ]
        controlled_terms.extend(aeacn_terms)
        
        # C66768 - Outcome of Adverse Event
        aeout_terms = [
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "RECOVERED/RESOLVED", "Recovered/Resolved", "C49494"),
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "RECOVERING/RESOLVING", "Recovering/Resolving", "C49495"),
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "NOT RECOVERED/NOT RESOLVED", "Not Recovered/Not Resolved", "C49496"),
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "RECOVERED/RESOLVED WITH SEQUELAE", "Recovered/Resolved with Sequelae", "C49497"),
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "FATAL", "Fatal", "C48275"),
            ControlledTerm("C66768", "Outcome of Adverse Event",
                          "UNKNOWN", "Unknown", "C17998"),
        ]
        controlled_terms.extend(aeout_terms)
        
        # C66769 - Severity
        severity_terms = [
            ControlledTerm("C66769", "Severity", "MILD", "Mild", "C70666"),
            ControlledTerm("C66769", "Severity", "MODERATE", "Moderate", "C70667"),
            ControlledTerm("C66769", "Severity", "SEVERE", "Severe", "C70668"),
        ]
        controlled_terms.extend(severity_terms)
        
        # NY - No Yes Response
        ny_terms = [
            ControlledTerm("NY", "No Yes Response", "N", "No", "C49488"),
            ControlledTerm("NY", "No Yes Response", "Y", "Yes", "C49487"),
        ]
        controlled_terms.extend(ny_terms)
        
        # Save controlled terms
        ct_file = self.output_dir / "ct_snapshot.json"
        with open(ct_file, 'w') as f:
            json.dump([asdict(ct) for ct in controlled_terms], f, indent=2)
            
        logger.info(f"Saved {len(controlled_terms)} controlled terms to {ct_file}")
        return controlled_terms
        
    def build_codelist_mappings(self, variables: List[Variable]) -> Dict[str, str]:
        """Build mapping of variables to their codelists"""
        logger.info("Building codelist mappings...")
        
        # Extract variable-to-codelist mappings
        codelist_map = {}
        
        for var in variables:
            if var.codelist:
                var_key = f"{var.domain}.{var.name}"
                codelist_map[var_key] = var.codelist
                
        # Add common patterns
        patterns = {
            "TESTCD": {  # Test codes by domain
                "VS": "C66741",  # Vital Signs Test Code
                "LB": "C65047",  # Laboratory Test Code  
                "EG": "C71150",  # ECG Test Code
                "QS": "C66742",  # Questionnaire Test Code
            },
            "UNIT": "C66770",  # Unit of Measure
            "ROUTE": "C66729",  # Route of Administration
            "FREQ": "C66728",  # Frequency
            "POSITION": "C71148",  # Position
            "SEV": "C66769",  # Severity
            "ACN": "C66767",  # Action Taken
            "REL": "C66766",  # Relationship to Study Treatment
            "OUT": "C66768",  # Outcome
        }
        
        # Save mappings
        mapping_file = self.output_dir / "codelist_mappings.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                "direct_mappings": codelist_map,
                "pattern_mappings": patterns
            }, f, indent=2)
            
        logger.info(f"Saved codelist mappings to {mapping_file}")
        return codelist_map
        
    def create_metadata(self):
        """Create metadata file with version information"""
        metadata_file = self.output_dir / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
            
        logger.info(f"Created metadata file: {metadata_file}")
        
    def build_all(self, ct_release_date: str = "2024-01"):
        """Build complete knowledge base"""
        logger.info("Building complete SDTM knowledge base...")
        
        # Build components
        domains = self.build_domains()
        variables = self.build_variables(domains)
        controlled_terms = self.build_controlled_terms(ct_release_date)
        codelist_mappings = self.build_codelist_mappings(variables)
        
        # Create metadata
        self.create_metadata()
        
        # Create summary
        summary = {
            "domains": len(domains),
            "variables": len(variables),
            "controlled_terms": len(controlled_terms),
            "codelists": len(set(ct.codelist_id for ct in controlled_terms)),
            "versions": self.versions
        }
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Knowledge base build complete: {self.output_dir}")
        logger.info(f"Summary: {summary}")


def main():
    """Build SDTM knowledge base"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build SDTM Knowledge Base")
    parser.add_argument("--output-dir", type=str, default="kb/sdtmig_v3_4",
                       help="Output directory for KB files")
    parser.add_argument("--ct-release", type=str, default="2024-01",
                       help="CT release date (YYYY-MM)")
    
    args = parser.parse_args()
    
    # Create builder
    builder = SDTMKnowledgeBaseBuilder(Path(args.output_dir))
    
    # Build knowledge base
    builder.build_all(args.ct_release)


if __name__ == "__main__":
    main()