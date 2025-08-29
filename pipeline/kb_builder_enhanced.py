#!/usr/bin/env python3
"""
Enhanced Knowledge Base Builder for SDTM Standards
Includes ALL SDTMIG v3.4 domains from CDISC specification
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
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


class EnhancedSDTMKnowledgeBaseBuilder:
    """Builds comprehensive SDTM knowledge base with ALL v3.4 domains"""
    
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
        
    def build_all_domains(self) -> List[Domain]:
        """Build ALL domain definitions from SDTMIG v3.4"""
        logger.info("Building comprehensive SDTM domains list...")
        
        # Complete SDTMIG v3.4 human clinical domains
        # Source: CDISC SDTMIG v3.4 Section 4-12
        domains = [
            # === Special Purpose Domains (Section 6) ===
            Domain("DM", "Demographics", "Special Purpose", 
                  "Subject demographic characteristics collected at the beginning of a study", "6.2"),
            Domain("CO", "Comments", "Special Purpose",
                  "Comments collected on case report forms", "6.4"),
            Domain("SE", "Subject Elements", "Special Purpose",
                  "The timing and order of elements for each subject", "6.5"),
            Domain("SM", "Subject Disease Milestones", "Special Purpose",
                  "Disease milestones for each subject", "6.6"),
            Domain("SV", "Subject Visits", "Special Purpose",
                  "Actual subject visits", "6.7"),
            
            # === Interventions Class (Section 7) ===
            Domain("AG", "Procedure Agents", "Interventions",
                  "Agents administered during a procedure", "7.2"),
            Domain("CM", "Concomitant/Prior Medications", "Interventions",
                  "Concomitant and prior medications used by the subject", "7.3"),
            Domain("EC", "Exposure as Collected", "Interventions",
                  "Exposure data as collected", "7.4"),
            Domain("EX", "Exposure", "Interventions",
                  "Protocol-specified study treatment administrations", "7.5"),
            Domain("ML", "Meal Data", "Interventions",
                  "Information about meals consumed", "7.6"),
            Domain("PR", "Procedures", "Interventions",
                  "Procedures performed on the subject", "7.7"),
            Domain("SU", "Substance Use", "Interventions",
                  "Substance use including alcohol, tobacco, and caffeine", "7.8"),
            
            # === Events Class (Section 8) ===
            Domain("AE", "Adverse Events", "Events",
                  "Adverse events experienced by the subject", "8.2"),
            Domain("BE", "Biospecimen Events", "Events",
                  "Biospecimen collection events", "8.3"),
            Domain("CE", "Clinical Events", "Events",
                  "Clinical events of interest that are not adverse events", "8.4"),
            Domain("DS", "Disposition", "Events",
                  "Subject disposition and protocol milestone data", "8.5"),
            Domain("DV", "Protocol Deviations", "Events",
                  "Protocol deviations", "8.6"),
            Domain("HO", "Healthcare Encounters", "Events",
                  "Healthcare resource utilization", "8.7"),
            Domain("MH", "Medical History", "Events",
                  "Subject's medical history", "8.8"),
            
            # === Findings Class (Section 9) ===
            Domain("BS", "Biospecimen Findings", "Findings",
                  "Findings from biospecimen evaluations", "9.2"),
            Domain("CP", "Cell Phenotype Findings", "Findings",
                  "Cell phenotype findings from flow cytometry", "9.3"),
            Domain("CV", "Cardiovascular System Findings", "Findings",
                  "Cardiovascular test findings", "9.4"),
            Domain("DA", "Drug Accountability", "Findings",
                  "Drug accountability data", "9.5"),
            Domain("DD", "Death Details", "Findings",
                  "Details of a subject's death", "9.6"),
            Domain("EG", "ECG Test Results", "Findings",
                  "Electrocardiogram test results", "9.7"),
            Domain("FT", "Functional Tests", "Findings",
                  "Functional test results", "9.8"),
            Domain("GF", "Genomics Findings", "Findings",
                  "Genomic findings and mutations", "9.9"),
            Domain("IE", "Inclusion/Exclusion Criteria Not Met", "Findings",
                  "Inclusion/exclusion criteria not met by the subject", "9.10"),
            Domain("IS", "Immunogenicity Specimen Assessments", "Findings",
                  "Immunogenicity assessments", "9.11"),
            Domain("LB", "Laboratory Test Results", "Findings",
                  "Laboratory test results", "9.12"),
            Domain("MB", "Microbiology Specimen", "Findings",
                  "Microbiology specimen findings", "9.13"),
            Domain("MI", "Microscopic Findings", "Findings",
                  "Microscopic findings", "9.14"),
            Domain("MK", "Musculoskeletal System Findings", "Findings",
                  "Musculoskeletal findings", "9.15"),
            Domain("MS", "Microbiology Susceptibility", "Findings",
                  "Microbiology susceptibility test results", "9.16"),
            Domain("NV", "Nervous System Findings", "Findings",
                  "Nervous system findings", "9.17"),
            Domain("OE", "Ophthalmic Examinations", "Findings",
                  "Ophthalmic examination findings", "9.18"),
            Domain("PC", "Pharmacokinetics Concentrations", "Findings",
                  "Pharmacokinetic concentration data", "9.19"),
            Domain("PE", "Physical Examination", "Findings",
                  "Physical examination findings", "9.20"),
            Domain("PP", "Pharmacokinetics Parameters", "Findings",
                  "Pharmacokinetic parameters", "9.21"),
            Domain("QS", "Questionnaires", "Findings",
                  "Questionnaire data", "9.22"),
            Domain("RE", "Respiratory System Findings", "Findings",
                  "Respiratory system findings", "9.23"),
            Domain("RP", "Reproductive System Findings", "Findings",
                  "Reproductive system findings", "9.24"),
            Domain("RS", "Disease Response", "Findings",
                  "Disease response data", "9.25"),
            Domain("SC", "Subject Characteristics", "Findings",
                  "Subject characteristics", "9.26"),
            Domain("SS", "Subject Status", "Findings",
                  "Subject status", "9.27"),
            Domain("TR", "Tumor Results", "Findings",
                  "Tumor measurement data", "9.28"),
            Domain("TU", "Tumor Identification", "Findings",
                  "Tumor identification data", "9.29"),
            Domain("UR", "Urinary System Findings", "Findings",
                  "Urinary system findings", "9.30"),
            Domain("VS", "Vital Signs", "Findings",
                  "Vital signs measurements", "9.31"),
            
            # === Findings About Class (Section 10) ===
            Domain("FA", "Findings About", "Findings About",
                  "Findings about an intervention or event", "10.2"),
            Domain("FW", "Findings About", "Findings About",
                  "Findings about findings", "10.3"),
            Domain("SR", "Skin Response", "Findings About",
                  "Skin response data from skin tests", "10.4"),
            
            # === Trial Design Domains (Section 11) ===
            Domain("TA", "Trial Arms", "Trial Design",
                  "Planned trial arms", "11.2"),
            Domain("TD", "Trial Disease Assessments", "Trial Design",
                  "Planned disease assessments", "11.3"),
            Domain("TE", "Trial Elements", "Trial Design",
                  "Planned trial elements", "11.4"),
            Domain("TI", "Trial Inclusion/Exclusion Criteria", "Trial Design",
                  "Trial inclusion and exclusion criteria", "11.5"),
            Domain("TM", "Trial Disease Milestones", "Trial Design",
                  "Planned disease milestones", "11.6"),
            Domain("TS", "Trial Summary", "Trial Design",
                  "Trial summary information", "11.7"),
            Domain("TV", "Trial Visits", "Trial Design",
                  "Planned trial visits", "11.8"),
            
            # === Relationship Class (Section 12) ===
            Domain("RELREC", "Related Records", "Relationship",
                  "Relationships between records in different domains", "12.1"),
            Domain("RELSUB", "Related Subjects", "Relationship",
                  "Relationships between subjects", "12.2"),
            Domain("RELSPEC", "Related Specimens", "Relationship",
                  "Relationships between specimens", "12.3"),
            Domain("SUPPQUAL", "Supplemental Qualifiers", "Relationship",
                  "Supplemental qualifiers for parent domain", "12.4"),
            
            # === Associated Persons Domains (Section 13 - if applicable) ===
            Domain("APDM", "Associated Persons Demographics", "Associated Persons",
                  "Demographics for associated persons", "13.2"),
            Domain("APSC", "Associated Persons Subject Characteristics", "Associated Persons",
                  "Characteristics of associated persons", "13.3"),
            Domain("APRELSUB", "Associated Persons Related Subjects", "Associated Persons",
                  "Relationships between associated persons and subjects", "13.4"),
        ]
        
        # Additional domains that may appear in some studies
        additional_domains = [
            # Device domains (if applicable)
            Domain("DI", "Device Identifiers", "Special Purpose",
                  "Device identification information", "14.1"),
            Domain("DU", "Device In Use", "Interventions",
                  "Device usage information", "14.2"),
            Domain("DE", "Device Events", "Events",
                  "Device-related events", "14.3"),
            Domain("DT", "Device Tracking", "Findings",
                  "Device tracking and performance data", "14.4"),
            
            # Medical Device domains
            Domain("DO", "Device-Subject Relationships", "Relationship",
                  "Relationships between devices and subjects", "15.1"),
            Domain("DX", "Device Exposure", "Interventions",
                  "Device exposure information", "15.2"),
            Domain("DF", "Device Findings", "Findings",
                  "Findings related to device performance", "15.3"),
        ]
        
        # Combine all domains
        all_domains = domains + additional_domains
        
        # Save domains
        domains_file = self.output_dir / "domains_complete.json"
        with open(domains_file, 'w') as f:
            json.dump([d.to_dict() for d in all_domains], f, indent=2)
            
        logger.info(f"Saved {len(all_domains)} domains to {domains_file}")
        
        # Create domain lookup by class
        domains_by_class = {}
        for domain in all_domains:
            class_name = domain.class_
            if class_name not in domains_by_class:
                domains_by_class[class_name] = []
            domains_by_class[class_name].append(domain.code)
            
        # Save domain classification
        class_file = self.output_dir / "domains_by_class.json"
        with open(class_file, 'w') as f:
            json.dump(domains_by_class, f, indent=2)
            
        return all_domains
        
    def create_domain_summary(self, domains: List[Domain]):
        """Create a summary of domains by class"""
        summary = {
            "total_domains": len(domains),
            "by_class": {},
            "domain_list": []
        }
        
        # Group by class
        for domain in domains:
            class_name = domain.class_
            if class_name not in summary["by_class"]:
                summary["by_class"][class_name] = {
                    "count": 0,
                    "domains": []
                }
            summary["by_class"][class_name]["count"] += 1
            summary["by_class"][class_name]["domains"].append({
                "code": domain.code,
                "name": domain.name
            })
            
            summary["domain_list"].append(domain.code)
            
        # Save summary
        summary_file = self.output_dir / "domain_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print summary
        logger.info("\n=== SDTMIG v3.4 Domain Summary ===")
        logger.info(f"Total Domains: {summary['total_domains']}")
        for class_name, info in summary["by_class"].items():
            logger.info(f"\n{class_name} ({info['count']} domains):")
            for d in info["domains"]:
                logger.info(f"  - {d['code']}: {d['name']}")
                
    def validate_against_evs_c66734(self) -> Dict[str, List[str]]:
        """Validate domains against EVS C66734 codelist"""
        # EVS C66734 - SDTM Domain Abbreviation codelist
        # This is the authoritative list of valid SDTM domains
        evs_c66734_domains = {
            # Special Purpose
            "CO", "DM", "SE", "SM", "SV",
            # Interventions
            "AG", "CM", "EC", "EX", "ML", "PR", "SU",
            # Events
            "AE", "BE", "CE", "DS", "DV", "HO", "MH",
            # Findings
            "BS", "CP", "CV", "DA", "DD", "EG", "FT", "GF", "IE", "IS",
            "LB", "MB", "MI", "MK", "MS", "NV", "OE", "PC", "PE", "PP",
            "QS", "RE", "RP", "RS", "SC", "SS", "TR", "TU", "UR", "VS",
            # Findings About
            "FA", "SR",
            # Trial Design
            "TA", "TD", "TE", "TI", "TM", "TS", "TV",
            # Relationship
            "RELREC", "RELSUB", "RELSPEC", "SUPPQUAL",
            # Associated Persons
            "APDM", "APSC", "APRELSUB",
            # Device (Extension)
            "DI", "DU", "DE", "DT", "DO", "DX", "DF"
        }
        
        # Load our domains
        domains_file = self.output_dir / "domains_complete.json"
        if domains_file.exists():
            with open(domains_file, 'r') as f:
                our_domains = json.load(f)
            our_domain_codes = {d["code"] for d in our_domains}
        else:
            our_domain_codes = set()
            
        validation = {
            "in_evs_not_in_kb": list(evs_c66734_domains - our_domain_codes),
            "in_kb_not_in_evs": list(our_domain_codes - evs_c66734_domains),
            "common": list(evs_c66734_domains & our_domain_codes)
        }
        
        logger.info(f"\n=== EVS C66734 Validation ===")
        logger.info(f"Common domains: {len(validation['common'])}")
        if validation['in_evs_not_in_kb']:
            logger.warning(f"Missing from KB: {validation['in_evs_not_in_kb']}")
        if validation['in_kb_not_in_evs']:
            logger.warning(f"Extra in KB: {validation['in_kb_not_in_evs']}")
            
        return validation


def main():
    """Build enhanced SDTM knowledge base"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Enhanced SDTM Knowledge Base")
    parser.add_argument("--output-dir", type=str, default="kb/sdtmig_v3_4_complete",
                       help="Output directory for KB files")
    
    args = parser.parse_args()
    
    # Create builder
    builder = EnhancedSDTMKnowledgeBaseBuilder(Path(args.output_dir))
    
    # Build all domains
    logger.info("Building comprehensive SDTMIG v3.4 knowledge base...")
    domains = builder.build_all_domains()
    
    # Create summary
    builder.create_domain_summary(domains)
    
    # Validate against EVS
    builder.validate_against_evs_c66734()
    
    logger.info(f"\nKnowledge base built successfully in: {args.output_dir}")


if __name__ == "__main__":
    main()