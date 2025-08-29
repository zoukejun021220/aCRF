# SDTMIG v3.4 Complete Knowledge Base

## Overview

This knowledge base contains comprehensive SDTM mapping information for the CRF annotation pipeline.

## Contents

- **domains.json**: All 63 SDTMIG v3.4 domains with metadata
- **variables_all.json**: 1,917 variables across all domains
- **variable_domain_map.json**: Maps 1,523 unique variables to their domains
- **common_ct_terms.json**: Common controlled terminology values
- **controlled_terminology.json**: Codelist references

## Usage in Pipeline

The pipeline uses this KB in two stages:

### Stage A (Digitization)
- Extracts CRF items without needing domain knowledge

### Stage B (SDTM Mapping)
1. Load KB:
   ```python
   from pathlib import Path
   kb_dir = Path("kb/sdtmig_v3_4_complete")
   ```

2. Domain selection based on item content
3. Variable mapping using domain-specific rules
4. CT value selection from predefined lists

## Domain Classes

- **Special Purpose** (5): DM, CO, SE, SV, SM
- **Interventions** (7): EX, EC, CM, SU, AG, ML, PR
- **Events** (7): AE, CE, DS, DV, HO, MH, BE
- **Findings** (30): LB, VS, EG, PE, etc.
- **Trial Design** (6): TA, TE, TV, TI, TS, TD
- **Relationship** (4): RELREC, RELSUB, RELSPEC, SUPPQUAL
- **Findings About** (2): FA, SR

## Key Variables

Multi-home variables (appear in many domains):
- STUDYID, DOMAIN, USUBJID (all domains)
- VISIT, VISITNUM, VISITDY (findings domains)
- EPOCH, TAETORD (most domains)

## Controlled Terminology

Common CT categories included:
- SEX: Sex values (M, F, U)
- NY: Yes/No responses (Y, N, U)
- AEACN: Action taken with study treatment
- AEOUT: Outcome of adverse event
- AESEV: Severity/Intensity
- NRIND: Normal range indicators

## Validation

Run validation with:
```bash
python pipeline/validate_integrated_kb.py --kb-dir kb/sdtmig_v3_4_complete
```
