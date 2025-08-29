# Controlled Terminology Reference

**Source**: SDTM Controlled Terminology Package 59 Effective 2025-03-28
**Total Codelists**: 1158
**Total Terms**: 43698
**Common Categories Extracted**: 17

## Common Categories

- **AESEV**: 3 terms
- **AGEU**: 5 terms
- **DOMAIN**: 83 terms
- **DSCAT**: 3 terms
- **EPOCH**: 15 terms
- **ETHNIC**: 4 terms
- **EVAL**: 60 terms
- **FREQ**: 102 terms
- **FRM**: 196 terms
- **NRIND**: 4 terms
- **NY**: 4 terms
- **RACE**: 8 terms
- **ROUTE**: 142 terms
- **SEX**: 4 terms
- **TSPARMCD**: 129 terms
- **UNIT**: 929 terms
- **VSRESU**: 29 terms

## Example Terms

### SEX (Sex)
- F: Female
- M: Male
- U: Unknown
- INTERSEX: Intersex

### AESEV (Adverse Event Severity)
- MILD: Mild Adverse Event
- MODERATE: Moderate Adverse Event
- SEVERE: Severe Adverse Event

### AEOUT (Adverse Event Outcome)
- RECOVERED/RESOLVED
- RECOVERING/RESOLVING
- NOT RECOVERED/NOT RESOLVED
- RECOVERED/RESOLVED WITH SEQUELAE
- FATAL
- UNKNOWN

## Usage in Pipeline

```python
# Load common terms
with open('kb/sdtmig_v3_4_complete/common_ct_terms_complete.json') as f:
    ct_terms = json.load(f)

# Get terms for a category
sex_terms = ct_terms['SEX']
for term in sex_terms:
    print(f'{term["code"]}: {term["value"]}')
```

## Files

- `common_ct_terms_complete.json` - Common terms organized by category
- `variable_codelist_map.json` - Variable to codelist mapping
- `codelist_info.json` - Codelist metadata
- `cdisc_all_controlled_terms.json` - Complete database (43,698 terms)