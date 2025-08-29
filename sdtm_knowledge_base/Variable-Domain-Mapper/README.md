# CDISC Variable-Domain Mapper

Automatically map CDISC variable names to their domains using the CDISC Library REST API.

## Features

- **Complete SDTM Coverage**: 1,500+ variables across 63 domains (AE, CM, VS, LB, etc.)
- **Domain Name Mapping**: Maps domain codes to full names (e.g., "AE" → "Adverse Events")
- **Variable-Domain Lookup**: Find which domain(s) a variable belongs to
- **Domain Validation**: Check if a code is a valid CDISC domain
- **Multi-home Variable Support**: Handles variables that appear in multiple domains
- **Domain Classification**: Groups domains by SDTM classes (Events, Findings, Interventions, etc.)
- **Caching**: 7-day cache for offline use and performance
- **Command-line & Python API**: Use as CLI tool or import as Python module

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your CDISC API key (get one free at https://library.cdisc.org):
   - Edit `.env` file with your API key, or
   - Set environment variable: `export CDISC_API_KEY=your_key_here`

3. Build the initial mappings:
```bash
python variable_lookup.py --build
```

This will fetch all SDTM v1.8 and ADaM v2.0 metadata and create local mapping files.

## Usage

### Command Line Usage

```bash
# Look up a single variable
python variable_lookup.py USUBJID

# Search for variables
python variable_lookup.py --search AGE

# List all variables in a domain
python variable_lookup.py --domain DM

# Build/update mappings from API
python variable_lookup.py --build
```

### Python Module Usage

```python
# Quick import and use
from cdisc_mapper import lookup_variable, search_variables, list_domain_variables

# Look up a single variable
domains = lookup_variable("USUBJID", "sdtm")
print(f"USUBJID found in: {domains}")
# Output: {'DM', 'CO', 'SE', 'SV', ...}

# Search for variables containing a pattern
age_vars = search_variables("AGE")
print(f"Variables with 'AGE': {age_vars}")
# Output: {'AGE': {'DM'}, 'AGETXT': {'DM'}, ...}

# List all variables in a domain
dm_variables = list_domain_variables("DM")
print(f"DM domain variables: {dm_variables}")
# Output: {'USUBJID', 'STUDYID', 'AGE', 'SEX', ...}
```

### Domain Name Mapping

```python
from cdisc_mapper import get_domain_name, get_domain_info, get_all_domain_names

# Get full name for a domain code
name = get_domain_name("AE")
print(name)  # "Adverse Events"

name = get_domain_name("CM") 
print(name)  # "Concomitant/Prior Medications"

# Get complete domain information
info = get_domain_info("AE")
print(f"{info['name']} - Class: {info['class']}")
# Output: "Adverse Events - Class: Events"

# Get all domain names as dictionary
all_names = get_all_domain_names()
print(all_names["VS"])  # "Vital Signs"
print(all_names["LB"])  # "Laboratory Test Results"
```

### Advanced Usage

```python
from cdisc_mapper import CDISCMapper, lookup_multiple, find_common_domains
from cdisc_mapper import is_valid_domain, get_all_domains

# Check if a code is a valid domain
is_domain = is_valid_domain("DM")  # Returns True
is_domain = is_valid_domain("AE")  # Returns True (now with complete SDTMIG)
is_domain = is_valid_domain("USUBJID")  # Returns False (it's a variable, not domain)

# Get all valid domain codes
all_domains = get_all_domains("sdtm")
print(f"Valid domains: {sorted(all_domains)}")
# Output: ['AC', 'APRELSUB', 'CO', 'DI', 'DM', ...]

# Batch lookup multiple variables
results = lookup_multiple(["AGE", "SEX", "RACE"], "sdtm")
# Output: {'AGE': {'DM'}, 'SEX': {'DM'}, 'RACE': {'DM'}}

# Find domains containing all specified variables
common = find_common_domains(["AGE", "SEX", "RACE"])
print(f"Common domains: {common}")
# Output: {'DM'}

# Use the class directly for more control
mapper = CDISCMapper()
is_multihome = mapper.is_multihome("USUBJID")
print(f"Is USUBJID multi-home? {is_multihome}")
# Output: True

# Get mapping statistics
stats = mapper.get_mapping_stats()
print(f"Total variables: {stats['sdtm']['total_variables']}")
# Output: 153
```

### Integration in Your Project

```python
# In your data processing script
import sys
sys.path.append('/path/to/Variable-Domain-Mapper')

from cdisc_mapper import lookup_variable

def validate_variable_domain(var_name, domain_code):
    """Check if a variable belongs to a specific domain"""
    domains = lookup_variable(var_name, "sdtm")
    return domain_code in domains

# Example usage
if validate_variable_domain("AGE", "DM"):
    print("AGE is correctly placed in DM domain")
```

## Files

- `cdisc_api_client.py` - CDISC Library API client with caching
- `variable_lookup.py` - Command-line lookup tool
- `sdtm_variable_map.json` - Generated SDTM variable mappings
- `adam_variable_map.json` - Generated ADaM variable mappings
- `cache/` - Cached API responses

## Notes

- API responses are cached for 7 days to reduce API calls
- The tool respects CDISC API rate limits
- Multi-home variables (like USUBJID) will show all possible domains
- First build may take several minutes to fetch all metadata

## API Documentation

See https://library.cdisc.org/api/doc for complete CDISC Library API documentation.