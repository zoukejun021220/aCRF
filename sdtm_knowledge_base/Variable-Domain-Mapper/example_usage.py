#!/usr/bin/env python3
"""
Example usage of the CDISC Mapper module
Demonstrates various ways to use the mapper functions
"""

# Import the convenience functions
from cdisc_mapper import (
    lookup_variable,
    list_domain_variables,
    search_variables,
    lookup_multiple,
    find_common_domains,
    get_stats,
    CDISCMapper
)


def example_simple_lookups():
    """Example 1: Simple variable lookups"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Variable Lookups")
    print("=" * 60)
    
    # Look up a single variable
    result = lookup_variable("USUBJID", "sdtm")
    print(f"\nUSUBJID appears in domains: {result}")
    
    # Look up another variable
    result = lookup_variable("AGE", "sdtm")
    print(f"AGE appears in domain: {result}")
    
    # Look up without specifying standard
    result = lookup_variable("STUDYID")
    print(f"\nSTUDYID across all standards: {result}")


def example_domain_exploration():
    """Example 2: Explore variables in a domain"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Domain Exploration")
    print("=" * 60)
    
    # Get all variables in DM domain
    dm_vars = list_domain_variables("DM", "sdtm")
    print(f"\nDM domain contains {len(dm_vars)} variables:")
    print(f"First 10: {sorted(list(dm_vars))[:10]}")
    
    # Get all variables in CO domain
    co_vars = list_domain_variables("CO", "sdtm")
    print(f"\nCO domain contains {len(co_vars)} variables:")
    print(f"Variables: {sorted(co_vars)}")


def example_search_patterns():
    """Example 3: Search for variables by pattern"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Pattern Search")
    print("=" * 60)
    
    # Search for all variables containing "DT"
    dt_vars = search_variables("DT", "sdtm")
    print(f"\nFound {len(dt_vars)} variables containing 'DT':")
    for var, domains in sorted(dt_vars.items())[:5]:
        print(f"  {var}: {domains}")
    
    # Search for all variables starting with "AE"
    ae_vars = {k: v for k, v in search_variables("AE", "sdtm").items() if k.startswith("AE")}
    print(f"\nVariables starting with 'AE':")
    for var, domains in sorted(ae_vars.items()):
        print(f"  {var}: {domains}")


def example_batch_operations():
    """Example 4: Batch operations"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Operations")
    print("=" * 60)
    
    # Look up multiple variables at once
    variables = ["AGE", "SEX", "RACE", "ETHNIC", "COUNTRY"]
    results = lookup_multiple(variables, "sdtm")
    
    print("\nBatch lookup results:")
    for var, domains in results.items():
        print(f"  {var}: {domains if domains else 'Not found'}")
    
    # Find common domains for a set of variables
    common = find_common_domains(["AGE", "SEX", "RACE"], "sdtm")
    print(f"\nDomains containing AGE, SEX, and RACE: {common}")


def example_class_usage():
    """Example 5: Using the CDISCMapper class directly"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Direct Class Usage")
    print("=" * 60)
    
    # Create a mapper instance
    mapper = CDISCMapper(auto_load=True)
    
    # Check if a variable is multi-home
    is_multi = mapper.is_multihome("USUBJID", "sdtm")
    print(f"\nIs USUBJID a multi-home variable? {is_multi}")
    
    is_multi = mapper.is_multihome("AGE", "sdtm")
    print(f"Is AGE a multi-home variable? {is_multi}")
    
    # Get statistics
    stats = mapper.get_mapping_stats()
    print(f"\nMapping Statistics:")
    print(f"  SDTM: {stats['sdtm']['total_variables']} variables across {stats['sdtm']['total_domains']} domains")
    print(f"  Multi-home variables: {stats['sdtm']['multihome_variables']}")


def example_practical_use_case():
    """Example 6: Practical use case - Variable validation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Practical Use Case - Variable Validation")
    print("=" * 60)
    
    # Scenario: Validate if variables are in the correct domain
    test_data = [
        ("AGE", "DM"),
        ("SEX", "DM"),
        ("AESEV", "AE"),  # This would fail as AE domain doesn't exist in our limited dataset
        ("USUBJID", "DM"),  # Multi-home variable, should pass
        ("STUDYID", "DM"),  # Multi-home variable, should pass
    ]
    
    print("\nValidating variable-domain pairs:")
    for var, expected_domain in test_data:
        domains = lookup_variable(var, "sdtm")
        if domains and expected_domain in domains:
            print(f"  ✓ {var} is valid in {expected_domain}")
        else:
            print(f"  ✗ {var} is NOT in {expected_domain} (found in: {domains})")


def example_domain_validation():
    """Example 7: Domain validation - Check if a code is a valid domain"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Domain Validation")
    print("=" * 60)
    
    from cdisc_mapper import is_valid_domain, get_all_domains
    
    # Get all valid domains
    all_domains = get_all_domains("sdtm")
    print(f"\nTotal valid SDTM domains: {len(all_domains)}")
    print(f"Examples: {sorted(list(all_domains))[:8]}")
    
    # Test if various codes are valid domains
    test_codes = ["DM", "CO", "AE", "USUBJID", "AGE", "XYZ"]
    
    print("\nChecking if codes are valid domains:")
    for code in test_codes:
        is_domain = is_valid_domain(code, "sdtm")
        if is_domain:
            print(f"  ✓ '{code}' is a valid SDTM domain")
        else:
            # Check if it's a variable instead
            var_domains = lookup_variable(code, "sdtm")
            if var_domains:
                print(f"  ✗ '{code}' is NOT a domain (it's a variable in: {sorted(list(var_domains)[:3])})")
            else:
                print(f"  ✗ '{code}' is NOT a valid domain")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("CDISC MAPPER - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    try:
        # Run all examples
        example_simple_lookups()
        example_domain_exploration()
        example_search_patterns()
        example_batch_operations()
        example_class_usage()
        example_practical_use_case()
        example_domain_validation()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have run 'python variable_lookup.py --build' first")


if __name__ == "__main__":
    main()