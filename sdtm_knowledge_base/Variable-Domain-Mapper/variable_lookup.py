#!/usr/bin/env python3
"""
Variable Domain Lookup Tool
Quickly lookup which CDISC domain(s) a variable belongs to
"""

import json
import argparse
from pathlib import Path
from typing import Set, Dict, Optional
import os
from dotenv import load_dotenv
from cdisc_api_client import CDISCAPIClient


class VariableDomainLookup:
    """Lookup tool for CDISC variable to domain mapping"""
    
    def __init__(self, mapping_file: Optional[str] = None):
        """Initialize the lookup tool
        
        Args:
            mapping_file: Path to the mapping JSON file. If not provided, will try to load default files.
        """
        self.sdtm_map = {}
        self.adam_map = {}
        
        if mapping_file:
            # Load specific mapping file
            self.load_mapping(mapping_file)
        else:
            # Try to load default mapping files
            self.load_default_mappings()
    
    def load_mapping(self, filename: str, standard: Optional[str] = None) -> Dict[str, Set[str]]:
        """Load a mapping file
        
        Args:
            filename: Path to the mapping file
            standard: Optional standard name ("sdtm" or "adam")
        
        Returns:
            The loaded mapping
        """
        if not Path(filename).exists():
            print(f"Warning: Mapping file {filename} not found")
            return {}
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists to sets
        mapping = {k: set(v) for k, v in data.items()}
        
        # Store in appropriate attribute if standard is specified
        if standard:
            if standard.lower() == "sdtm":
                self.sdtm_map = mapping
            elif standard.lower() == "adam":
                self.adam_map = mapping
        
        return mapping
    
    def load_default_mappings(self):
        """Load default SDTM and ADaM mapping files if they exist"""
        if Path("sdtm_variable_map.json").exists():
            self.sdtm_map = self.load_mapping("sdtm_variable_map.json", "sdtm")
            print(f"Loaded SDTM mapping: {len(self.sdtm_map)} variables")
        
        if Path("adam_variable_map.json").exists():
            self.adam_map = self.load_mapping("adam_variable_map.json", "adam")
            print(f"Loaded ADaM mapping: {len(self.adam_map)} variables")
    
    def lookup_variable(self, variable: str, standard: Optional[str] = None) -> Dict[str, Set[str]]:
        """Lookup a variable and return its domains
        
        Args:
            variable: Variable name to lookup (case-insensitive)
            standard: Optional - specify "sdtm" or "adam" to search only that standard
        
        Returns:
            Dictionary with keys "sdtm" and/or "adam" containing sets of domain codes
        """
        variable_upper = variable.upper()
        results = {}
        
        if standard is None or standard.lower() == "sdtm":
            if variable_upper in self.sdtm_map:
                results["sdtm"] = self.sdtm_map[variable_upper]
        
        if standard is None or standard.lower() == "adam":
            if variable_upper in self.adam_map:
                results["adam"] = self.adam_map[variable_upper]
        
        return results
    
    def search_variables(self, pattern: str, standard: Optional[str] = None) -> Dict[str, Dict[str, Set[str]]]:
        """Search for variables matching a pattern
        
        Args:
            pattern: Pattern to search for (case-insensitive substring match)
            standard: Optional - specify "sdtm" or "adam" to search only that standard
        
        Returns:
            Dictionary of matching variables and their domains
        """
        pattern_upper = pattern.upper()
        matches = {}
        
        if standard is None or standard.lower() == "sdtm":
            for var, domains in self.sdtm_map.items():
                if pattern_upper in var:
                    if var not in matches:
                        matches[var] = {}
                    matches[var]["sdtm"] = domains
        
        if standard is None or standard.lower() == "adam":
            for var, domains in self.adam_map.items():
                if pattern_upper in var:
                    if var not in matches:
                        matches[var] = {}
                    matches[var]["adam"] = domains
        
        return matches
    
    def list_domain_variables(self, domain: str, standard: str = "sdtm") -> Set[str]:
        """List all variables in a specific domain
        
        Args:
            domain: Domain code (e.g., "AE", "DM")
            standard: Standard to search in ("sdtm" or "adam")
        
        Returns:
            Set of variable names in that domain
        """
        domain_upper = domain.upper()
        variables = set()
        
        mapping = self.sdtm_map if standard.lower() == "sdtm" else self.adam_map
        
        for var, domains in mapping.items():
            if domain_upper in domains:
                variables.add(var)
        
        return variables


def build_mappings():
    """Build new mappings from the CDISC API"""
    load_dotenv()
    
    print("Building new variable-domain mappings from CDISC API...")
    print("This may take several minutes on first run.")
    print()
    
    try:
        client = CDISCAPIClient()
        
        # Build SDTM mapping
        print("Building SDTM v1.8 mapping...")
        sdtm_map = client.build_variable_domain_map("sdtm", "1-8")
        client.save_mapping(sdtm_map, "sdtm_variable_map.json")
        
        # Build ADaM mapping  
        print("\nBuilding ADaM v2.0 mapping...")
        adam_map = client.build_variable_domain_map("adam", "2-0")
        client.save_mapping(adam_map, "adam_variable_map.json")
        
        print("\nMappings built successfully!")
        
    except Exception as e:
        print(f"Error building mappings: {e}")
        return False
    
    return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CDISC Variable Domain Lookup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AESEV                    # Lookup single variable
  %(prog)s AESEV --standard sdtm    # Lookup in SDTM only
  %(prog)s --search AE              # Search for variables containing "AE"
  %(prog)s --domain AE              # List all variables in AE domain
  %(prog)s --build                  # Build/update mappings from API
        """
    )
    
    parser.add_argument(
        "variable",
        nargs="?",
        help="Variable name to lookup"
    )
    
    parser.add_argument(
        "--standard",
        choices=["sdtm", "adam"],
        help="Limit search to specific standard"
    )
    
    parser.add_argument(
        "--search",
        metavar="PATTERN",
        help="Search for variables containing pattern"
    )
    
    parser.add_argument(
        "--domain",
        metavar="CODE",
        help="List all variables in a domain"
    )
    
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build/update mappings from CDISC API"
    )
    
    parser.add_argument(
        "--mapping-file",
        help="Use specific mapping file"
    )
    
    args = parser.parse_args()
    
    # Handle build command
    if args.build:
        build_mappings()
        return
    
    # Initialize lookup tool
    lookup = VariableDomainLookup(args.mapping_file)
    
    # Check if any mappings were loaded
    if not lookup.sdtm_map and not lookup.adam_map:
        print("No mapping files found. Run with --build to fetch from CDISC API.")
        return
    
    # Handle domain listing
    if args.domain:
        standard = args.standard or "sdtm"
        variables = lookup.list_domain_variables(args.domain, standard)
        
        if variables:
            print(f"\nVariables in {args.domain} domain ({standard.upper()}):")
            for var in sorted(variables):
                print(f"  {var}")
        else:
            print(f"No variables found in domain {args.domain}")
        return
    
    # Handle search
    if args.search:
        matches = lookup.search_variables(args.search, args.standard)
        
        if matches:
            print(f"\nVariables matching '{args.search}':")
            for var, standards in sorted(matches.items()):
                for std, domains in standards.items():
                    print(f"  {var} ({std.upper()}): {', '.join(sorted(domains))}")
        else:
            print(f"No variables found matching '{args.search}'")
        return
    
    # Handle single variable lookup
    if args.variable:
        results = lookup.lookup_variable(args.variable, args.standard)
        
        if results:
            print(f"\nDomains for variable '{args.variable.upper()}':")
            for standard, domains in results.items():
                if len(domains) == 1:
                    print(f"  {standard.upper()}: {list(domains)[0]}")
                else:
                    print(f"  {standard.upper()}: {', '.join(sorted(domains))} (multi-home variable)")
        else:
            print(f"Variable '{args.variable.upper()}' not found")
            
            # Try to suggest similar variables
            similar = lookup.search_variables(args.variable[:3] if len(args.variable) >= 3 else args.variable)
            if similar:
                print("\nDid you mean one of these?")
                for var in sorted(list(similar.keys())[:5]):
                    print(f"  {var}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()