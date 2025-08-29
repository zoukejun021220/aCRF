"""
CDISC Variable-Domain Mapper Module
Easy-to-use functions for CDISC variable to domain mapping
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, List, Optional, Union, Tuple
from dotenv import load_dotenv
from cdisc_api_client import CDISCAPIClient
from domain_metadata import DomainMetadata


class CDISCMapper:
    """Main class for CDISC variable-domain mapping operations"""
    
    def __init__(self, api_key: Optional[str] = None, auto_load: bool = True):
        """
        Initialize the CDISC Mapper
        
        Args:
            api_key: CDISC API key (optional, will use env variable if not provided)
            auto_load: Automatically load existing mappings if available
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize mappings
        self.sdtm_map = {}
        self.adam_map = {}
        
        # Initialize domain metadata
        self.domain_metadata = DomainMetadata(auto_load=auto_load)
        
        # Store API key
        self.api_key = api_key or os.environ.get("CDISC_API_KEY")
        
        # Auto-load existing mappings if requested
        if auto_load:
            self._load_existing_mappings()
    
    def _load_existing_mappings(self):
        """Load existing mapping files if they exist"""
        if Path("sdtm_variable_map.json").exists():
            with open("sdtm_variable_map.json", 'r') as f:
                data = json.load(f)
                self.sdtm_map = {k: set(v) for k, v in data.items()}
        
        if Path("adam_variable_map.json").exists():
            with open("adam_variable_map.json", 'r') as f:
                data = json.load(f)
                self.adam_map = {k: set(v) for k, v in data.items()}
    
    def build_mappings(self, standards: List[str] = ["sdtm"], force_rebuild: bool = False) -> Dict[str, Dict]:
        """
        Build or rebuild mappings from CDISC API
        
        Args:
            standards: List of standards to build ("sdtm", "adam")
            force_rebuild: Force rebuild even if mappings exist
            
        Returns:
            Dictionary with built mappings
        """
        if not self.api_key:
            raise ValueError("API key required. Set CDISC_API_KEY or pass api_key to constructor")
        
        client = CDISCAPIClient(self.api_key)
        results = {}
        
        for standard in standards:
            if standard.lower() == "sdtm":
                if not self.sdtm_map or force_rebuild:
                    print(f"Building SDTM mapping...")
                    self.sdtm_map = client.build_variable_domain_map("sdtm", "1-8")
                    client.save_mapping(self.sdtm_map, "sdtm_variable_map.json")
                results["sdtm"] = self.sdtm_map
                
            elif standard.lower() == "adam":
                if not self.adam_map or force_rebuild:
                    try:
                        print(f"Building ADaM mapping...")
                        self.adam_map = client.build_variable_domain_map("adam", "1-3")
                        client.save_mapping(self.adam_map, "adam_variable_map.json")
                    except:
                        print("ADaM mapping not available")
                        self.adam_map = {}
                results["adam"] = self.adam_map
        
        return results
    
    def get_domains(self, variable: str, standard: Optional[str] = None) -> Union[Set[str], Dict[str, Set[str]]]:
        """
        Get domains for a variable
        
        Args:
            variable: Variable name (case-insensitive)
            standard: Optional - "sdtm" or "adam" to limit search
            
        Returns:
            Set of domains if standard specified, otherwise dict with all matches
            
        Example:
            >>> mapper.get_domains("USUBJID")
            {'sdtm': {'DM', 'CO', 'SE', ...}}
            
            >>> mapper.get_domains("AGE", standard="sdtm")
            {'DM'}
        """
        variable_upper = variable.upper()
        
        if standard:
            if standard.lower() == "sdtm":
                return self.sdtm_map.get(variable_upper, set())
            elif standard.lower() == "adam":
                return self.adam_map.get(variable_upper, set())
        else:
            results = {}
            if variable_upper in self.sdtm_map:
                results["sdtm"] = self.sdtm_map[variable_upper]
            if variable_upper in self.adam_map:
                results["adam"] = self.adam_map[variable_upper]
            return results
    
    def get_variables(self, domain: str, standard: str = "sdtm") -> Set[str]:
        """
        Get all variables in a domain
        
        Args:
            domain: Domain code (e.g., "DM", "AE")
            standard: Standard to search ("sdtm" or "adam")
            
        Returns:
            Set of variable names in the domain
            
        Example:
            >>> mapper.get_variables("DM")
            {'USUBJID', 'STUDYID', 'AGE', 'SEX', ...}
        """
        domain_upper = domain.upper()
        variables = set()
        
        mapping = self.sdtm_map if standard.lower() == "sdtm" else self.adam_map
        
        for var, domains in mapping.items():
            if domain_upper in domains:
                variables.add(var)
        
        return variables
    
    def search_variables(self, pattern: str, standard: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Search for variables containing a pattern
        
        Args:
            pattern: Pattern to search for (case-insensitive)
            standard: Optional - limit to "sdtm" or "adam"
            
        Returns:
            Dictionary of matching variables and their domains
            
        Example:
            >>> mapper.search_variables("AE")
            {'AESEV': {'AE'}, 'AETERM': {'AE'}, ...}
        """
        pattern_upper = pattern.upper()
        matches = {}
        
        if standard is None or standard.lower() == "sdtm":
            for var, domains in self.sdtm_map.items():
                if pattern_upper in var:
                    matches[var] = domains
        
        if standard is None or standard.lower() == "adam":
            for var, domains in self.adam_map.items():
                if pattern_upper in var:
                    if var in matches:
                        matches[var] = matches[var].union(domains)
                    else:
                        matches[var] = domains
        
        return matches
    
    def is_multihome(self, variable: str, standard: str = "sdtm") -> bool:
        """
        Check if a variable appears in multiple domains
        
        Args:
            variable: Variable name
            standard: Standard to check
            
        Returns:
            True if variable appears in multiple domains
            
        Example:
            >>> mapper.is_multihome("USUBJID")
            True
            >>> mapper.is_multihome("AGE")
            False
        """
        domains = self.get_domains(variable, standard)
        if isinstance(domains, set):
            return len(domains) > 1
        return False
    
    def get_mapping_stats(self) -> Dict:
        """
        Get statistics about loaded mappings
        
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            "sdtm": {
                "total_variables": len(self.sdtm_map),
                "total_domains": len(set(d for domains in self.sdtm_map.values() for d in domains)),
                "multihome_variables": sum(1 for domains in self.sdtm_map.values() if len(domains) > 1)
            },
            "adam": {
                "total_variables": len(self.adam_map),
                "total_domains": len(set(d for domains in self.adam_map.values() for d in domains)),
                "multihome_variables": sum(1 for domains in self.adam_map.values() if len(domains) > 1)
            }
        }
        return stats
    
    def get_all_domains(self, standard: Optional[str] = None) -> Set[str]:
        """
        Get all unique domain codes
        
        Args:
            standard: Optional - "sdtm" or "adam" to limit to one standard
            
        Returns:
            Set of all domain codes
        """
        domains = set()
        
        if standard is None or standard.lower() == "sdtm":
            for domain_set in self.sdtm_map.values():
                domains.update(domain_set)
        
        if standard is None or standard.lower() == "adam":
            for domain_set in self.adam_map.values():
                domains.update(domain_set)
        
        return domains
    
    def is_domain(self, code: str, standard: Optional[str] = None) -> bool:
        """
        Check if a code is a valid domain
        
        Args:
            code: Code to check (e.g., "AE", "DM")
            standard: Optional - "sdtm" or "adam" to check specific standard
            
        Returns:
            True if code is a valid domain, False otherwise
            
        Example:
            >>> mapper.is_domain("DM")
            True
            >>> mapper.is_domain("XYZ")
            False
        """
        code_upper = code.upper()
        all_domains = self.get_all_domains(standard)
        return code_upper in all_domains
    
    def get_domain_name(self, code: str) -> Optional[str]:
        """
        Get the full name for a domain code
        
        Args:
            code: Domain code (e.g., "AE")
            
        Returns:
            Full domain name or None
            
        Example:
            >>> mapper.get_domain_name("AE")
            "Adverse Events"
        """
        return self.domain_metadata.get_domain_name(code)
    
    def get_domain_info(self, code: str) -> Optional[Dict]:
        """
        Get complete metadata for a domain
        
        Args:
            code: Domain code
            
        Returns:
            Domain metadata including name, class, type
        """
        return self.domain_metadata.get_domain_info(code)
    
    def get_all_domain_names(self) -> Dict[str, str]:
        """
        Get mapping of all domain codes to names
        
        Returns:
            Dictionary mapping codes to names
        """
        return self.domain_metadata.get_all_domain_names()


# Convenience functions for direct import and use

_default_mapper = None

def init_mapper(api_key: Optional[str] = None, auto_load: bool = True) -> CDISCMapper:
    """
    Initialize the global mapper instance
    
    Args:
        api_key: CDISC API key
        auto_load: Auto-load existing mappings
        
    Returns:
        CDISCMapper instance
    """
    global _default_mapper
    _default_mapper = CDISCMapper(api_key, auto_load)
    return _default_mapper


def get_mapper() -> CDISCMapper:
    """Get or create the default mapper instance"""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = CDISCMapper(auto_load=True)
    return _default_mapper


def lookup_variable(variable: str, standard: Optional[str] = None) -> Union[Set[str], Dict[str, Set[str]]]:
    """
    Quick lookup function for variable domains
    
    Args:
        variable: Variable name to lookup
        standard: Optional standard ("sdtm" or "adam")
        
    Returns:
        Set of domains or dict of standards->domains
        
    Example:
        >>> from cdisc_mapper import lookup_variable
        >>> lookup_variable("USUBJID")
        {'sdtm': {'DM', 'CO', 'SE', ...}}
        
        >>> lookup_variable("AGE", "sdtm")
        {'DM'}
    """
    mapper = get_mapper()
    return mapper.get_domains(variable, standard)


def list_domain_variables(domain: str, standard: str = "sdtm") -> Set[str]:
    """
    List all variables in a domain
    
    Args:
        domain: Domain code
        standard: Standard (default "sdtm")
        
    Returns:
        Set of variable names
        
    Example:
        >>> from cdisc_mapper import list_domain_variables
        >>> list_domain_variables("DM")
        {'USUBJID', 'STUDYID', 'AGE', 'SEX', ...}
    """
    mapper = get_mapper()
    return mapper.get_variables(domain, standard)


def search_variables(pattern: str, standard: Optional[str] = None) -> Dict[str, Set[str]]:
    """
    Search for variables containing a pattern
    
    Args:
        pattern: Search pattern
        standard: Optional standard filter
        
    Returns:
        Dict of matching variables and their domains
        
    Example:
        >>> from cdisc_mapper import search_variables
        >>> search_variables("AE")
        {'AESEV': {'AE'}, 'AETERM': {'AE'}, ...}
    """
    mapper = get_mapper()
    return mapper.search_variables(pattern, standard)


def build_mappings(standards: List[str] = ["sdtm"], force: bool = False) -> Dict:
    """
    Build or update mappings from CDISC API
    
    Args:
        standards: List of standards to build
        force: Force rebuild even if exists
        
    Returns:
        Built mappings
        
    Example:
        >>> from cdisc_mapper import build_mappings
        >>> build_mappings(["sdtm"])
    """
    mapper = get_mapper()
    return mapper.build_mappings(standards, force)


def get_stats() -> Dict:
    """
    Get mapping statistics
    
    Returns:
        Statistics dictionary
        
    Example:
        >>> from cdisc_mapper import get_stats
        >>> stats = get_stats()
        >>> print(f"Total SDTM variables: {stats['sdtm']['total_variables']}")
    """
    mapper = get_mapper()
    return mapper.get_mapping_stats()


# Batch processing functions

def lookup_multiple(variables: List[str], standard: Optional[str] = None) -> Dict[str, Union[Set[str], Dict]]:
    """
    Lookup multiple variables at once
    
    Args:
        variables: List of variable names
        standard: Optional standard filter
        
    Returns:
        Dictionary mapping each variable to its domains
        
    Example:
        >>> from cdisc_mapper import lookup_multiple
        >>> lookup_multiple(["AGE", "SEX", "USUBJID"], "sdtm")
        {'AGE': {'DM'}, 'SEX': {'DM'}, 'USUBJID': {'DM', 'CO', ...}}
    """
    mapper = get_mapper()
    results = {}
    for var in variables:
        results[var] = mapper.get_domains(var, standard)
    return results


def find_common_domains(variables: List[str], standard: str = "sdtm") -> Set[str]:
    """
    Find domains that contain ALL specified variables
    
    Args:
        variables: List of variable names
        standard: Standard to search
        
    Returns:
        Set of domains containing all variables
        
    Example:
        >>> from cdisc_mapper import find_common_domains
        >>> find_common_domains(["AGE", "SEX", "RACE"])
        {'DM'}
    """
    mapper = get_mapper()
    if not variables:
        return set()
    
    # Get domains for first variable
    common = mapper.get_domains(variables[0], standard)
    if isinstance(common, dict):
        common = common.get(standard, set())
    
    # Intersect with domains of other variables
    for var in variables[1:]:
        domains = mapper.get_domains(var, standard)
        if isinstance(domains, dict):
            domains = domains.get(standard, set())
        common = common.intersection(domains)
    
    return common


def is_valid_domain(code: str, standard: Optional[str] = None) -> bool:
    """
    Check if a code is a valid CDISC domain
    
    Args:
        code: Code to check (e.g., "AE", "DM", "VS")
        standard: Optional - "sdtm" or "adam" to check specific standard
        
    Returns:
        True if code is a valid domain, False otherwise
        
    Example:
        >>> from cdisc_mapper import is_valid_domain
        >>> is_valid_domain("DM")
        True
        >>> is_valid_domain("AE")  # Returns False if AE not in loaded datasets
        False
        >>> is_valid_domain("XYZ")
        False
    """
    mapper = get_mapper()
    return mapper.is_domain(code, standard)


def get_all_domains(standard: Optional[str] = None) -> Set[str]:
    """
    Get all available domain codes
    
    Args:
        standard: Optional - "sdtm" or "adam" to get domains for specific standard
        
    Returns:
        Set of all domain codes
        
    Example:
        >>> from cdisc_mapper import get_all_domains
        >>> domains = get_all_domains("sdtm")
        >>> print(sorted(domains))
        ['AC', 'APRELSUB', 'CO', 'DI', 'DM', 'DR', ...]
    """
    mapper = get_mapper()
    return mapper.get_all_domains(standard)


def get_domain_name(code: str) -> Optional[str]:
    """
    Get the full name for a domain code
    
    Args:
        code: Domain code (e.g., "AE", "CM", "VS")
        
    Returns:
        Full domain name or None if not found
        
    Example:
        >>> from cdisc_mapper import get_domain_name
        >>> get_domain_name("AE")
        "Adverse Events"
        >>> get_domain_name("CM")
        "Concomitant/Prior Medications"
        >>> get_domain_name("VS")
        "Vital Signs"
    """
    mapper = get_mapper()
    return mapper.get_domain_name(code)


def get_domain_info(code: str) -> Optional[Dict]:
    """
    Get complete metadata for a domain
    
    Args:
        code: Domain code
        
    Returns:
        Domain metadata including name, class, type
        
    Example:
        >>> from cdisc_mapper import get_domain_info
        >>> info = get_domain_info("AE")
        >>> print(f"{info['name']} - Class: {info['class']}")
        "Adverse Events - Class: Events"
    """
    mapper = get_mapper()
    return mapper.get_domain_info(code)


def get_all_domain_names() -> Dict[str, str]:
    """
    Get mapping of all domain codes to their full names
    
    Returns:
        Dictionary mapping domain codes to full names
        
    Example:
        >>> from cdisc_mapper import get_all_domain_names
        >>> names = get_all_domain_names()
        >>> print(names["AE"])
        "Adverse Events"
        >>> print(names["LB"])
        "Laboratory Test Results"
    """
    mapper = get_mapper()
    return mapper.get_all_domain_names()


if __name__ == "__main__":
    # Example usage
    print("CDISC Mapper Module - Example Usage")
    print("=" * 50)
    
    # Initialize mapper
    mapper = CDISCMapper()
    
    # Example lookups
    print("\nExample 1: Single variable lookup")
    domains = lookup_variable("USUBJID", "sdtm")
    print(f"USUBJID domains: {domains}")
    
    print("\nExample 2: Search for variables")
    matches = search_variables("AGE")
    print(f"Variables containing 'AGE': {list(matches.keys())}")
    
    print("\nExample 3: List domain variables")
    dm_vars = list_domain_variables("DM")
    print(f"DM domain has {len(dm_vars)} variables")
    
    print("\nExample 4: Batch lookup")
    batch_results = lookup_multiple(["AGE", "SEX", "RACE"], "sdtm")
    for var, doms in batch_results.items():
        print(f"  {var}: {doms}")
    
    print("\nExample 5: Find common domains")
    common = find_common_domains(["AGE", "SEX", "RACE"], "sdtm")
    print(f"Domains with AGE, SEX, and RACE: {common}")
    
    print("\nExample 6: Statistics")
    stats = get_stats()
    print(f"Total SDTM variables: {stats['sdtm']['total_variables']}")
    print(f"Multi-home variables: {stats['sdtm']['multihome_variables']}")