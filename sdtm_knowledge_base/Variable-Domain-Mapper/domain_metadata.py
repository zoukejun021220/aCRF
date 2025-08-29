"""
CDISC Domain Metadata Module
Maps domain codes to their full names and descriptions
"""

import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from cdisc_api_client import CDISCAPIClient
import os


class DomainMetadata:
    """Manages domain code to full name mappings"""
    
    def __init__(self, auto_load: bool = True):
        """Initialize domain metadata
        
        Args:
            auto_load: Automatically load existing metadata if available
        """
        self.domain_map = {}
        self.metadata_file = "domain_metadata.json"
        
        if auto_load and Path(self.metadata_file).exists():
            self.load_metadata()
    
    def build_domain_metadata(self, api_key: Optional[str] = None) -> Dict[str, Dict]:
        """Build domain metadata from CDISC API
        
        Args:
            api_key: CDISC API key (optional, will use env variable if not provided)
            
        Returns:
            Dictionary mapping domain codes to metadata
        """
        api_key = api_key or os.environ.get("CDISC_API_KEY")
        if not api_key:
            raise ValueError("API key required for building metadata")
        
        client = CDISCAPIClient(api_key)
        
        # Get SDTMIG datasets which have full names
        print("Fetching domain metadata from CDISC API...")
        datasets = client.get_sdtmig_datasets("3-4")
        
        self.domain_map = {}
        
        for dataset in datasets:
            # Extract domain code from href (last part)
            href = dataset.get("href", "")
            if href:
                domain_code = href.split("/")[-1]
                full_name = dataset.get("title", "")
                dataset_type = dataset.get("type", "SDTM Dataset")
                
                self.domain_map[domain_code] = {
                    "code": domain_code,
                    "name": full_name,
                    "type": dataset_type,
                    "href": href
                }
        
        # Add domain class information for better categorization
        self._add_domain_classes()
        
        print(f"Built metadata for {len(self.domain_map)} domains")
        return self.domain_map
    
    def _add_domain_classes(self):
        """Add domain class categorization based on SDTM model"""
        
        # Define domain classes based on SDTM structure
        domain_classes = {
            "Interventions": ["AG", "CM", "EC", "EX", "ML", "PR", "SU"],
            "Events": ["AE", "BE", "CE", "DS", "DV", "HO", "MH"],
            "Findings": ["BS", "CP", "CV", "DA", "DD", "EG", "FT", "GF", "IE", "IS", 
                        "LB", "MB", "MI", "MK", "MS", "NV", "OE", "PC", "PE", "PP", 
                        "QS", "RE", "RP", "RS", "SC", "SS", "TR", "TU", "UR", "VS"],
            "Findings About": ["FA", "SR"],
            "Special Purpose": ["CO", "DM", "SE", "SV"],
            "Trial Design": ["TA", "TD", "TE", "TI", "TM", "TS", "TV"],
            "Relationship": ["RELREC", "RELSPEC", "RELSUB", "SUPPQUAL"],
            "Study Reference": ["DI", "OI"]
        }
        
        # Add class information to each domain
        for class_name, domains in domain_classes.items():
            for domain in domains:
                if domain in self.domain_map:
                    self.domain_map[domain]["class"] = class_name
    
    def save_metadata(self, filename: Optional[str] = None):
        """Save domain metadata to JSON file
        
        Args:
            filename: Output filename (default: domain_metadata.json)
        """
        filename = filename or self.metadata_file
        with open(filename, 'w') as f:
            json.dump(self.domain_map, f, indent=2, sort_keys=True)
        print(f"Saved domain metadata to {filename}")
    
    def load_metadata(self, filename: Optional[str] = None) -> Dict:
        """Load domain metadata from JSON file
        
        Args:
            filename: Input filename (default: domain_metadata.json)
            
        Returns:
            Domain metadata dictionary
        """
        filename = filename or self.metadata_file
        with open(filename, 'r') as f:
            self.domain_map = json.load(f)
        return self.domain_map
    
    def get_domain_name(self, code: str) -> Optional[str]:
        """Get full name for a domain code
        
        Args:
            code: Domain code (e.g., "AE")
            
        Returns:
            Full domain name or None if not found
        """
        domain = self.domain_map.get(code.upper())
        return domain["name"] if domain else None
    
    def get_domain_info(self, code: str) -> Optional[Dict]:
        """Get complete metadata for a domain
        
        Args:
            code: Domain code
            
        Returns:
            Domain metadata dictionary or None
        """
        return self.domain_map.get(code.upper())
    
    def get_domains_by_class(self, class_name: str) -> List[str]:
        """Get all domains in a specific class
        
        Args:
            class_name: Class name (e.g., "Events", "Findings")
            
        Returns:
            List of domain codes in that class
        """
        domains = []
        for code, info in self.domain_map.items():
            if info.get("class") == class_name:
                domains.append(code)
        return sorted(domains)
    
    def get_all_domain_names(self) -> Dict[str, str]:
        """Get simple mapping of all domain codes to names
        
        Returns:
            Dictionary mapping domain codes to full names
        """
        return {code: info["name"] for code, info in self.domain_map.items()}
    
    def search_domains(self, search_term: str) -> List[Tuple[str, str]]:
        """Search for domains by code or name
        
        Args:
            search_term: Search term (case-insensitive)
            
        Returns:
            List of (code, name) tuples matching the search
        """
        search_term = search_term.lower()
        results = []
        
        for code, info in self.domain_map.items():
            if (search_term in code.lower() or 
                search_term in info["name"].lower()):
                results.append((code, info["name"]))
        
        return sorted(results)
    
    def format_domain_list(self, by_class: bool = False) -> str:
        """Format domain list for display
        
        Args:
            by_class: Group domains by class
            
        Returns:
            Formatted string of domains
        """
        if not by_class:
            # Simple alphabetical list
            lines = []
            for code in sorted(self.domain_map.keys()):
                info = self.domain_map[code]
                lines.append(f"  {code:10} - {info['name']}")
            return "\n".join(lines)
        
        # Group by class
        classes = {}
        for code, info in self.domain_map.items():
            class_name = info.get("class", "Other")
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append((code, info["name"]))
        
        lines = []
        for class_name in sorted(classes.keys()):
            lines.append(f"\n{class_name}:")
            lines.append("-" * (len(class_name) + 1))
            for code, name in sorted(classes[class_name]):
                lines.append(f"  {code:10} - {name}")
        
        return "\n".join(lines)


# Convenience functions for easy import

_metadata = None

def get_metadata() -> DomainMetadata:
    """Get or create the metadata instance"""
    global _metadata
    if _metadata is None:
        _metadata = DomainMetadata(auto_load=True)
    return _metadata


def get_domain_name(code: str) -> Optional[str]:
    """Get the full name for a domain code
    
    Args:
        code: Domain code (e.g., "AE", "CM")
        
    Returns:
        Full domain name or None
        
    Example:
        >>> get_domain_name("AE")
        "Adverse Events"
        >>> get_domain_name("VS")
        "Vital Signs"
    """
    metadata = get_metadata()
    return metadata.get_domain_name(code)


def get_all_domain_names() -> Dict[str, str]:
    """Get mapping of all domain codes to names
    
    Returns:
        Dictionary mapping codes to names
        
    Example:
        >>> names = get_all_domain_names()
        >>> names["AE"]
        "Adverse Events"
    """
    metadata = get_metadata()
    return metadata.get_all_domain_names()


def get_domain_info(code: str) -> Optional[Dict]:
    """Get complete metadata for a domain
    
    Args:
        code: Domain code
        
    Returns:
        Domain metadata including name, class, type
        
    Example:
        >>> get_domain_info("AE")
        {'code': 'AE', 'name': 'Adverse Events', 'class': 'Events', ...}
    """
    metadata = get_metadata()
    return metadata.get_domain_info(code)


def get_domains_by_class(class_name: str) -> List[str]:
    """Get all domains in a specific SDTM class
    
    Args:
        class_name: Class name (Events, Findings, Interventions, etc.)
        
    Returns:
        List of domain codes in that class
        
    Example:
        >>> get_domains_by_class("Events")
        ['AE', 'BE', 'CE', 'DS', 'DV', 'HO', 'MH']
    """
    metadata = get_metadata()
    return metadata.get_domains_by_class(class_name)


def search_domains(term: str) -> List[Tuple[str, str]]:
    """Search for domains by code or name
    
    Args:
        term: Search term
        
    Returns:
        List of (code, name) tuples
        
    Example:
        >>> search_domains("adverse")
        [('AE', 'Adverse Events')]
    """
    metadata = get_metadata()
    return metadata.search_domains(term)


def build_domain_metadata(api_key: Optional[str] = None) -> Dict:
    """Build/rebuild domain metadata from API
    
    Args:
        api_key: CDISC API key
        
    Returns:
        Domain metadata dictionary
    """
    metadata = get_metadata()
    result = metadata.build_domain_metadata(api_key)
    metadata.save_metadata()
    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("CDISC Domain Metadata Builder")
    print("=" * 50)
    
    # Build metadata
    metadata = DomainMetadata()
    
    if not Path("domain_metadata.json").exists():
        print("\nBuilding domain metadata from API...")
        metadata.build_domain_metadata()
        metadata.save_metadata()
    else:
        print("\nLoading existing domain metadata...")
        metadata.load_metadata()
    
    # Display summary
    print(f"\nLoaded {len(metadata.domain_map)} domains")
    
    # Show some examples
    print("\nExample Domain Mappings:")
    examples = ["AE", "CM", "VS", "LB", "EX", "DM", "QS", "MH"]
    for code in examples:
        name = metadata.get_domain_name(code)
        if name:
            info = metadata.get_domain_info(code)
            class_name = info.get("class", "Unknown")
            print(f"  {code:4} → {name:35} [{class_name}]")
    
    # Show domains by class
    print("\nDomains by Class:")
    print(metadata.format_domain_list(by_class=True))