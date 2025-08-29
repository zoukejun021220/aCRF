"""
CDISC Variable-Domain Mapper Package
Easy-to-use functions for CDISC variable to domain mapping
"""

from .cdisc_mapper import (
    # Main class
    CDISCMapper,
    
    # Initialization functions
    init_mapper,
    get_mapper,
    
    # Core lookup functions
    lookup_variable,
    list_domain_variables,
    search_variables,
    
    # Domain validation functions
    is_valid_domain,
    get_all_domains,
    
    # Domain metadata functions
    get_domain_name,
    get_domain_info,
    get_all_domain_names,
    
    # Batch operations
    lookup_multiple,
    find_common_domains,
    
    # Utility functions
    build_mappings,
    get_stats
)

# Import domain metadata functions too
from .domain_metadata import (
    get_domains_by_class,
    search_domains
)

__version__ = "1.0.0"

__all__ = [
    "CDISCMapper",
    "init_mapper",
    "get_mapper",
    "lookup_variable", 
    "list_domain_variables",
    "search_variables",
    "is_valid_domain",
    "get_all_domains",
    "get_domain_name",
    "get_domain_info",
    "get_all_domain_names",
    "get_domains_by_class",
    "search_domains",
    "lookup_multiple",
    "find_common_domains",
    "build_mappings",
    "get_stats"
]