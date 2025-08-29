import requests
import json
import os
from typing import Dict, Set, List, Optional
from pathlib import Path
import time
from datetime import datetime, timedelta


class CDISCAPIClient:
    """Client for interacting with CDISC Library REST API"""
    
    BASE_URL = "https://library.cdisc.org/api"
    CACHE_DIR = Path("cache")
    CACHE_EXPIRY_DAYS = 7
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the CDISC API client
        
        Args:
            api_key: CDISC API key. If not provided, will look for CDISC_API_KEY env variable
        """
        self.api_key = api_key or os.environ.get("CDISC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set CDISC_API_KEY environment variable or pass api_key parameter"
            )
        
        self.headers = {"api-key": self.api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Create cache directory if it doesn't exist
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def _get_cache_path(self, endpoint: str) -> Path:
        """Generate cache file path for an endpoint"""
        safe_name = endpoint.replace("/", "_").replace("?", "_")
        return self.CACHE_DIR / f"{safe_name}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is still valid"""
        if not cache_path.exists():
            return False
        
        # Check file modification time
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(days=self.CACHE_EXPIRY_DAYS)
        
        return file_time > expiry_time
    
    def _get_cached_or_fetch(self, endpoint: str) -> Dict:
        """Get data from cache if valid, otherwise fetch from API"""
        cache_path = self._get_cache_path(endpoint)
        
        # Check cache first
        if self._is_cache_valid(cache_path):
            print(f"Using cached data for {endpoint}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Fetch from API
        print(f"Fetching from API: {endpoint}")
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Be nice to the API - rate limiting
            time.sleep(0.5)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            # Try to return cached data even if expired
            if cache_path.exists():
                print("Returning expired cache due to API error")
                with open(cache_path, 'r') as f:
                    return json.load(f)
            raise
    
    def get_products(self) -> List[Dict]:
        """Get list of all CDISC products/packages"""
        data = self._get_cached_or_fetch("/mdr/products")
        return data.get("_links", {}).get("products", [])
    
    def get_sdtm_datasets(self, version: str = "1-8") -> List[Dict]:
        """Get SDTM datasets for a specific version
        
        Args:
            version: SDTM version (e.g., "1-8" for v1.8)
        """
        data = self._get_cached_or_fetch(f"/mdr/sdtm/{version}/datasets")
        return data.get("_links", {}).get("datasets", [])
    
    def get_sdtmig_datasets(self, version: str = "3-4") -> List[Dict]:
        """Get SDTMIG (Implementation Guide) datasets which contain all domains
        
        Args:
            version: SDTMIG version (e.g., "3-4" for v3.4)
        """
        data = self._get_cached_or_fetch(f"/mdr/sdtmig/{version}/datasets")
        return data.get("_links", {}).get("datasets", [])
    
    def get_adam_datasets(self, version: str = "2-0") -> List[Dict]:
        """Get ADaM datasets for a specific version
        
        Args:
            version: ADaM version (e.g., "2-0" for v2.0)
        """
        data = self._get_cached_or_fetch(f"/mdr/adam/{version}/datasets")
        return data.get("_links", {}).get("datasets", [])
    
    def get_dataset_variables(self, dataset_href: str) -> List[Dict]:
        """Get variables for a specific dataset
        
        Args:
            dataset_href: The href link to the dataset endpoint
        """
        # Extract the endpoint from the full URL if needed
        if dataset_href.startswith("http"):
            endpoint = dataset_href.replace(self.BASE_URL, "")
        else:
            endpoint = dataset_href
            
        data = self._get_cached_or_fetch(endpoint)
        return data.get("datasetVariables", [])
    
    def build_variable_domain_map(self, standard: str = "sdtm", version: str = "1-8", use_ig: bool = True) -> Dict[str, Set[str]]:
        """Build a complete variable to domain mapping
        
        Args:
            standard: Either "sdtm", "sdtmig", or "adam"
            version: Version of the standard (e.g., "1-8", "3-4", "2-0")
            use_ig: For SDTM, use Implementation Guide (SDTMIG) for complete domains
            
        Returns:
            Dictionary mapping variable names to set of domain codes
        """
        var_map = {}
        
        # Get all datasets
        if standard.lower() == "sdtm":
            if use_ig:
                # Use SDTMIG for complete domain coverage
                datasets = self.get_sdtmig_datasets("3-4")  # Latest SDTMIG version
                print(f"Using SDTMIG v3.4 for complete domain coverage")
            else:
                datasets = self.get_sdtm_datasets(version)
        elif standard.lower() == "sdtmig":
            datasets = self.get_sdtmig_datasets(version)
        elif standard.lower() == "adam":
            datasets = self.get_adam_datasets(version)
        else:
            raise ValueError(f"Unknown standard: {standard}")
        
        print(f"Found {len(datasets)} datasets for {standard.upper() if not use_ig else 'SDTMIG'} v{version if not use_ig else '3.4'}")
        
        # For each dataset, get variables
        for dataset in datasets:
            dataset_title = dataset.get("title", "Unknown")
            dataset_href = dataset.get("href", "")
            
            # Extract dataset code from title or href
            # For SDTM, it's usually the last part of the href like "/datasets/DM"
            if dataset_href:
                dataset_code = dataset_href.split("/")[-1]
            else:
                continue
            
            if not dataset_href:
                continue
                
            print(f"Processing dataset: {dataset_code} - {dataset_title}")
            
            try:
                variables = self.get_dataset_variables(dataset_href)
                
                for var in variables:
                    var_name = var.get("name", "")
                    if var_name:
                        if var_name not in var_map:
                            var_map[var_name] = set()
                        var_map[var_name].add(dataset_code)
                        
            except Exception as e:
                print(f"  Error processing dataset {dataset_code}: {e}")
        
        return var_map
    
    def save_mapping(self, var_map: Dict[str, Set[str]], filename: str = "variable_domain_map.json"):
        """Save the variable-domain mapping to a JSON file
        
        Args:
            var_map: Variable to domain mapping
            filename: Output filename
        """
        # Convert sets to lists for JSON serialization
        serializable_map = {k: list(v) for k, v in var_map.items()}
        
        with open(filename, 'w') as f:
            json.dump(serializable_map, f, indent=2, sort_keys=True)
        
        print(f"Saved mapping to {filename} ({len(var_map)} variables)")
    
    def load_mapping(self, filename: str = "variable_domain_map.json") -> Dict[str, Set[str]]:
        """Load a variable-domain mapping from a JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Variable to domain mapping
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to sets
        var_map = {k: set(v) for k, v in data.items()}
        
        print(f"Loaded mapping from {filename} ({len(var_map)} variables)")
        return var_map


if __name__ == "__main__":
    # Example usage
    print("CDISC API Client - Building Variable-Domain Mapping")
    print("=" * 50)
    
    try:
        client = CDISCAPIClient()
        
        # Build SDTM mapping using SDTMIG for complete domains
        print("\nBuilding complete SDTM mapping using SDTMIG...")
        sdtm_map = client.build_variable_domain_map("sdtm", use_ig=True)
        client.save_mapping(sdtm_map, "sdtm_variable_map.json")
        
        # Build ADaM mapping - check if adam/2-0 exists
        print("\nBuilding ADaM mapping...")
        try:
            adam_map = client.build_variable_domain_map("adam", "1-3")  # Try v1.3 which is more common
            client.save_mapping(adam_map, "adam_variable_map.json")
        except Exception as e:
            print(f"Note: ADaM mapping failed ({e}). This is normal if the version doesn't exist.")
            adam_map = {}
        
        # Example lookups
        print("\n" + "=" * 50)
        print("Example lookups:")
        
        test_vars = ["AESEV", "USUBJID", "AVAL", "AETERM", "STUDYID", "AGE", "SEX"]
        for var in test_vars:
            if var in sdtm_map:
                domains = sdtm_map[var]
                if len(domains) > 5:
                    print(f"  {var}: {', '.join(sorted(list(domains)[:5]))}... (multi-home, {len(domains)} total)")
                else:
                    print(f"  {var}: {', '.join(sorted(domains))}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the CDISC_API_KEY environment variable")