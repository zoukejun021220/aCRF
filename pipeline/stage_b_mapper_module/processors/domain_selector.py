"""SDTM Domain selection processor"""

import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class DomainSelector:
    """Handles SDTM domain selection based on KB definitions"""
    
    def __init__(self, domains_by_class: Dict, proto_define: Dict, llm_model):
        self.domains_by_class = domains_by_class
        self.proto_define = proto_define
        self.llm_model = llm_model
        
    def select_domain(self, field_data: Dict[str, Any], sdtm_class: str = None) -> Tuple[str, float]:
        """Select domain based on class or directly"""
        if sdtm_class and sdtm_class in self.domains_by_class:
            return self._select_domain_from_class(field_data, sdtm_class)
        else:
            return self._select_domain_direct(field_data)
            
    def _select_domain_from_class(self, field_data: Dict[str, Any], sdtm_class: str) -> Tuple[str, float]:
        """Select domain within a specific class"""
        domains_desc = self._get_domain_descriptions_for_class(sdtm_class)
        
        if not domains_desc:
            return "UNKNOWN", 0.0
            
        system_prompt = f"""You are an SDTM domain selector for the {sdtm_class} observation class.

{domains_desc}

Rules:
- Select the MOST appropriate domain based on the field content
- Consider the specific type of data being collected
- Match based on the domain's purpose and typical content"""

        user_prompt = f"""Field to map:
Question: {field_data.get('label', '')}
Type: {field_data.get('type', '')}
Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}

Select the most appropriate {sdtm_class} domain. Reply with ONLY the 2-letter domain code."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm_model.query_with_messages(messages, max_tokens=50)
        domain = response.strip().upper()[:2]
        
        # Validate domain
        raw_domains = self.domains_by_class.get(sdtm_class, [])
        valid_domains = [d["code"] if isinstance(d, dict) else d for d in raw_domains]
        if domain in valid_domains:
            return domain, 0.8
            
        # Try to find partial match
        for valid_domain in valid_domains:
            if domain in valid_domain or valid_domain in domain:
                return valid_domain, 0.6
                
        return "UNKNOWN", 0.0
        
    def _select_domain_direct(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select domain without class context"""
        # Get all domains
        all_domains = set()
        for domains in self.domains_by_class.values():
            for d in domains:
                all_domains.add(d["code"] if isinstance(d, dict) else d)
            
        # Build domain descriptions
        domains_desc = self._get_all_domain_descriptions()
        
        system_prompt = f"""You are an SDTM domain selector.

{domains_desc}

Select the MOST appropriate domain based on the field content."""

        user_prompt = f"""Field to map:
Question: {field_data.get('label', '')}
Type: {field_data.get('type', '')}
Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}

Select the most appropriate SDTM domain. Reply with ONLY the 2-letter domain code."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm_model.query_with_messages(messages, max_tokens=50)
        domain = response.strip().upper()[:2]
        
        if domain in all_domains:
            return domain, 0.7
            
        return "UNKNOWN", 0.0
        
    def _get_domain_descriptions_for_class(self, sdtm_class: str) -> str:
        """Get formatted domain descriptions for a class"""
        if sdtm_class not in self.domains_by_class:
            return ""
            
        domains = self.domains_by_class[sdtm_class]
        lines = [f"Available {sdtm_class} domains:"]
        
        # Normalize to codes
        codes = [d["code"] if isinstance(d, dict) else d for d in domains]
        for domain in sorted(codes):
            if domain in self.proto_define.get('datasets', {}):
                domain_info = self.proto_define['datasets'][domain]
                desc = domain_info.get('description', '')
                lines.append(f"\n{domain}: {desc}")
                
        return "\n".join(lines)
        
    def _get_all_domain_descriptions(self) -> str:
        """Get all domain descriptions grouped by class"""
        lines = ["SDTM Domains by Class:"]
        
        for sdtm_class, domains in self.domains_by_class.items():
            lines.append(f"\n{sdtm_class}:")
            codes = [d["code"] if isinstance(d, dict) else d for d in domains]
            for domain in sorted(codes):
                if domain in self.proto_define.get('datasets', {}):
                    domain_info = self.proto_define['datasets'][domain]
                    desc = domain_info.get('description', '')[:100]
                    lines.append(f"  {domain}: {desc}")
                    
        return "\n".join(lines)
        
    def suggest_domains(self, field_data: Dict[str, Any]) -> List[str]:
        """Suggest possible domains for a field"""
        label_lower = field_data.get('label', '').lower()
        suggestions = []
        
        # Check each domain's typical content
        for domain_code, domain_info in self.proto_define.get('datasets', {}).items():
            domain_desc = domain_info.get('description', '').lower()
            domain_name = domain_info.get('name', '').lower()
            
            # Score based on matches
            score = 0
            if any(word in label_lower for word in domain_name.split()):
                score += 2
            if any(word in label_lower for word in domain_desc.split()[:10]):
                score += 1
                
            if score > 0:
                suggestions.append((domain_code, score))
                
        # Sort by score and return top domains
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in suggestions[:5]]
