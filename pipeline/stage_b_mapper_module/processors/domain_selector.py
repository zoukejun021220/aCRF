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
        
        # Constrained selection: score valid domains and pick best
        raw_domains = self.domains_by_class.get(sdtm_class, [])
        valid_domains = [d["code"] if isinstance(d, dict) else d for d in raw_domains]
        scores = self.llm_model.score_candidates_with_messages(messages, valid_domains)
        if scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return valid_domains[best_idx], 0.9
        # Fallback to free-form
        response = self.llm_model.query_with_messages(messages, max_tokens=16)
        cand = response.strip().upper()[:2]
        if cand in valid_domains:
            return cand, 0.7
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
        
        # Constrained selection over all domains
        candidates = sorted(list(all_domains))
        scores = self.llm_model.score_candidates_with_messages(messages, candidates)
        if scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx], 0.85
        # Fallback
        response = self.llm_model.query_with_messages(messages, max_tokens=16)
        cand = response.strip().upper()[:2]
        if cand in all_domains:
            return cand, 0.6
        return "UNKNOWN", 0.0

    def select_domains_multi(self, field_data: Dict[str, Any], sdtm_class: str = None,
                              max_domains: int = 3, margin: float = 0.15) -> List[str]:
        """Select all applicable domains (top-N within margin of best score).

        - If a class is provided, restrict to that class.
        - Scores are computed via continuation log-probabilities.
        - Returns a list of domain codes sorted by score (best first).
        """
        if sdtm_class and sdtm_class in self.domains_by_class:
            domains_desc = self._get_domain_descriptions_for_class(sdtm_class)
            system_prompt = f"You are an SDTM domain selector for the {sdtm_class} observation class.\n\n{domains_desc}\n\nSelect all applicable domains."
            user_prompt = (
                f"Field to map:\nQuestion: {field_data.get('label','')}\n"
                f"Type: {field_data.get('type','')}\n"
                f"Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}\n\n"
                f"Reply with ONLY a domain code from the list above."
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            raw_domains = self.domains_by_class.get(sdtm_class, [])
            candidates = [d["code"] if isinstance(d, dict) else d for d in raw_domains]
        else:
            # All domains
            system_prompt = f"You are an SDTM domain selector.\n\n{self._get_all_domain_descriptions()}\n\nSelect all applicable domains."
            user_prompt = (
                f"Field to map:\nQuestion: {field_data.get('label','')}\n"
                f"Type: {field_data.get('type','')}\n"
                f"Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}\n\n"
                f"Reply with ONLY a domain code from the list above."
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            all_domains = set()
            for domains in self.domains_by_class.values():
                for d in domains:
                    all_domains.add(d["code"] if isinstance(d, dict) else d)
            candidates = sorted(list(all_domains))

        scores = self.llm_model.score_candidates_with_messages(messages, candidates)
        if not scores:
            return []
        best = max(scores)
        # Keep those within margin of best
        picked = [(candidates[i], scores[i]) for i in range(len(scores)) if (best - scores[i]) <= margin]
        picked.sort(key=lambda t: t[1], reverse=True)
        return [d for d, _ in picked[:max_domains]]
        
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
