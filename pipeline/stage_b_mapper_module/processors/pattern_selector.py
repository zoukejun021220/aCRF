"""Pattern selection processor for SDTM mapping"""

import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class PatternSelector:
    """Handles pattern selection based on KB definitions"""
    
    def __init__(self, patterns: Dict, llm_model):
        # Normalize patterns structure (supports KB annotation_patterns or legacy patterns)
        self.patterns = patterns or {}
        if isinstance(self.patterns, dict) and 'patterns' in self.patterns and not self.patterns.get('annotation_patterns'):
            self.patterns = {'annotation_patterns': self.patterns.get('patterns')}
        self.llm_model = llm_model
        
    def select_pattern(self, field_data: Dict[str, Any], domain: str = None) -> Tuple[str, float]:
        """Select mapping pattern for field"""
        if not self.patterns:
            logger.warning("No pattern definitions available")
            return "direct", 0.5
            
        # Get all pattern info
        pattern_info = self.patterns.get("annotation_patterns", {})
        if not pattern_info:
            return "direct", 0.5
            
        # Build pattern selection prompt
        system_prompt = self._build_pattern_selection_prompt(pattern_info, domain)
        
        user_prompt = f"""Field to analyze:
Question: {field_data.get('label', '')}
Type: {field_data.get('type', '')}
Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}
{f"Domain: {domain}" if domain else ""}

Which mapping pattern best fits this field? Reply with ONLY the pattern name."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Constrained selection among known patterns
        candidates = list(pattern_info.keys())
        scores = self.llm_model.score_candidates_with_messages(messages, candidates)
        if scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx], 0.85
        # Fallback to heuristic if scoring failed
        return self._pattern_selection_fallback(field_data)
        
    def _build_pattern_selection_prompt(self, pattern_info: Dict, domain: str = None) -> str:
        """Build prompt for pattern selection"""
        lines = ["You are an SDTM mapping pattern selector."]
        if domain:
            lines.append(f"The selected domain is {domain}.")
            
        lines.append("\nAvailable mapping patterns:")
        
        # Use pattern order if available, else default alphabetical
        pattern_order = [
            "direct", "variable_with_ct", "conditional_population", "test_measurement",
            "other_specify", "checkbox", "multiple_variables", "supplemental",
            "cross_domain", "derived", "not_submitted", "skip"
        ]
        sorted_patterns = sorted(
            pattern_info.items(),
            key=lambda x: pattern_order.index(x[0]) if x[0] in pattern_order else 999
        )
        
        for pattern_name, info in sorted_patterns:
            lines.append(f"\n{pattern_name}:")
            if isinstance(info, dict) and 'description' in info:
                lines.append(f"  Purpose: {info['description']}")
            if isinstance(info, dict) and 'when_to_use' in info:
                lines.append(f"  When to use: {info['when_to_use']}")
            if isinstance(info, dict) and 'examples' in info and info['examples']:
                lines.append(f"  Examples: {info['examples'][0]}")
                
        lines.append("\nSelection rules:")
        lines.append("- Choose based on the field's data collection pattern")
        lines.append("- Consider if the field has fixed options, free text, or conditional logic")
        lines.append("- Think about how the data maps to SDTM structure")
        
        return "\n".join(lines)
        
    def _pattern_selection_fallback(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Fallback pattern selection using heuristics"""
        field_type = field_data.get('type', '').lower()
        options = field_data.get('options', [])
        label = field_data.get('label', '').lower()
        
        # Check for specific patterns
        if field_type == 'checkbox' or (options and all('□' in str(o) or '☐' in str(o) for o in options)):
            return "checkbox", 0.7
            
        if 'other' in label and 'specify' in label:
            return "other_specify", 0.7
            
        if options and len(options) > 1:
            # Check if it's a yes/no/unknown pattern
            opt_lower = [o.lower() for o in options]
            if any(yn in opt_lower for yn in ['yes', 'no']) or any(yn in ' '.join(opt_lower) for yn in ['yes', 'no']):
                return "conditional", 0.6
            else:
                return "fixed_value", 0.6
                
        if field_type in ['text', 'number', 'date']:
            return "direct", 0.7
            
        # Default
        return "direct", 0.5
        
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get detailed information about a pattern"""
        if not self.patterns or "patterns" not in self.patterns:
            return {}
            
        return self.patterns["patterns"].get(pattern_name, {})
