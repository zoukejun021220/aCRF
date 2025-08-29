"""SDTM Class selection processor"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ClassSelector:
    """Handles SDTM class selection based on KB definitions"""
    
    def __init__(self, class_definitions: Dict, llm_model):
        self.class_definitions = class_definitions
        self.llm_model = llm_model
        
    def select_class(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select SDTM class based on KB definitions"""
        if not self.class_definitions:
            logger.warning("No class definitions available")
            return self._class_selection_fallback(field_data)
            
        prompt = self._build_class_selection_prompt()
        
        # Build user message
        user_msg = f"""Field to classify:
Question: {field_data.get('label', '')}
Type: {field_data.get('type', '')}
Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}

Select the MOST appropriate SDTM class for this field. Reply with ONLY the class name."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg}
        ]
        
        response = self.llm_model.query_with_messages(messages, max_tokens=50)
        
        # Parse response
        selected_class = response.strip().upper()
        
        # Validate class exists
        for class_name in self.class_definitions.keys():
            if class_name.upper() in selected_class or selected_class in class_name.upper():
                return class_name, 0.8
                
        # Fallback if no valid class found
        return self._class_selection_fallback(field_data)
        
    def _build_class_selection_prompt(self) -> str:
        """Build system prompt for class selection"""
        lines = ["You are an SDTM class selector. Select the appropriate observation class for CRF fields."]
        lines.append("\nAvailable SDTM Classes:")
        
        # Sort classes by typical order
        class_order = ["EVENTS", "INTERVENTIONS", "FINDINGS", "SPECIAL PURPOSE"]
        sorted_classes = sorted(self.class_definitions.items(), 
                               key=lambda x: class_order.index(x[0]) if x[0] in class_order else 999)
        
        for class_name, class_info in sorted_classes:
            lines.append(f"\n{class_name}:")
            if 'description' in class_info:
                lines.append(f"  Description: {class_info['description']}")
            if 'examples' in class_info:
                lines.append(f"  Examples: {', '.join(class_info['examples'][:5])}")
                
        lines.append("\nRules:")
        lines.append("- EVENTS: Medical conditions, adverse events, medical history")
        lines.append("- INTERVENTIONS: Treatments, procedures, medications")  
        lines.append("- FINDINGS: Measurements, test results, assessments")
        lines.append("- SPECIAL PURPOSE: Demographics, comments, supplemental info")
        
        return "\n".join(lines)
        
    def _class_selection_fallback(self, field_data: Dict[str, Any]) -> Tuple[str, float]:
        """Fallback heuristic class selection"""
        label_lower = field_data.get('label', '').lower()
        
        # Check for clear indicators
        if any(term in label_lower for term in ['adverse', 'event', 'diagnosis', 'medical history', 'condition']):
            return "EVENTS", 0.6
        elif any(term in label_lower for term in ['medication', 'drug', 'dose', 'treatment', 'therapy']):
            return "INTERVENTIONS", 0.6
        elif any(term in label_lower for term in ['test', 'result', 'measurement', 'lab', 'vital', 'exam']):
            return "FINDINGS", 0.6
        elif any(term in label_lower for term in ['age', 'sex', 'race', 'ethnicity', 'demographic']):
            return "SPECIAL PURPOSE", 0.6
        else:
            # Default to FINDINGS for most clinical data
            return "FINDINGS", 0.4