"""Variable selection processor for SDTM mapping"""

import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class VariableSelector:
    """Handles variable selection and controlled terminology"""
    
    def __init__(self, domain_variables: Dict, controlled_terms: Dict, 
                 proto_define: Dict, llm_model):
        self.domain_variables = domain_variables
        self.controlled_terms = controlled_terms
        self.proto_define = proto_define
        self.llm_model = llm_model
        
    def select_variables(self, field_data: Dict[str, Any], domain: str, 
                        pattern: str) -> Tuple[str, float]:
        """Select variables and apply controlled terminology"""
        # Get domain variables
        domain_vars = self.get_domain_variables(domain)
        
        if not domain_vars:
            logger.warning(f"No variables found for domain {domain}")
            return f"No variables available for domain {domain}", 0.0
            
        # Build variable selection prompt based on pattern
        system_prompt = self._build_variable_selection_prompt(domain, domain_vars, pattern)
        
        user_prompt = f"""Field to map:
Question: {field_data.get('label', '')}
Type: {field_data.get('type', '')}
Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}

Pattern: {pattern}

Select the appropriate SDTM variable(s) and provide the mapping annotation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm_model.query_with_messages(messages, max_tokens=256)
        
        # Post-process to apply CT if needed
        processed_response = self._apply_controlled_terminology(response, domain, field_data)
        
        return processed_response, 0.8
        
    def get_domain_variables(self, domain: str) -> List[Dict]:
        """Get all variables for a domain"""
        # First check domain_variables (from variables_all.json)
        domain_vars = self.domain_variables.get(domain, {})
        
        # Fallback to proto_define
        if not domain_vars and domain in self.proto_define.get('datasets', {}):
            proto_vars = self.proto_define['datasets'][domain].get('variables', {})
            domain_vars = proto_vars
            
        # Convert to list format
        var_list = []
        for var_name, var_info in domain_vars.items():
            var_dict = {'name': var_name}
            if isinstance(var_info, dict):
                var_dict.update(var_info)
            var_list.append(var_dict)
            
        return var_list
        
    def _build_variable_selection_prompt(self, domain: str, domain_vars: List[Dict], 
                                       pattern: str) -> str:
        """Build prompt for variable selection"""
        lines = [f"You are an SDTM variable selector for domain {domain} using pattern {pattern}."]
        
        # Add domain description
        if domain in self.proto_define.get('datasets', {}):
            domain_info = self.proto_define['datasets'][domain]
            lines.append(f"\nDomain: {domain} - {domain_info.get('description', '')}")
            
        # Add relevant variables based on pattern
        lines.append(f"\nAvailable {domain} variables:")
        
        # Filter variables by relevance to pattern
        relevant_vars = self._filter_variables_by_pattern(domain_vars, pattern, domain)
        
        for var in relevant_vars[:15]:  # Limit to prevent token overflow
            var_name = var.get('name', '')
            var_label = var.get('label', '')
            var_type = var.get('type', '')
            
            line = f"{var_name}: {var_label}"
            if var_type:
                line += f" ({var_type})"
            if self._variable_has_ct(var_name, domain):
                line += " [CT]"
                
            lines.append(f"  {line}")
            
        # Add pattern-specific instructions
        lines.extend(self._get_pattern_instructions(pattern, domain))
        
        return "\n".join(lines)
        
    def _filter_variables_by_pattern(self, domain_vars: List[Dict], pattern: str, 
                                    domain: str) -> List[Dict]:
        """Filter variables based on pattern relevance"""
        if pattern == "direct":
            # For direct mapping, include identifier and topic variables
            return [v for v in domain_vars 
                   if any(role in str(v.get('role', '')).lower() 
                         for role in ['identifier', 'topic', 'result', 'grouping'])]
                         
        elif pattern == "fixed_value":
            # For fixed value, focus on categorical variables
            return [v for v in domain_vars 
                   if self._variable_has_ct(v.get('name', ''), domain)]
                   
        elif pattern == "checkbox":
            # For checkbox, include flag variables
            return [v for v in domain_vars 
                   if 'flag' in str(v.get('type', '')).lower() 
                   or 'yn' in str(v.get('name', '')).lower()]
                   
        else:
            # Return all for other patterns
            return domain_vars
            
    def _get_pattern_instructions(self, pattern: str, domain: str) -> List[str]:
        """Get pattern-specific instructions"""
        instructions = ["\nMapping format:"]
        
        if pattern == "direct":
            instructions.append("variable_name=@VALUE")
            instructions.append("Example: LBTEST=@VALUE")
            
        elif pattern == "fixed_value":
            instructions.append('variable_name="fixed_value"')
            instructions.append('Example: LBTEST="Hemoglobin"')
            
        elif pattern == "conditional":
            instructions.append("IF condition THEN variable=value")
            instructions.append('Example: IF @VALUE="Yes" THEN LBYN="Y"')
            
        elif pattern == "checkbox":
            instructions.append("For each checked option: variable=value")
            instructions.append('Example: CMYN="Y" for checked medications')
            
        return instructions
        
    def _apply_controlled_terminology(self, response: str, domain: str, 
                                     field_data: Dict[str, Any]) -> str:
        """Apply controlled terminology to response"""
        # This would check if variables in response have CT and suggest values
        # For now, return as-is
        return response
        
    def _variable_has_ct(self, variable_name: str, domain: str) -> bool:
        """Check if variable has controlled terminology"""
        # Check in proto_define
        if domain in self.proto_define.get('datasets', {}):
            vars_info = self.proto_define['datasets'][domain].get('variables', {})
            if variable_name in vars_info:
                var_info = vars_info[variable_name]
                if isinstance(var_info, dict):
                    return 'codelist' in var_info or 'controlled_terms' in var_info
                    
        return False
        
    def get_ct_values(self, variable_name: str, domain: str) -> List[Dict]:
        """Get controlled terminology values for a variable"""
        ct_values = []
        
        # Check various sources for CT
        # This is simplified - full implementation would check all CT sources
        if variable_name in self.controlled_terms:
            ct_values = self.controlled_terms[variable_name]
            
        return ct_values