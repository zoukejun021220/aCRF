"""Variable selection processor for SDTM mapping"""

import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class VariableSelector:
    """Handles variable selection and controlled terminology"""
    
    def __init__(self, domain_variables: Dict, controlled_terms: Dict, 
                 proto_define: Dict, llm_model, cdisc_ct: Dict = None):
        self.domain_variables = domain_variables
        self.controlled_terms = controlled_terms
        self.proto_define = proto_define
        self.llm_model = llm_model
        self.cdisc_ct = cdisc_ct or {}
        
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
        user_prompt = (
            f"Field to map:\nQuestion: {field_data.get('label','')}\n"
            f"Type: {field_data.get('type','')}\n"
            f"Options: {', '.join(field_data.get('options', [])) if field_data.get('options') else 'N/A'}\n\n"
            f"Pattern: {pattern}\n\n"
            f"Select ALL applicable SDTM variable(s) and provide the mapping annotation."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # Multi-select logic
        multi = pattern in {"multiple_variables", "checkbox", "variable_with_ct"}
        if multi:
            # Score candidate variables and pick all within margin
            cand_vars = [v.get('name') for v in domain_vars if v.get('name')]
            # Use a short “return only the variable name” scoring prompt
            score_msgs = [
                {"role": "system", "content": system_prompt + "\nReturn ONLY the variable name."},
                {"role": "user", "content": f"Field: {field_data.get('label','')}"}
            ]
            scores = self.llm_model.score_candidates_with_messages(score_msgs, cand_vars)
            picked = []
            if scores:
                best = max(scores)
                for i, s in enumerate(scores):
                    if (best - s) <= 0.15:  # within margin of best
                        picked.append(cand_vars[i])
            # Build mapping: for variables with CT, map options to submissionValues; else list variables
            parts: List[str] = []
            options = field_data.get('options') or []
            for var in picked[:6]:
                if self._variable_has_ct(var, domain):
                    ct_vals = self.get_ct_values(var, domain)
                    # Map options to CT terms; if no options, fall back to top CT
                    mapped = self._match_options_to_ct(ct_vals, options)
                    if not mapped and ct_vals:
                        mapped = [ct_vals[0].get('submissionValue') or ct_vals[0].get('code') or '']
                    for mv in filter(None, mapped):
                        parts.append(f"{var} = {mv}")
                else:
                    parts.append(f"{var}")
            annotation = " / ".join(dict.fromkeys(parts)) if parts else ""
            if annotation:
                return annotation, 0.85
            # Fallback to generative path if no picks

        # Single selection or fallback: generative
        response = self.llm_model.query_with_messages(messages, max_tokens=256)
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
        elif pattern in {"variable_with_ct", "multiple_variables", "conditional_population"}:
            instructions.append("Select ALL that apply; separate multiple entries with ' / '")
            instructions.append("Examples:")
            instructions.append("- VAR1 = CT1 / VAR2 = CT2")
            instructions.append("- VAR1 when CONDVAR = CTX")
        
        return instructions

    def _match_options_to_ct(self, ct_terms: List[Dict[str, Any]], options: List[str]) -> List[str]:
        """Match UI/options to CT submission values (case-insensitive contains)."""
        if not options or not ct_terms:
            return []
        res: List[str] = []
        for opt in options:
            o = str(opt).strip().lower()
            for t in ct_terms:
                sub = (t.get('submissionValue') or t.get('code') or '').lower()
                term = (t.get('preferredTerm') or t.get('value') or '').lower()
                if o == sub or o == term or o in term:
                    sv = t.get('submissionValue') or t.get('code') or ''
                    if sv and sv not in res:
                        res.append(sv)
                    break
        return res
        
    def _apply_controlled_terminology(self, response: str, domain: str, 
                                     field_data: Dict[str, Any]) -> str:
        """Normalize VAR = VALUE to use submissionValue when CT exists."""
        try:
            parts = [p.strip() for p in response.split('/')]
            new_parts = []
            for part in parts:
                if '=' not in part:
                    new_parts.append(part)
                    continue
                var, val = part.split('=', 1)
                var = var.strip().strip('.')
                val_raw = val.strip().strip('"').strip("'")
                # Try to map via variables_all codelist → cdisc_ct terms
                ct_vals = self._ct_terms_for_variable(domain, var)
                if ct_vals:
                    low = val_raw.lower()
                    best = None
                    for t in ct_vals:
                        sub = t.get('submissionValue') or t.get('code') or ''
                        term = t.get('preferredTerm') or t.get('value') or ''
                        if low == sub.lower() or low == term.lower() or low in term.lower():
                            best = sub
                            break
                    val_out = best or val_raw
                    new_parts.append(f"{var} = {val_out}")
                else:
                    new_parts.append(part)
            return " / ".join(new_parts)
        except Exception:
            return response
        
    def _variable_has_ct(self, variable_name: str, domain: str) -> bool:
        """Check if variable has controlled terminology"""
        # Prefer variables_all information
        if domain in self.domain_variables and variable_name in self.domain_variables[domain]:
            var_info = self.domain_variables[domain][variable_name]
            return bool(var_info.get('codelist'))
        # Fallback to proto_define
        if domain in self.proto_define.get('datasets', {}):
            vars_info = self.proto_define['datasets'][domain].get('variables', {})
            if isinstance(vars_info, dict) and variable_name in vars_info:
                var_info = vars_info[variable_name]
                if isinstance(var_info, dict):
                    return 'codelist' in var_info or 'controlled_terms' in var_info
        return False
        
    def get_ct_values(self, variable_name: str, domain: str) -> List[Dict]:
        """Get controlled terminology values for a variable using cdisc_ct if available."""
        # From variables_all mapping
        var_info = self.domain_variables.get(domain, {}).get(variable_name)
        if isinstance(var_info, dict) and var_info.get('codelist'):
            cl = var_info['codelist']
            cl_code = cl.get('conceptId') or cl.get('code')
            if self.cdisc_ct and cl_code:
                for clist in self.cdisc_ct.get('codelists', []):
                    c = clist.get('codelist', {})
                    if c.get('conceptId') == cl_code or c.get('code') == cl_code:
                        return clist.get('terms', [])
        # Fallback legacy mapping
        return self.controlled_terms.get(variable_name, [])
