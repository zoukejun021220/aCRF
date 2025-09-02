#!/usr/bin/env python3
"""
Test inference with the fine-tuned SDTM mapper model
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ModelScope
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False


def _kb_paths():
    base = Path(__file__).parent / "kb" / "sdtmig_v3_4_complete"
    return {
        "domains_by_class": base / "domains_by_class.json",
        "pattern_definitions": base / "pattern_definitions.json",
    }


def _all_domain_descriptions() -> str:
    paths = _kb_paths()
    try:
        with open(paths["domains_by_class"]) as f:
            dbc = json.load(f)
    except Exception:
        dbc = {}
    text = "SDTM Domains (organized by observation class):\n\n"
    for cls in ["Special Purpose", "Interventions", "Events", "Findings"]:
        if cls in dbc:
            text += f"**{cls} Class**:\n"
            for d in dbc[cls]:
                code = d.get('code','')
                name = d.get('name','')
                desc = d.get('description','')
                text += f"- {code} ({name}) – {desc}\n"
            text += "\n"
    return text


def _all_patterns_description() -> str:
    paths = _kb_paths()
    try:
        with open(paths["pattern_definitions"]) as f:
            pats = json.load(f).get("annotation_patterns", {})
    except Exception:
        pats = {}
    desc = "Available Annotation Patterns:\n\n"
    for k, info in pats.items():
        desc += f"**{k}**:\n"
        if isinstance(info, dict):
            desc += f"- Description: {info.get('description','')}\n"
            desc += f"- Formula: {info.get('formula','')}\n"
            if info.get('keywords'):
                desc += f"- Keywords: {', '.join(info.get('keywords', []))}\n"
        desc += "\n"
    return desc


class SDTMInference:
    def __init__(self, base_model_path: str, adapter_path: str = None, use_4bit: bool = True, use_modelscope: bool = False):
        """Initialize inference with base model and optional LoRA adapter"""
        
        # Download from ModelScope if requested
        if use_modelscope and MODELSCOPE_AVAILABLE:
            logger.info("Downloading model from ModelScope...")
            modelscope_mapping = {
                "Qwen/Qwen2.5-14B-Instruct": "qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
            }
            ms_model_id = modelscope_mapping.get(base_model_path, base_model_path)
            base_model_path = snapshot_download(ms_model_id, cache_dir="./models")
            logger.info(f"Model downloaded to: {base_model_path}")
        
        # Quantization config
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        # Load base model
        logger.info(f"Loading base model: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter if provided
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Load tokenizer (prefer adapter_path tokenizer if provided)
        tok_source = adapter_path if adapter_path and Path(adapter_path).exists() else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def annotate_step_by_step(self, field_data: dict, kb_data: dict = None):
        """Perform step-by-step annotation like the training process"""
        results = {}
        
        # Step 1: Domain Selection
        domain = self.select_domain(field_data, kb_data)
        results["domain"] = domain
        logger.info(f"Selected domain: {domain}")
        
        if domain and domain != "[NOT SUBMITTED]":
            # Step 2: Pattern Selection
            pattern = self.select_pattern(field_data, domain, kb_data)
            results["pattern"] = pattern
            logger.info(f"Selected pattern: {pattern}")
            
            # Step 3: Variable/Value Selection based on pattern
            if pattern == "findings":
                annotation = self.build_findings_annotation(field_data, domain, kb_data)
            elif pattern == "conditional":
                annotation = self.build_conditional_annotation(field_data, domain, kb_data)
            elif pattern == "supplemental":
                annotation = self.build_supplemental_annotation(field_data, domain, kb_data)
            elif pattern == "plain":
                annotation = self.build_plain_annotation(field_data, domain, kb_data)
            else:
                annotation = "[NOT SUBMITTED]"
            
            results["annotation"] = annotation
            logger.info(f"Generated annotation: {annotation}")
        else:
            results["pattern"] = "not_submitted"
            results["annotation"] = "[NOT SUBMITTED]"
        
        return results

    def select_domain(self, field_data: dict, kb_data: dict = None):
        """Step 1: Select SDTM domain (faithful to unified mapper prompts)."""
        system_prompt = (
            "You are an SDTM domain selection expert. Select the most appropriate SDTM domain for the given CRF field.\n\n"
            "IMPORTANT: You must select from the available domains listed below. Consider:\n"
            "- The field content and medical context\n"
            "- The form name and section (especially for Inclusion/Exclusion criteria → IE domain)\n"
            "- Whether it's demographic, event, intervention, or finding data\n\n"
            "Return ONLY the domain code (e.g., DM, AE, VS, IE, etc.)."
        )
        user_prompt = (
            f"Select the SDTM domain for this CRF field:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Form: {field_data.get('form_name','')}\n"
            f"Section: {field_data.get('section','N/A')}\n"
            f"Field Type: {field_data.get('input_type','')}\n"
            f"Options: {field_data.get('options', [])}\n\n"
            f"{_all_domain_descriptions()}\n"
            f"Based on the field information above, which SDTM domain should this field map to?\n"
            f"Return only the domain code."
        )
        response = self.generate_response(system_prompt, user_prompt)
        return response.strip().upper()

    def select_pattern(self, field_data: dict, domain: str, kb_data: dict = None):
        """Step 2: Select annotation pattern (faithful prompts)."""
        system_prompt = (
            f"You are an SDTM annotation pattern expert. The domain {domain} has been selected.\n"
            f"Now select the appropriate annotation pattern based on the field characteristics.\n\n"
            f"Consider:\n"
            f"- Field type (text, radio, checkbox, date)\n"
            f"- Whether it has \"other, specify\" text\n"
            f"- Whether it's a measurement/test (for findings domains)\n"
            f"- Whether it maps to multiple domains\n"
            f"- Whether it needs supplemental qualifiers\n\n"
            f"{_all_patterns_description()}\n"
            f"Return ONLY the pattern name from the list above."
        )
        user_prompt = (
            f"Select the annotation pattern for this field in domain {domain}:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Field Type: {field_data.get('input_type','')}\n"
            f"Options: {field_data.get('options', [])}\n"
            f"Has Units: {field_data.get('has_units', False)}\n\n"
            f"Context:\n"
            f"- Contains \"other specify\"? {'yes' if ('other' in field_data.get('label','').lower() and 'specify' in field_data.get('label','').lower()) else 'no'}\n"
            f"- Is measurement/test? {'yes' if domain in ['VS','LB','EG','PE','QS'] else 'no'}\n"
            f"- Is checkbox with multiple options? {'yes' if field_data.get('input_type') == 'checkbox' else 'no'}\n\n"
            f"Which pattern should be used?"
        )
        response = self.generate_response(system_prompt, user_prompt)
        return response.strip().lower()

    def build_findings_annotation(self, field_data: dict, domain: str, kb_data: dict = None):
        """Build findings annotation using TESTCODE/HAS_UNITS prompt."""
        system_prompt = (
            f"You are selecting test codes for a findings domain {domain}.\n"
            f"Select the appropriate test code and specify if units are needed.\n\n"
            f"Return format:\nTESTCODE: <code>\nHAS_UNITS: <yes/no>"
        )
        user_prompt = (
            f"Select test code for this measurement:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Has Units: {field_data.get('has_units', False)}\n\n"
            f"Which test code should be used?"
        )
        resp = self.generate_response(system_prompt, user_prompt)
        testcd, has_units = None, field_data.get('has_units', False)
        for line in resp.splitlines():
            if line.strip().upper().startswith('TESTCODE:'):
                testcd = line.split(':',1)[1].strip().upper()
            if line.strip().upper().startswith('HAS_UNITS:'):
                has_units = 'YES' in line.upper()
        if not testcd:
            testcd = 'TEST'
        return (
            f"{domain}ORRES / {domain}ORRESU when {domain}TESTCD = {testcd}"
            if has_units else f"{domain}ORRES when {domain}TESTCD = {testcd}"
        )

    def build_conditional_annotation(self, field_data: dict, domain: str, kb_data: dict = None):
        """Build conditional pattern annotation (mainly for IE)"""
        
        if domain == "IE":
            system_prompt = """You are annotating inclusion/exclusion criteria for the IE domain.

For IE domain:
- Each criterion gets a unique IETESTCD (e.g., IN01, IN02 for inclusions, EX01, EX02 for exclusions)
- The criterion code usually appears in the field label in parentheses
- IETEST contains the criterion description
- IEORRES contains the response (Y/N)"""
            
            user_prompt = f"""Extract the criterion code for:
Field Label: {field_data.get('label', '')}

Return the IETESTCD value (e.g., IN01, EX01)."""
            
            testcd = self.generate_response(system_prompt, user_prompt).strip().upper()
            
            # For yes/no options
            if field_data.get('options') == ["Yes", "No"]:
                return f"Question: IETEST when IETESTCD = {testcd} | Option 'Yes': [NOT SUBMITTED] | Option 'No': IEORRES = N"
            else:
                return f"IEORRES when IETESTCD = {testcd}"
        
        return "[NOT SUBMITTED]"

    def build_supplemental_annotation(self, field_data: dict, domain: str, kb_data: dict = None):
        """Build supplemental (QNAM) annotation using unified mapper prompt."""
        system_prompt = (
            f"You are creating a supplemental qualifier (QNAM) for domain {domain}.\n"
            f"Create an 8-character uppercase identifier based on the field label.\n\n"
            f"Rules:\n"
            f"- Maximum 8 characters\n"
            f"- All uppercase\n"
            f"- No spaces or special characters\n"
            f"- Should be descriptive of the field content\n\n"
            f"Return format:\nQNAM: <identifier>"
        )
        user_prompt = (
            f"Create QNAM for this supplemental field:\n\n"
            f"Question Text: {field_data.get('label','')}\n"
            f"Domain: {domain}\n\n"
            f"What should the QNAM be?"
        )
        resp = self.generate_response(system_prompt, user_prompt)
        qnam = resp.strip()
        if qnam.upper().startswith('QNAM:'):
            qnam = qnam.split(':',1)[1].strip().upper()
        qnam = qnam[:8]
        return f"{qnam} in SUPP{domain}"

    def build_plain_annotation(self, field_data: dict, domain: str, kb_data: dict = None):
        """Build plain pattern annotation"""
        
        system_prompt = f"""You are selecting SDTM variables for direct mapping in domain {domain}.

Common variables include:
- {domain}TERM: Reported term
- {domain}OCCUR: Did event occur (Y/N)
- {domain}DAT: Date variables
- {domain}STAT: Completion status
- {domain}REASND: Reason not done

Select the appropriate variable(s) for this field."""
        
        user_prompt = f"""Select variables for:
Domain: {domain}
Field Label: {field_data.get('label', '')}
Pattern: direct mapping

Return the variable name(s) separated by '/' if multiple."""
        
        response = self.generate_response(system_prompt, user_prompt).strip().upper()
        return response

    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 50):
        """Generate response using the model"""
        
        # Format as conversation
        if self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"### Instruction:\n{system_prompt}\n\n### Input:\n{user_prompt}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response


def test_model():
    """Test the model with sample fields"""
    
    # Initialize model
    base_model = "Qwen/Qwen2.5-14B-Instruct"
    adapter_path = "./output/checkpoint-final"  # Path to fine-tuned adapter
    
    inference = SDTMInference(base_model, adapter_path)
    
    # Test cases
    test_fields = [
        {
            "label": "Date of baseline visit (VSTDT)",
            "form_name": "Baseline",
            "section": None,
            "options": [],
            "has_units": False
        },
        {
            "label": "Age ≥ 18 years (INC1)",
            "form_name": "Baseline",
            "section": "Inclusion Criteria",
            "options": ["Yes", "No"],
            "has_units": False
        },
        {
            "label": "Systolic Blood Pressure",
            "form_name": "Vital Signs",
            "section": "Measurements",
            "options": [],
            "has_units": True
        },
        {
            "label": "Did any adverse events occur?",
            "form_name": "Adverse Events",
            "section": None,
            "options": ["Yes", "No"],
            "has_units": False
        }
    ]
    
    # Test each field
    for field in test_fields:
        print("\n" + "="*60)
        print(f"Testing field: {field['label']}")
        print("="*60)
        
        result = inference.annotate_step_by_step(field)
        
        print(f"Domain: {result.get('domain')}")
        print(f"Pattern: {result.get('pattern')}")
        print(f"Annotation: {result.get('annotation')}")


if __name__ == "__main__":
    test_model()
