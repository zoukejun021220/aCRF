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
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
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
        """Step 1: Select SDTM domain"""
        
        # Build system prompt with available domains
        system_prompt = """You are helping select the appropriate SDTM domain for CRF fields.

Available SDTM Domains:
AE - Adverse Events
CM - Concomitant/Prior Medications  
DM - Demographics
DS - Disposition
EG - ECG
IE - Inclusion/Exclusion Criteria
LB - Laboratory Test Results
MH - Medical History
PE - Physical Examination
QS - Questionnaires
RS - Disease Response
SV - Subject Visits
VS - Vital Signs

Select the most appropriate domain code."""
        
        user_prompt = f"""Select the SDTM domain for this field:
Field Label: {field_data.get('label', '')}
Form: {field_data.get('form_name', '')}
Section: {field_data.get('section', 'N/A')}

Return only the domain code."""
        
        response = self.generate_response(system_prompt, user_prompt)
        return response.strip().upper()

    def select_pattern(self, field_data: dict, domain: str, kb_data: dict = None):
        """Step 2: Select annotation pattern"""
        
        system_prompt = f"""You are selecting the annotation pattern for domain {domain}.

Available patterns:
1. plain - Direct mapping to a single variable or multiple variables
2. findings - For test results with TESTCD (used in Findings class domains like VS, LB, EG)
3. conditional - For criteria that map based on conditions (common in IE domain)
4. supplemental - For non-standard variables that go to SUPP domain
5. not_submitted - For fields not submitted to SDTM

Select the appropriate pattern based on the field characteristics."""
        
        user_prompt = f"""Select the annotation pattern for:
Domain: {domain}
Field Label: {field_data.get('label', '')}
Has Options: {bool(field_data.get('options', []))}
Has Units: {field_data.get('has_units', False)}

Return only the pattern name."""
        
        response = self.generate_response(system_prompt, user_prompt)
        return response.strip().lower()

    def build_findings_annotation(self, field_data: dict, domain: str, kb_data: dict = None):
        """Build findings pattern annotation"""
        
        system_prompt = f"""You are selecting a test code for findings pattern in domain {domain}.

For findings domains, you need to select:
1. The appropriate TESTCD value that identifies what is being measured
2. Whether units (ORRESU) are needed

Common test codes should be short, uppercase identifiers."""
        
        user_prompt = f"""Select the test code for:
Domain: {domain}
Field Label: {field_data.get('label', '')}
Has Units: {field_data.get('has_units', False)}

Return the TESTCD value."""
        
        testcd = self.generate_response(system_prompt, user_prompt).strip().upper()
        
        # Build annotation
        if field_data.get('has_units'):
            return f"{domain}ORRES / {domain}ORRESU when {domain}TESTCD = {testcd}"
        else:
            return f"{domain}ORRES when {domain}TESTCD = {testcd}"

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
        """Build supplemental pattern annotation"""
        
        system_prompt = f"""You are creating a supplemental qualifier name (QNAM) for domain {domain}.

QNAMs should be:
- 8 characters or less
- Uppercase
- Descriptive of what is being captured
- Common QNAMs include: {domain}OTH, {domain}OTHSP (for 'other specify' fields)"""
        
        user_prompt = f"""Create QNAM for:
Domain: {domain}
Field Label: {field_data.get('label', '')}

Return the QNAM (8 chars max)."""
        
        qnam = self.generate_response(system_prompt, user_prompt).strip().upper()[:8]
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