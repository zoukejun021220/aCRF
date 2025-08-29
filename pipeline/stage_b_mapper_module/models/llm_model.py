"""Language model handling for SDTM mapping"""

import torch
import logging
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LLMModel:
    """Handles language model initialization and querying"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct", 
                 device: str = "auto", use_4bit: bool = True):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        """Initialize the language model"""
        logger.info(f"Loading LLM model: {self.model_name}")
        
        # Model configuration
        if self.use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("LLM model loaded successfully")
        
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the model with a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
        
    def query_with_messages(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """Query with chat format"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response