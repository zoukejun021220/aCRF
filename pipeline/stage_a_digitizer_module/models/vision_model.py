"""Vision model handling for Stage A"""

import torch
import logging
from typing import Optional
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

logger = logging.getLogger(__name__)


class VisionModel:
    """Handles vision model initialization and inference"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "auto", use_4bit: bool = True):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit
        self.model = None
        self.processor = None
        self._process_vision_info = process_vision_info
    
    def initialize(self):
        """Initialize Qwen-VL model with proper configuration"""
        logger.info(f"Loading Qwen2.5-VL-7B model...")
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        # Use device_map for CUDA
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        if Qwen2_5_VLForConditionalGeneration is not None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            logger.info("Loaded model with Qwen2_5_VLForConditionalGeneration")
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            logger.info("Loaded model with AutoModelForVision2Seq")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Don't try to move the model if device_map is used
        if "device_map" not in model_kwargs and self.device != "auto":
            self.model = self.model.to(self.device)
            
        logger.info("Model loaded successfully")
    
    def query_model(self, page_image, prompt: str) -> str:
        """Query the model with an image and prompt"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a precise data extractor. Output only the requested format."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Generate with controlled temperature
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text],
                images=[page_image],
                return_tensors="pt"
            )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.device == "cuda":
            torch.cuda.empty_cache()