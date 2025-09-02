"""Language model handling for SDTM mapping"""

import torch
import logging
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # optional

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
        # Optional pre-downloaded model path override
        self.local_model_path = self._discover_local_model_path()

    def _discover_local_model_path(self) -> Optional[str]:
        """Try to locate a pre-downloaded model locally.

        Priority:
        - Env var QWEN_MODEL_PATH
        - Env var ACRF_LOCAL_MODEL_DIR
        - TRANSFORMERS_CACHE/HF_HOME hub snapshots
        - Common local folders under repo: models/, local_models/
        """
        import os
        from pathlib import Path

        # 1) Direct env override
        for key in ("QWEN_MODEL_PATH", "ACRF_LOCAL_MODEL_DIR"):
            p = os.getenv(key)
            if p and (Path(p) / "config.json").exists():
                logger.info(f"Using local model from {key}: {p}")
                return p

        # 2) Search HF caches for snapshot with config.json
        candidates = []
        # Respect local caches exposed by training scripts
        for cache_var in ("TRANSFORMERS_CACHE", "HF_HOME"):
            base = os.getenv(cache_var)
            if not base:
                continue
            hub = Path(base) / "hub" / f"models--{self.model_name.replace('/', '--')}"
            snaps = hub / "snapshots"
            if snaps.exists():
                for snap in snaps.iterdir():
                    cfg = snap / "config.json"
                    tok = snap / "tokenizer_config.json"
                    if cfg.exists() and tok.exists():
                        candidates.append(str(snap))

        # 3) Common local folders relative to repo
        repo_root = Path(__file__).resolve().parents[3]
        for rel in (
            f"models/{self.model_name.split('/')[-1]}",
            f"local_models/{self.model_name.split('/')[-1]}",
        ):
            p = repo_root / rel
            if (p / "config.json").exists():
                candidates.append(str(p))

        if candidates:
            # Prefer the latest snapshot by mtime
            candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
            logger.info(f"Found local model candidates, using: {candidates[0]}")
            return candidates[0]

        return None

    def _download_model(self) -> Optional[str]:
        """Download the model if not found locally.

        Respects:
        - ACRF_AUTO_DOWNLOAD (default: true)
        - USE_MODELSCOPE / ACRF_USE_MODELSCOPE (prefer ModelScope if available)
        - TRANSFORMERS_CACHE / HF_HOME for cache location
        Returns a local filesystem path to the downloaded snapshot, or None if download not possible.
        """
        import os
        from pathlib import Path

        auto = os.getenv("ACRF_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no"}
        if not auto:
            logger.info("Auto-download disabled (ACRF_AUTO_DOWNLOAD=0)")
            return None

        # If offline, do not attempt
        if os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1":
            logger.warning("Offline mode detected; cannot download model")
            return None

        # Attempt ModelScope if requested and available
        use_ms = os.getenv("USE_MODELSCOPE", "").lower() in {"1", "true", "yes"} or \
                 os.getenv("ACRF_USE_MODELSCOPE", "").lower() in {"1", "true", "yes"}
        if use_ms:
            try:
                from modelscope import snapshot_download as ms_snapshot_download  # type: ignore
                logger.info("Downloading model via ModelScope...")
                ms_id = self.model_name
                # Minimal mapping for common Qwen repo IDs
                if "/" in ms_id and ms_id.split("/")[0].lower() == "qwen":
                    ms_id = f"qwen/{ms_id.split('/')[-1]}"
                local_path = ms_snapshot_download(ms_id)
                logger.info(f"Model downloaded to: {local_path}")
                return local_path
            except Exception as e:
                logger.warning(f"ModelScope download failed: {e}. Falling back to Hugging Face.")

        # Hugging Face Hub download
        try:
            from huggingface_hub import snapshot_download  # type: ignore
            cache_dir = os.getenv("TRANSFORMERS_CACHE") or (
                Path(os.getenv("HF_HOME", "")).joinpath("hub").as_posix() if os.getenv("HF_HOME") else None
            )
            logger.info("Downloading model via Hugging Face Hub...")
            local_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=cache_dir,
                resume_download=True,
            )
            logger.info(f"Model downloaded to: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Hugging Face download failed: {e}")
            return None
        
    def initialize(self):
        """Initialize the language model"""
        logger.info(f"Loading LLM model: {self.model_name}")
        # Ensure local availability or download
        if self.local_model_path is None:
            self.local_model_path = self._download_model()

        # Model configuration
        if self.use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_source = self.local_model_path or self.model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model_source = self.local_model_path or self.model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
        # If a fine-tuned adapter/tokenizer is provided, prefer it
        import os
        adapter_dir = os.getenv("ACRF_ADAPTER_DIR") or os.getenv("ACRF_LORA_ADAPTER")
        tok_source = adapter_dir if adapter_dir and (Path(adapter_dir) / "tokenizer_config.json").exists() else model_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=True
        )
        # Load LoRA adapter if available
        if adapter_dir and PeftModel is not None and Path(adapter_dir).exists():
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_dir)
                logger.info(f"Loaded PEFT adapter from {adapter_dir}")
            except Exception as e:
                logger.warning(f"Failed to load adapter from {adapter_dir}: {e}")
        
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

    def score_candidates_with_messages(self, messages: List[Dict[str, str]], candidates: List[str]) -> List[float]:
        """Return average log-prob scores for each candidate string as the assistant continuation.

        Builds the chat prompt (with add_generation_prompt=True), then appends each candidate
        and computes the log-likelihood of the candidate tokens conditioned on the prompt.
        """
        if not candidates:
            return []
        # Build prompt prefix
        prefix = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        scores: List[float] = []
        for cand in candidates:
            full = prefix + cand
            enc = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
            pref = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048)
            if self.device != "auto":
                enc = {k: v.to(self.device) for k, v in enc.items()}
                pref = {k: v.to(self.device) for k, v in pref.items()}
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
            # Only score the candidate segment
            input_ids = enc["input_ids"]
            pref_len = pref["input_ids"].shape[1]
            cand_ids = input_ids[:, pref_len:]
            # Shift for next-token prediction
            logits_slice = logits[:, pref_len - 1 : -1, :].contiguous()
            # Gather token log-probs
            log_probs = torch.nn.functional.log_softmax(logits_slice, dim=-1)
            token_log_probs = log_probs.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)
            # Average over tokens
            avg = token_log_probs.mean().item() if cand_ids.numel() > 0 else float('-inf')
            scores.append(avg)
        return scores
