from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import time
import warnings


import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod

import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig,
                          TextStreamer,
                          pipeline)

# ──────────────────────────────────────────────────────────────────────────────
# Helper — device selection
# ──────────────────────────────────────────────────────────────────────────────

def _select_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():          # Apple Silicon
        return "mps"
    warnings.warn(
        "No GPU detected — running Mistral-Nemo on CPU will be very slow (~5 min+). "
        "Consider using a machine with a CUDA-capable GPU or enabling MPS on Apple Silicon.",
        RuntimeWarning,
        stacklevel=2,
        )
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Helper — quantization config
# ──────────────────────────────────────────────────────────────────────────────

def _build_bnb_config() -> BitsAndBytesConfig | None:
    """
    Return a 4-bit NF4 quantization config when BitsAndBytes is available
    and a CUDA GPU is present. Falls back to None (full precision) otherwise.
    """
    try:
        import bitsandbytes
    except ImportError:
        warnings.warn(
            "bitsandbytes not installed — loading model in bfloat16 (slower). "
            "Install with: pip install bitsandbytes",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    if not torch.cuda.is_available():
        # BitsAndBytes 4-bit is CUDA-only; skip silently on MPS/CPU
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",         
        bnb_4bit_use_double_quant=True,     # second quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# ──────────────────────────────────────────────────────────────────────────────
# BaseSummarizer
# ──────────────────────────────────────────────────────────────────────────────

class BaseSummarizer(ABC):
    """
    Abstract Base Class for all LLM Summarizers.
    Ensures consistent metadata tracking across different model backends.
    """
    def __init__(self, config):
        self.config = config
        self.model_name = None
        self.params = self.config.get('generation_params', {})
        self.device = _select_device()
        self.bnb_config=_build_bnb_config()

    @abstractmethod
    def _run_inference(self, prompt: str) -> str:
        """
        Hidden method that specific models must implement.
        """
        pass
    @property
    @abstractmethod
    def prompt_key(self) -> str:
        """The config key for this model's prompt template."""
        pass

    def format_prompt(self, article_text: str) -> str:
        """Plugs the article into a specific template from config.yaml."""
        template = self.config["prompts"].get(self.prompt_key)
        if not template:
            # Fallback to a basic string if the key is missing
            return f"Summarize this: {article_text}"
        return template.format(article_text=article_text)
    
    def _prepare_prompt(self, article_text: str):
        """Default: return a formatted string. Subclasses can override."""
        return self.format_prompt(article_text)
    
    def summarize(self, article_text: str) -> dict:
        """
        Calculates metadata (latency) and executes the summary.
        """
        start_time = time.time()
        prompt = self._prepare_prompt(article_text)

        try:
            generated_text = self._run_inference(prompt)
            status = "success"
        except Exception as e:
            generated_text = f"Error: {str(e)}"
            status = "failed"

        end_time = time.time()
        return {
            "model": self.model_name,
            "generated_summary": generated_text.strip(),
            "latency_seconds": round(end_time - start_time, 2),
            "status": status,
            "word_count": len(generated_text.split())
        }

# ──────────────────────────────────────────────────────────────────────────────
# LlamaSummarizer
# ──────────────────────────────────────────────────────────────────────────────

class LlamaSummarizer(BaseSummarizer):
    """
    Local implementation of the Llama-3 model using Hugging Face.
    Handles hardware allocation (GPU/CPU) and tokenization.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config["models"]["baseline"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"--- Loading {self.model_name} onto {self.device} ---")

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            dtype=torch.bfloat16,
            token=os.getenv("HF_TOKEN")
        )
    @property
    def prompt_key(self) -> str:
        return "llama_summary"
    
    def _run_inference(self, prompt) -> str:
        """
        Executes local inference.
        """
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.params.get('max_new_tokens', 128),
            do_sample=True,
            temperature=self.params.get('temperature', 0.1),
            repetition_penalty=self.params.get('repetition_penalty'),
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
            )
    
        full_text = outputs[0]["generated_text"]
    
        # 1. Split by the assistant header to isolate the real response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
            summary = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            # Fallback for other models/formats
            summary = full_text.replace(prompt, "")

        # 2. Final cleanup of any lingering end-of-text tokens
        return summary.replace("<|eot_id|>", "").strip()

################ NemoSummarizer ################
class NemoSummarizer(BaseSummarizer):
    """
    Local implementation of the Mistral Nemo model using Hugging Face.
    Handles hardware allocation (GPU/CPU) and tokenization.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config["models"]["challenger"]["name"]
        self.generation_params = self.config["generation_params"]
        self.stream = True
        print(f"--- Loading {self.model_name} onto {self.device} ---")
        
        self._token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=self._token
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        model_kwargs: dict = {
            "token": self._token,
            "low_cpu_mem_usage": True,      # stream weights from disk → lower peak RAM
            }
        if self.bnb_config:
            # 4-bit: let BitsAndBytes decide the device map
            model_kwargs["quantization_config"] = self.bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            # No quantization: place model explicitly on selected device
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = self.device
 
        
        self.system_prompt = self.config["prompts"]["system_message"]
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
    @property
    def prompt_key(self) -> str:
        return "nemo_summary"
    
# ──────────────────────────────────────────────────────────────────────
# Private helpers for Nemo
# ──────────────────────────────────────────────────────────────────────
      
    def _prepare_prompt(self, article_text: str) -> list[dict]:
        """Return the chat messages list for apply_chat_template."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Article:\n{article_text}"},
        ]

    def _tokenize(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        """Apply chat template and tokenize to tensors on the correct device."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    def _decode_new_tokens(
        self,
        output_ids: torch.Tensor,
        input_length: int,
        ) -> str:
        """Decode only the newly generated token IDs (skip the prompt)."""
        new_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    
    def _run_inference(self, prompt) -> str:
        """
        Executes local inference.
        """
        inputs = self._tokenize(prompt)
        input_length = inputs["input_ids"].shape[1]

        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.stream else None

        with torch.inference_mode():        # lighter than no_grad; disables autograd engine
            output_ids = self.model.generate(
                **inputs,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.params,
            )
        summary = self._decode_new_tokens(output_ids, input_length)
        return summary

################ Unit Testing ################
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # # LlamaSummarizer
    # summarizer = LlamaSummarizer(config=config)
    # result = summarizer.summarize("Brown University is located in Providence, RI. It was founded in 1764 as the College of Rhode Island and has since become one of the most prestigious private universities in the United States. Brown University's academic programs are highly regarded for their rigor and diversity. The university offers over 40 undergraduate majors and numerous graduate degree options across various fields such as business, engineering, humanities, social sciences, and more. The university's athletic teams, known as the Bears, compete in the Ivy League and have won several national championships in sports like basketball, soccer, and lacrosse.")
    # print(f"Summary: {result['generated_summary']}")
    # print(f"Latency: {result['latency_seconds']}s")
    # NemoSummarizer
    summarizer = NemoSummarizer(config=config)
    result = summarizer.summarize("Brown University is located in Providence, RI. It was founded in 1764 as the College of Rhode Island and has since become one of the most prestigious private universities in the United States. Brown University's academic programs are highly regarded for their rigor and diversity. The university offers over 40 undergraduate majors and numerous graduate degree options across various fields such as business, engineering, humanities, social sciences, and more. The university's athletic teams, known as the Bears, compete in the Ivy League and have won several national championships in sports like basketball, soccer, and lacrosse.")
    print(f"Summary: {result['generated_summary']}")
    print(f"Latency: {result['latency_seconds']}s")