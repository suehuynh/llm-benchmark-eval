import os
import yaml
import time
from dotenv import load_dotenv
from abc import ABC, abstractmethod

class BaseSummarizer(ABC):
    """
    Abstract Base Class for all LLM Summarizers.
    Ensures consistent metadata tracking across different model backends.
    """
    def __init__(self, config):
        self.config = config
        self.model_name = None
        self.params = self.config.get('generation_params', {})
    
    @abstractmethod
    def _run_inference(self, prompt: str) -> str:
        """
        Hidden method that specific models must implement 
        (e.g., calling OpenAI API or Hugging Face local model).
        """
        pass

    def format_prompt(self, article_text: str) -> str:
        """Plugs the article into the template defined in config.yaml."""
        template = self.config["prompts"]["default_summary"]
        return template.format(article_text=article_text)
    
    def summarize(self, article_text: str) -> dict:
        """
        Calculates metadata (latency) and executes the summary.
        """
        prompt = self.format_prompt(article_text)

        start_time = time.time()

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

################ GPTSummarizer ################
from openai import OpenAI
load_dotenv()

class GPTSummarizer(BaseSummarizer):
    """
    OpenAI-specific implementation of the Summarizer.
    Inherits prompt formatting and metadata tracking from BaseSummarizer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config["models"]["baseline"]["name"]
        # Setup the API Client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _run_inference(self, prompt: str) -> str:
        """
        The implementation of the inference logic for OpenAI.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.config["prompts"]["system_message"]},
                {"role": "user", "content": prompt}
            ],
            temperature = self.params.get('temperature'),
            max_tokens=self.params.get('max_new_tokens')
        )   
        return response.choices[0].message.content

################ LlamaSummarizer ################
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
class LlamaSummarizer(BaseSummarizer):
    """
    Local implementation of the Llama-3 model using Hugging Face.
    Handles hardware allocation (GPU/CPU) and tokenization.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config["models"]["challenger"]["name"]

        if torch.cuda.is_available():
            self.device="cuda"
        elif torch.backends.mps.is_available():
            self.device="mps"
        else:
            self.device="cpu"
        
        print(f"--- Loading {self.model_name} onto {self.device} ---")

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )

    def _run_inference(self, prompt) -> str:
        """
        Executes local inference.
        """
        messages = [
            {"role": "system", "content": self.config["prompts"]["system_message"]},
            {"role": "user", "content": prompt}
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=self.params.get('max_new_tokens'),
            temperature=self.params.get('temperature'),
            do_sample=True,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][-1]["content"]
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # # GPTSummarize
    # summarizer = GPTSummarizer(config=config)
    # result = summarizer.summarize("Brown University is located in Providence, RI.")
    # print(f"Summary: {result['generated_summary']}")
    # print(f"Latency: {result['latency_seconds']}s")
    # LlamaSummarize
    summarizer = LlamaSummarizer(config=config)
    result = summarizer.summarize("Brown University is located in Providence, RI.")
    print(f"Summary: {result['generated_summary']}")
    print(f"Latency: {result['latency_seconds']}s")