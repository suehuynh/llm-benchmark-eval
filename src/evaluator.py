import pandas as pd
import torch
import evaluate
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class SummarizationEvaluation:
    """
    Evaluates LLM outputs using structural (ROUGE) and 
    semantic (BERTScore) metrics.
    """
    def __init__(self, config):
        self.config = config
        self.metrics_cfg = config.get("evaluation", {})

        print("--- Initializing Evaluation Engines ---")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_metrics(self, predictions, references):
        """
        Calculates scores for a batch of summaries.
        """
        # 1. Calculate ROUGE (N-gram overlap)
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        # 2. Calculate BERTScore (Uses AI to understand meaning)
        # We use 'roberta-large' as the judge model for high precision
        bs_results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            device=self.device,
            model_type=self.config["evaluation"]["bertscore_model"]
        )

        return {
            "rougeL": round(rouge_results['rougeL'], 4),
            "bert_f1_avg": round(sum(bs_results['f1']) / len(bs_results['f1']), 4),
        }
    
    def evaluate_csv(self, file_path):
        """
        Loads a results CSV and calculates total performance.
        """
        df = pd.read_csv(file_path)
        
        # Filter out failed generations
        valid_df = df[df["status"] == "success"]

        print(f"Evaluating {len(valid_df)} successful samples from {file_path}...")

        scores = self.compute_metrics(
            predictions=valid_df["generated_summary"].tolist(),
            references=valid_df["reference_summary"].tolist(),
        )
        scores["avg_latency_seconds"] = round(valid_df["latency_seconds"].mean(), 2)
        return scores
    
if __name__ == "__main__":
    config = {"evaluation": {"bertscore_model": "distilbert-base-uncased"}}
    evaluator = SummarizationEvaluation(config)
    
    p = ["The cat is on the mat."]
    r = ["A cat is sitting on a rug."]
    
    print(evaluator.compute_metrics(p, r))