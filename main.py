import os
import yaml
import logging
import pandas as pd
from dotenv import load_dotenv

from src.loader import DataLoader
from src.generator import GPTSummarizer, LlamaSummarizer
from src.evaluator import SummarizationEvaluation

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error("config.yaml not found! Please ensure it is in the root directory.")
        return
    
    # DATA ACQUISITION
    logger.info("Loading news articles...")
    loader = DataLoader(config)
    df = loader.get_data()
    logger.info(f"Loaded {len(df)} samples for testing.")

    # INITIALIZE MODELS
    summarizers = {
        "GPT-Baseline": GPTSummarizer(config),
        "Llama-Challenger": LlamaSummarizer(config)
    }

    # SUMMARY GENERATION
    os.makedirs("data/outputs", exist_ok=True)

    for name, model in summarizers.items():
        logger.info(f"--- Starting Inference: {name} ---")
        model_results = []

        for idx, row in df.iterrows():
            logger.info(f"Processing sample {idx+1}/{len(df)}...")

            result = model.summarize(row["article"])
            result["reference_summary"] = row["highlights"]
            model_results.append(result)
        
        save_path = f"data/outputs/{name.lower().replace('-', '_')}_raw.csv"
        pd.DataFrame(model_results).to_csv(save_path, index=False)
        logger.info(f"Saved raw {name} results to {save_path}")
    
    # EVALUATION PHASE
    logger.info("--- Starting Evaluation ---")
    evaluator = SummarizationEvaluation(config)
    final_scorecard = []

    for name in summarizers.keys():
        raw_file = f"data/outputs/{name.lower().replace('-', '_')}_raw.csv"

        scores = evaluator.evaluate_csv(raw_file)
        scores["Model"] = name
        final_scorecard.append(scores)

    report_df = pd.DataFrame(final_scorecard)
    cols = ["Model"] + [c for c in report_df.columns if c != "Model"]
    report_df = report_df[cols]

    print("\n" + "="*50)
    print("         NLP RESEARCH FINAL SCORECARD")
    print("="*50)
    print(report_df.to_string(index=False))
    print("="*50)

    report_df.to_csv("data/outputs/final_comparison_report.csv", index=False)
    logger.info("Final report saved successfully.")

if __name__ == "__main__":
    main()