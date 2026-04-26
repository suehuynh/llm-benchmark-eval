# Evaluation of LLM Summarization Performance: A Comparative Benchmark

**Objective:** This project evaluates the performance of open-source vs. proprietary Large Language Models (LLMs) on Abstractive Summarization. Using a sociotechnical lens, I analyze not just linguistic accuracy (ROUGE), but semantic alignment and potential failure modes (hallucinations/bias).

## 1. Experimental Design
- Dataset: cnn_dailymail (v3.0.0) – A standard benchmark for news summarization.
- Models Compared: * Challenger: Llama-3.2-1B (Open-source, local inference).
- Baseline: GPT-4o-mini (Proprietary, API-based).
- Prompt Strategy: Zero-shot, instruction-tuned prompt focused on "Who, What, Where" extraction to minimize verbosity bias.

## 2. Methodology & Metrics
To ensure a holistic evaluation, I utilized two distinct categories of metrics:
- N-gram Overlap (ROUGE-L): Measures structural similarity to human-written "Gold Standard" highlights.
- Semantic Similarity (BERTScore): Leverages contextual embeddings to evaluate if the meaning is preserved, even when exact wording differs.
- Qualitative "Human-in-the-loop" Audit: A manual review of 10% of outputs to identify hallucinations or "lost in the middle" phenomena.

## 3. Key Findings:
## 4. Analysis: 
## 5. Discussion:
## 6. Future Work:

---
How to Reproduce
1. Clone the repo.
2. Install dependencies: pip install -r requirements.txt.
3. Run the evaluation: python main.py.
