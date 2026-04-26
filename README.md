# Evaluation of LLM Summarization Performance: A Comparative Benchmark

**Objective:** This project evaluates the performance of open-source vs. proprietary Large Language Models (LLMs) on Abstractive Summarization. Using a sociotechnical lens, I analyze not just linguistic accuracy (ROUGE), but semantic alignment and potential failure modes (hallucinations/bias).

## 1. Experimental Design
 
| | Baseline | Challenger |
|---|---|---|
| **Model** | meta-llama/Llama-3.2-3B-Instruct | mistralai/Mistral-Nemo-Instruct-2407 |
| **Type** | Open-source, local inference | Open-source, local inference |
| **Parameters** | 3B | 12B |
| **Quantization** | bfloat16 | 4-bit NF4 (BitsAndBytes) |
 
**Dataset:** `cnn_dailymail` v3.0.0 — a standard benchmark for news summarization, using 20 samples from the test split.
 
**Prompt strategy:** Zero-shot instruction-tuned prompts. Each model uses its native chat template via `apply_chat_template`, with a shared system instruction focused on extracting 3 concise bullet points to minimize verbosity bias.

## 2. Methodology & Metrics
To ensure a holistic evaluation, I utilized two distinct categories of metrics:
- **ROUGE-L** measures the longest common subsequence overlap between generated and reference summaries, capturing structural and lexical fidelity to the human-written gold standard.
- **BERTScore F1** leverages contextual embeddings (`roberta-large`) to evaluate semantic similarity — capturing cases where meaning is preserved even when exact wording differs.

## 3. Key Findings
 
| Model | ROUGE-L | BERTScore F1 | Avg Latency (s) |
|---|---|---|---|
| Llama-3.2-3B (Baseline) | 0.1633 | 0.8635 | 124.71 |
| Mistral-Nemo (Challenger) | 0.2288 | 0.8762 | 2871.84 |
 
- Mistral-Nemo outperforms Llama-3.2-3B on both quality metrics, with a **+40% relative improvement in ROUGE-L** and a modest gain in BERTScore F1.
- Llama-3.2-3B is **23× faster** on average, making it significantly more practical for latency-sensitive applications.
> **Note:** Nemo's latency was measured under CPU inference without GPU acceleration. With a CUDA-capable GPU and 4-bit NF4 quantization, this gap would narrow substantially.
 
## 4. Analysis: 
The ROUGE-L gap (0.1633 vs 0.2288) reflects a meaningful difference in lexical alignment with reference summaries. Nemo's larger parameter count (12B vs 3B) likely accounts for its stronger capacity to identify and reproduce salient phrases from source articles.
 
The BERTScore gap is smaller (0.8635 vs 0.8762), suggesting both models preserve semantic meaning at a similar level despite the lexical difference — Llama frequently paraphrases where Nemo reproduces closer to the original phrasing.
 
The latency gap is the most operationally significant finding. Under CPU-only conditions, Nemo's 12B parameter footprint makes it impractical for real-time use. The 4-bit NF4 quantization used here already reduces its memory footprint from ~24 GB (bfloat16) to ~7 GB; GPU inference would further reduce latency to an estimated 15–30 seconds per sample.

## 5. Discussion:
This benchmark surfaces a core tradeoff in applied NLP: quality vs. deployability. Mistral-Nemo produces higher-quality summaries by standard metrics, but its inference cost under CPU-only constraints makes Llama-3.2-3B the pragmatically superior choice for resource-constrained environments.
Evaluation was conducted using ROUGE-L and BERTScore F1 — metrics that measure similarity to a reference summary. While these are standard in the summarization literature, neither directly penalises hallucination. A model can score well on both while still fabricating facts absent from the source article. This is a known limitation of reference-based evaluation and motivates the faithfulness and QAG analysis planned for future work.
Both models were evaluated zero-shot. Fine-tuning on domain-specific summarization data (e.g. CNN/DM training split) would likely close the ROUGE gap and reduce hallucination rates for both.
 
## 6. Future Work:
- **Human-in-the-loop audit:** A manual review of 10% of outputs to identify hallucinations or "lost in the middle" phenomena not captured by automated metrics.
- **Faithfulness & QAG evaluation:** Complement reference-based metrics with SummaC faithfulness scoring and question-answering generation (QAG) to directly penalise hallucinations not captured by ROUGE or BERTScore.
- **Scale the dataset:** Extend from 20 to 200+ samples to reduce variance and improve statistical reliability of metric comparisons.
- **Fine-tuning:** Evaluate the effect of instruction fine-tuning on CNN/DM training data for both models.
- **Bias analysis:** Apply the `critical_audit` prompt from config to flag emotionally loaded language and source bias in generated summaries.
---
How to Reproduce
1. Clone the repo.
2. Install dependencies: pip install -r requirements.txt.
3. Run the evaluation: python main.py.
