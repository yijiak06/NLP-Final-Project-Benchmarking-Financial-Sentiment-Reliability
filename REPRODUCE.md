# Reproducibility Documentation

This document provides step-by-step instructions to reproduce the analysis for the Financial Sentiment & Logic Failure project.

## 1. Environment Setup
The project requires Python 3.9+. To install all necessary libraries for both traditional scripts and LLM benchmarking, run:
```bash
pip install -r requirements.txt
```

*Note: Key dependencies include `transformers`, `nltk`, `pandas`, `google-generativeai`, and `groq`.*

## 2. Dataset Preparation

Ensure the following datasets are placed in the `dataset/` directory:

* `WSB_full.csv`: Primary dataset for **Baseline comparisons** and **LLM Benchmarking**.
* `stock_tweets.csv`: Primary dataset for **TSLA-specific analysis** and secondary benchmarks.

## 3. Execution Pipeline

### Stage 1: Traditional Models & Logic Failure Analysis

Run these scripts to generate sentiment reports and identify logic failures for VADER and FinBERT:

1. **Tesla (TSLA) Specific Analysis**:
```bash
python analysis_code/data_tsla.py

```


**Output**: `files_generated/tsla_true_logic_failures.csv`
2. **Secondary Baseline Benchmark**:
```bash
python analysis_code/data_secondary_benchmark.py

```


**Outputs**: `files_generated/finbert_logic_failures.csv`, `vader_logic_failures.csv`

### Stage 2: LLM Benchmarking (WSB Dataset Only)

This stage evaluates Large Language Models (Gemini, Llama, Mixtral) specifically on the Reddit WSB dataset.

1. Open `analysis_code/LLM_Benchmark.ipynb` in **Google Colab**.
2. Upload `dataset/WSB_full.csv` to the Colab runtime.
3. Set your API Keys for **Google (Gemini)** and **Groq (Llama)** in the designated environment cell.
4. Run all cells to perform Zero-shot, Few-shot, and Chain-of-Thought (CoT) prompting tests.

## 4. Expected Outputs

* **CSV Data**: `benchmark_summary.csv` and `benchmark_progress.csv` summarizing accuracy and F1 scores.

---

*Last Updated: March 2026*
*Course: BANA 275 - Group 16*

```

**您需要我为您把所有这些更新（README, REPRODUCE, requirements.txt）打包列在一个清单里，方便您最后检查吗？**

```
