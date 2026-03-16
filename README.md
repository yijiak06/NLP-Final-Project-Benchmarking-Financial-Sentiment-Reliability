# Financial Sentiment Analysis & Logic Failure Detection: From Baselines to LLMs

## Project Overview
This project investigates the effectiveness of various NLP models in the financial domain. We evaluate models ranging from traditional lexicon-based tools to state-of-the-art Large Language Models (LLMs) to identify "Logic Failures"—scenarios where models fail to grasp complex financial context or retail investor slang.

## Team Members (12)
- **Yijia Kang**
- **Manasvi Vardham**
- **Akansha Rawat**

## Repository Structure
- `analysis_code/`:
    - `data_tsla.py`: Sentiment analysis focused on TSLA (Tesla) stock tweets.
    - `data_secondary_benchmark.py`: Baseline evaluation using VADER and FinBERT.
    - `LLM_Benchmark.ipynb`: Comprehensive benchmarking of Gemini, Llama, and Mixtral.
- `dataset/`:
    - `WSB_full.csv`: Reddit WallStreetBets discussion data (Primary dataset for LLM Benchmarking).
    - `stock_tweets.csv`: General financial twitter data.
- `files_generated/`: Contains output CSVs of logic failures and performance visualizations.
- `REPRODUCE.md`: Step-by-step instructions for environment setup and execution.

## Methodology & Model Scope

### 1. Traditional Baselines (VADER & FinBERT)
We utilized VADER (rule-based) and FinBERT (domain-specific Transformer) across both datasets to establish a baseline. We focused on identifying **Logic Failures**, such as the misinterpretation of "Shorting" or "Buying the dip."

### 2. LLM Benchmarking (WSB Dataset Exclusive)
To push the boundaries of our analysis, we benchmarked advanced LLMs specifically on the **WSB_full.csv** dataset to test their ability to decode high-noise, sarcastic retail investor sentiment.
- **Models**: Gemini 1.5 Pro, Llama 3.1 70B, and Mixtral 8x7B.
- **Prompting Strategies**: Zero-shot, Few-shot, and Chain-of-Thought (CoT).

## Key Results & Findings

- **Baseline Comparison**: On the WSB dataset, the VADER baseline achieved ~44.18% accuracy, while FinBERT achieved ~32.95%, highlighting the difficulty of social media financial text.
- **LLM Superiority**: LLMs showed significantly better performance in handling sarcasm and slang (e.g., "diamond hands").
- **The "Logic Gap"**: Despite LLM improvements, logic failures persist when a financial action contradicts standard linguistic sentiment. For instance, "TSLA to the moon" requires understanding ticker-specific hype that general models may still miss.

## Performance Visualization
The project generates a **Macro F1 Heatmap** (saved in `files_generated/`) comparing different model-prompt combinations, providing a clear visual representation of which strategy works best for financial social media.

## How to Run
Please refer to the [REPRODUCE.md](REPRODUCE.md) file for detailed instructions on setting up the Python environment and running the analysis pipeline.

## Project Structure

```text
├── analysis_code/
│   ├── data_tsla.py                 # Sentiment analysis for Tesla stock tweets
│   ├── data_secondary_benchmark.py  # Baseline evaluation for VADER and FinBERT
│   └── LLM_Benchmark.ipynb          # Benchmarking for Gemini, Llama, and Mixtral
├── dataset/
│   └── WSB_full.csv                 # Primary dataset for LLM and Baseline analysis
├── files_generated/
│   ├── vader_logic_failures.csv     # VADER model logic mismatch results
│   ├── finbert_logic_failures.csv   # FinBERT model logic mismatch results
│   ├── tsla_true_logic_failures.csv # Tesla-specific sentiment logic failures
│   ├── gpt result.csv               # LLM output results
│   └── Gemini_v1_wsb_predictions.csv # Gemini-specific predictions on WSB dataset
├── README.md                        # Project overview and findings
├── REPRODUCE.md                     # Step-by-step reproduction guide
└── requirements.txt                 # Environment dependencies

```
