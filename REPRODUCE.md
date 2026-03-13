# Reproducibility Documentation

This document provides step-by-step instructions to reproduce the sentiment analysis and logic failure detection experiments for the NLP Final Project.

## 1. Prerequisites
- **Python Version**: Python 3.9 or higher is recommended.
- **Hardware**: A machine with at least 8GB RAM. GPU acceleration is recommended but not required for inference scripts.

## 2. Environment Setup
To ensure all dependencies are correctly installed, navigate to the project root directory and run:

```bash
pip install -r requirements.txt

```

*Note: This will install core libraries including `transformers`, `torch`, `nltk`, and `pandas`.*

## 3. Data Requirements

Before running the scripts, verify that the following datasets are present in the `dataset/` folder:

* `WSB_full.csv`: Raw social media discussions from Reddit.
* `stock_tweets.csv`: General stock market microblogging data.

*If these files are missing, please refer to the download instructions in the main README.md.*

## 4. Execution Pipeline

The analysis is performed in two sequential stages. Follow the order below to generate the results found in `files_generated/`.

### Stage 1: Tesla (TSLA) Sentiment Analysis

This script filters for Tesla-specific mentions and evaluates model performance against "true" labels to identify logic failures.

```bash
python analysis_code/data_tsla.py

```

**Output**: `files_generated/tsla_true_logic_failures.csv`

### Stage 2: Secondary Benchmark Evaluation

This script runs the VADER and FinBERT models across the broader datasets to establish a performance baseline and log systematic errors.

```bash
python analysis_code/data_secondary_benchmark.py

```

**Outputs**:

* `files_generated/finbert_logic_failures.csv`
* `files_generated/vader_logic_failures.csv`

## 5. Result Verification

After execution, compare the generated CSV files in the `files_generated/` directory with the documentation provided in the final report. The logic failures are categorized by sentiment mismatches where the model failed to interpret financial context (e.g., mistaking "shorting" for a physical description rather than a financial position).
