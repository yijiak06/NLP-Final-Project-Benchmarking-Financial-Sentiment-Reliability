# NLP-Final-Project-Benchmarking-Financial-Sentiment-Reliability
## Project Overview
This project focuses on evaluating the effectiveness of NLP models in the financial domain. We specifically compare **FinBERT** (a domain-specific Transformer model) and **VADER** (a rule-based lexicon tool) to identify where they fail to capture the complex logic of stock market sentiment, particularly within the **TSLA (Tesla)** ecosystem and **Reddit WallStreetBets (WSB)**.

## Team Members
- **Yijia Kang**
- **Manasvi Vardham**
- **Akansha Rawat**


## Repository Structure
- `analysis_code/`: Contains Python scripts for data processing and model inference.
- `dataset/`: Contains the raw CSV data (WSB and Stock Tweets).
- `files_generated/`: Contains the output of our analysis, focusing on **Logic Failures** (where models predicted sentiment incorrectly based on financial context).
- `REPRODUCE.md`: Step-by-step instructions to run the code.

## Key Models
1. **FinBERT**: A pre-trained NLP model fine-tuned on financial corpora (Financial PhraseBank).
2. **VADER**: A sentiment analysis tool specifically tuned to social media microblogging.

## Results & Analysis

### 1. The "Logic Failure" Concept
We defined a "Logic Failure" as a scenario where a model fails to understand financial terminology. For example:
- **"Shorting the stock"**: Often misinterpreted as a negative physical description rather than a bearish financial position.
- **"Buying the dip"**: Misinterpreted as negative due to the word "dip," despite being a bullish action.

### 2. Comparative Findings
- **FinBERT** significantly outperformed VADER in professional contexts but still struggled with the high-noise, sarcastic nature of Reddit's WallStreetBets.
- **TSLA Specifics**: Given Tesla's high volatility, we found that models often failed to capture "diamond hands" (holding through volatility) or "squeezing shorts," leading to the failures logged in `tsla_true_logic_failures.csv`.

### 3. Conclusion
Our analysis suggests that while domain-specific models like FinBERT are superior to general-purpose tools, a robust financial sentiment engine requires a "logic layer" to account for retail investor slang and complex trading strategies.

## Setup & Reproduction
Please refer to [REPRODUCE.md](REPRODUCE.md) for environment setup and execution instructions.
