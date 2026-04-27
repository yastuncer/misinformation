# Misinformation Linguistic Analysis
### Comparing COVID-19 and Climate Change Misinformation Through Text Features, Emotion, and Rhetoric

---

## Overview

This project investigates whether COVID-19 and climate change misinformation are **linguistically and emotionally distinct**. Using a multi-dataset corpus of ~9,000 labeled misinformation texts, we apply TF-IDF term analysis, transformer-based emotion classification, VADER sentiment scoring, and LLM-powered rhetorical feature extraction to statistically compare the two domains.

**Core finding:** COVID misinformation is characterized by fear-driven, negative-sentiment language centered on bodily harm (sadness 2× higher), while climate misinformation is driven by institutional anger and authority-invoking rhetoric (authority language 2× higher, anger 1.4× higher).

---

## Project Structure

```
misinformation/
├── data/
│   ├── raw/                  # downloaded source datasets
│   ├── processed/            # cleaned and labeled CSVs (covid, climate, general)
│   └── analysis/             # feature outputs and statistical results
├── src/
│   ├── preprocessing/        # data ingestion, normalization, cleaning
│   │   ├── pipeline.py       # main preprocessing pipeline
│   │   ├── download_datasets.py
│   │   ├── normalize.py
│   │   ├── harmonize_labels.py
│   │   ├── filter_datasets.py
│   │   └── clean.py
│   ├── features/             # feature extraction and analysis
│   │   ├── pipeline.py       # main analysis pipeline
│   │   ├── emotion.py        # transformer emotion classifier
│   │   ├── rhetoric.py       # LLM rhetorical strategy scoring
│   │   ├── tf_idf.py         # TF-IDF vectorization and term comparison
│   │   ├── vader.py          # VADER sentiment analysis
│   │   ├── lemmatization.py  # spaCy lemmatization
│   │   └── visualize.py      # plot generation
│   └── analysis/
│       └── stats.py          # statistical tests
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Domain | Source | Rows (misinfo only) |
|---|---|---|---|
| ComplexDataLab/Misinfo_Datasets | COVID + General | HuggingFace | ~4,000 |
| covid_19_rumor | COVID | HuggingFace (CDL) | ~1,200 |
| jns-covid_misinfo | COVID | Princeton (Excel) | ~2,500 |
| climate_fever | Climate | HuggingFace (CDL) | ~251 |
| QuotaClimat/frugalaichallenge | Climate | HuggingFace | ~3,554 |
| tdiggelm/climate_fever | Climate | HuggingFace | ~2 |
| rabuahmad/climatecheck | Climate | HuggingFace | ~600 |

**Final processed sizes:** COVID ~4,200 rows · Climate ~4,600 rows

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd misinformation

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ollama for LLM rhetoric analysis (optional)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

> **Note on PyTorch:** If you need GPU support, install PyTorch manually from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

> **Note on spaCy:** If installation fails on `en_core_web_sm`, run `pip install spacy==3.8.13` first.

---

## Usage

### Step 1 — Download raw datasets

```bash
python -m src.preprocessing.download_datasets
```

### Step 2 — Run preprocessing pipeline

Normalizes columns, harmonizes labels, filters to English, deduplicates, and outputs three processed CSVs.

```bash
python -m src.preprocessing.pipeline
```

**Output:** `data/processed/covid.csv`, `data/processed/climate.csv`, `data/processed/general.csv`

### Step 3 — Run feature extraction and analysis pipeline

```bash
python -m src.features.pipeline
```

**Output (in `data/analysis/`):**
- `covid_emotions.csv` / `climate_emotions.csv` — per-row emotion scores (7 emotions)
- `covid_vader.csv` / `climate_vader.csv` — per-row VADER sentiment scores
- `rhetoric_features.csv` — per-row urgency/authority/doubt scores from LLM
- `rhetoric_stats.csv` — Mann-Whitney U and Chi-squared test results
- `top_tfidf_terms.json` — top 25 TF-IDF terms per domain
- `vader_sentiment.json` — aggregate VADER scores per domain

---

## Methods

### Preprocessing
- URL replacement (`URL` token), mention replacement (`USER` token), hashtag stripping, emoji removal, contraction expansion, unicode escape decoding
- Language filtering via `langdetect` (applied to COVID corpus only — climate datasets are English-only sources)
- Label harmonization across 7 different labeling schemas into binary `misinformation` / `credible`

### TF-IDF
- Fitted on both domains combined so IDF scores are directly comparable
- `ngram_range=(1,2)` captures both unigrams and bigrams
- Lemmatized text used as input (spaCy `en_core_web_sm`, stopwords removed)

### Emotion Analysis
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Labels: anger, disgust, fear, joy, neutral, sadness, surprise
- `top_k=None` returns all 7 scores per text
- Batched inference (batch_size=32) on CPU

### Sentiment Analysis
- VADER compound score per row (−1 to +1)
- Aggregate neg/neu/pos/compound means per domain

### Rhetorical Analysis
- LLM: `llama3.2` via local Ollama
- Scores urgency, authority, and doubt as continuous 0–1 floats per text
- Authority-to-doubt ratio derived feature

### Statistical Tests
- **Mann-Whitney U** for continuous features (non-parametric, appropriate for skewed score distributions)
- **Chi-squared** for binary presence features (2×2 contingency table: domain × feature present/absent)

---

## Key Findings

| Feature | COVID | Climate | Test | Significance |
|---|---|---|---|---|
| Sadness (emotion) | 0.127 | 0.080 | Mann-Whitney U | *** |
| Anger (emotion) | 0.100 | 0.142 | Mann-Whitney U | *** |
| Authority (rhetoric) | 0.046 | 0.112 | Mann-Whitney U | *** |
| Authority presence | 16.2% | 33.0% | Chi-squared | *** |
| Doubt presence | 89.3% | 94.6% | Chi-squared | *** |
| VADER compound | −0.092 | +0.056 | — | — |

COVID misinformation is **emotionally negative and fear-driven**, clustering around bodily harm language. Climate misinformation is **institutionally adversarial**, invoking or attacking scientific authority at twice the rate, with anger as its dominant non-neutral emotion.

---

## Dependencies

Key packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `spacy`, `transformers`, `torch`, `vaderSentiment`, `langdetect`, `datasets`, `matplotlib`, `seaborn`, `requests` (for Ollama)

See `requirements.txt` for full pinned versions.

---

## Notes and Limitations

- **Source imbalance:** 93% of climate data comes from QuotaClimat (newspaper/media quotes), while COVID data is predominantly social media. Observed differences may partly reflect medium rather than domain.
- **LLM rhetoric scoring** is non-deterministic — scores vary slightly between runs. Results are cached after first run.
- **`general.csv`** is not used in the comparative analysis — it serves as a potential baseline corpus for future work.
- Language filtering is skipped for climate datasets since all sources are English-only. It is applied only to the COVID corpus where multilingual contamination was detected (primarily `esoc` and `jns-covid_misinfo`).