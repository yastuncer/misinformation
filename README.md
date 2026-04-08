# Misinformation Detection
Social Media Mining Final Project

## Research Question
How do linguistic trust-cue patterns differ between COVID and climate misinformation, and are these differences statistically significant?

## Datasets
| Dataset | Domain | Size |
|---|---|---|
| ComplexDataLab | COVID-19 + Climate Change (filtered) | 45 sub-datasets |
| MuMiN | General (filtered) | 12,914 claims / 21M tweets |

> **Note:** Dataset selection pending team decision. CMU-MisCov19 was dropped — only contains tweet IDs, not tweet text. Evaluating ComplexDataLab as potential source for both COVID and climate data.

## Setup
1. Create and activate a virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

## Project Structure
```
misinformation/
├── data/
│   ├── raw/          # Original datasets (not committed to git)
│   └── processed/    # Cleaned outputs from pipeline
├── notebooks/        # Exploratory analysis
├── src/
│   ├── preprocessing/  # Data cleaning & harmonization
│   ├── features/       # Trust-cue feature extraction
│   └── analysis/       # Statistical analysis
└── tests/
```