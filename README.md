# Misinformation Detection
Social Media Mining Final Project

## Research Question
How do linguistic trust-cue patterns differ between COVID and climate misinformation, and are these differences statistically significant?

## Datasets
| Dataset | Domain | Size |
|---|---|---|
| CMU-MisCov19 | COVID-19 | 4,573 tweets |
| ClimateMiSt | Climate Change | 2,008 tweets |
| MuMiN | General (filtered) | 12,914 claims / 21M tweets |

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