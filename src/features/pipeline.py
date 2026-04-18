import sys
import os
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.features.tf_idf import top_terms_per_domain, print_top_terms
from src.features.lemmatization import lemmatize_series

PROCESSED_DIR = 'data/processed'
ANALYSIS_DIR  = 'data/analysis'
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def run_pipeline():

    # 1. load
    print("Loading data...")
    covid   = pd.read_csv(f'{PROCESSED_DIR}/covid.csv').dropna(subset=['text'])
    climate = pd.read_csv(f'{PROCESSED_DIR}/climate.csv').dropna(subset=['text'])

    print(f"  COVID:   {len(covid):,} rows")
    print(f"  Climate: {len(climate):,} rows")

    # 2. clean & lemmatize
    print("\nLemmatizing text (this may take a few minutes)...")
    covid['text_lemma']   = lemmatize_series(covid['text'].tolist())
    climate['text_lemma'] = lemmatize_series(climate['text'].tolist())

    covid = covid[covid['text_lemma'].str.strip() != ''].reset_index(drop=True)
    climate = climate[climate['text_lemma'].str.strip() != ''].reset_index(drop=True)

    print(f"  After lemmatization — COVID: {len(covid):,}, Climate: {len(climate):,}")

    # 3. TF-IDF
    print("\nRunning TF-IDF comparison...")
    vectorizer, tfidf_matrix, results = top_terms_per_domain(
        covid['text_lemma'].tolist(),
        climate['text_lemma'].tolist(),
        top_n=25
    )
    print_top_terms(results, top_n=25)

    # 3. save
    tfidf_serializable = {
        domain: [(term, float(score)) for term, score in terms]
        for domain, terms in results.items()
    }
    with open(f'{ANALYSIS_DIR}/top_tfidf_terms.json', 'w') as f:
        json.dump(tfidf_serializable, f, indent=2)
    print(f"\nSaved → {ANALYSIS_DIR}/top_tfidf_terms.json")


if __name__ == '__main__':
    run_pipeline()