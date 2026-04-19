import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis.stats import SUMMARY_PATH, TEST_RESULTS_PATH, run_stats
from src.analysis.tfidf_context import TFIDF_CONTEXT_PATH, run_tfidf_context_analysis
from src.features.emotion import get_emotions_batch
from src.features.lemmatization import lemmatize_series
from src.features.tf_idf import print_top_terms, top_terms_per_domain
from src.features.vader import get_vader_scores
from src.features.visualize import plot_all, plot_tfidf_context

PROCESSED_DIR = "data/processed"
ANALYSIS_DIR = "data/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def run_pipeline():
    print("Loading data...")
    covid = pd.read_csv(f"{PROCESSED_DIR}/covid.csv").dropna(subset=["text"])
    climate = pd.read_csv(f"{PROCESSED_DIR}/climate.csv").dropna(subset=["text"])

    print(f"  COVID:   {len(covid):,} rows")
    print(f"  Climate: {len(climate):,} rows")

    print("\nLemmatizing text (this may take a few minutes)...")
    covid["text_lemma"] = lemmatize_series(covid["text"].tolist())
    climate["text_lemma"] = lemmatize_series(climate["text"].tolist())

    covid = covid[covid["text_lemma"].str.strip() != ""].reset_index(drop=True)
    climate = climate[climate["text_lemma"].str.strip() != ""].reset_index(drop=True)
    print(f"  After lemmatization - COVID: {len(covid):,}, Climate: {len(climate):,}")

    print("\nRunning emotion analysis...")
    covid_emotions = get_emotions_batch(covid["text"].tolist(), batch_size=32)
    climate_emotions = get_emotions_batch(climate["text"].tolist(), batch_size=32)
    covid = pd.concat([covid.reset_index(drop=True), covid_emotions], axis=1)
    climate = pd.concat([climate.reset_index(drop=True), climate_emotions], axis=1)

    print("\nScoring row-level VADER sentiment...")
    covid_vader = get_vader_scores(covid["text"].tolist())
    climate_vader = get_vader_scores(climate["text"].tolist())
    covid = pd.concat([covid.reset_index(drop=True), covid_vader], axis=1)
    climate = pd.concat([climate.reset_index(drop=True), climate_vader], axis=1)

    covid_path = f"{ANALYSIS_DIR}/covid_emotions.csv"
    climate_path = f"{ANALYSIS_DIR}/climate_emotions.csv"
    covid.to_csv(covid_path, index=False)
    climate.to_csv(climate_path, index=False)
    print("  Saved emotion and VADER scores.")

    print("\nRunning TF-IDF comparison...")
    vectorizer, tfidf_matrix, results = top_terms_per_domain(
        covid["text_lemma"].tolist(),
        climate["text_lemma"].tolist(),
        top_n=25,
    )
    print_top_terms(results, top_n=25)

    print("\nCalculating average VADER sentiment scores...")
    covid_sentiment = covid_vader.mean()
    climate_sentiment = climate_vader.mean()
    sentiment_results = {
        "covid": {
            "neg": float(covid_sentiment["vader_neg"]),
            "neu": float(covid_sentiment["vader_neu"]),
            "pos": float(covid_sentiment["vader_pos"]),
            "compound": float(covid_sentiment["vader_compound"]),
        },
        "climate": {
            "neg": float(climate_sentiment["vader_neg"]),
            "neu": float(climate_sentiment["vader_neu"]),
            "pos": float(climate_sentiment["vader_pos"]),
            "compound": float(climate_sentiment["vader_compound"]),
        },
    }

    print("\nAverage VADER Sentiment Scores:")
    for domain, scores in sentiment_results.items():
        print(f"  {domain.capitalize()}:")
        print(f"    Neg: {scores['neg']:.4f}")
        print(f"    Neu: {scores['neu']:.4f}")
        print(f"    Pos: {scores['pos']:.4f}")
        print(f"    Compound: {scores['compound']:.4f}")

    tfidf_path = f"{ANALYSIS_DIR}/top_tfidf_terms.json"
    tfidf_serializable = {
        domain: [(term, float(score)) for term, score in terms]
        for domain, terms in results.items()
    }
    with open(tfidf_path, "w", encoding="utf-8") as handle:
        json.dump(tfidf_serializable, handle, indent=2)
    print(f"\nSaved -> {tfidf_path}")

    vader_path = f"{ANALYSIS_DIR}/vader_sentiment.json"
    with open(vader_path, "w", encoding="utf-8") as handle:
        json.dump(sentiment_results, handle, indent=2)
    print(f"Saved -> {vader_path}")

    print("\nBuilding TF-IDF context summaries...")
    run_tfidf_context_analysis(covid, climate, vectorizer, tfidf_matrix, results)

    print("\nRunning statistical comparisons...")
    run_stats(covid_path=covid_path, climate_path=climate_path)

    print("\nRendering summary figure...")
    plot_all(
        covid_path=covid_path,
        climate_path=climate_path,
        tfidf_path=tfidf_path,
        vader_path=vader_path,
        output_path=f"{ANALYSIS_DIR}/misinformation_analysis.png",
        stats_path=str(TEST_RESULTS_PATH),
        summary_path=str(SUMMARY_PATH),
        show=False,
    )
    plot_tfidf_context(
        context_path=str(TFIDF_CONTEXT_PATH),
        output_path=f"{ANALYSIS_DIR}/tfidf_context_comparison.png",
        show=False,
    )


if __name__ == "__main__":
    run_pipeline()
