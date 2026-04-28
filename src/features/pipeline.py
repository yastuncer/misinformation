import sys
import os
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.analysis.stats import run_stats
from src.analysis.tfidf_context import TFIDF_CONTEXT_PATH, run_tfidf_context_analysis
from src.features.tf_idf import top_terms_per_domain, print_top_terms
from src.features.lemmatization import lemmatize_series
from src.features.vader import avg_vader, vader_series
from src.features.emotion import get_emotions_batch
from src.features.rhetoric import extract_rhetoric_features, run_rhetoric_tests
from src.features.visualize import plot_tfidf_context
from scipy.stats import mannwhitneyu


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

    # 3. emotion analysis
    # after lemmatization, before TF-IDF:
    print("\nRunning emotion analysis...")
    covid_emotions   = get_emotions_batch(covid['text'].tolist(), batch_size=32)
    climate_emotions = get_emotions_batch(climate['text'].tolist(), batch_size=32)

    # attach emotion columns to dataframes
    covid   = pd.concat([covid.reset_index(drop=True), covid_emotions], axis=1)
    climate = pd.concat([climate.reset_index(drop=True), climate_emotions], axis=1)

    # save with emotions
    covid.to_csv(f'{ANALYSIS_DIR}/covid_emotions.csv', index=False)
    climate.to_csv(f'{ANALYSIS_DIR}/climate_emotions.csv', index=False)
    print("  Saved emotion scores.")

    # 4. TF-IDF
    print("\nRunning TF-IDF comparison...")
    vectorizer, tfidf_matrix, results = top_terms_per_domain(
        covid['text_lemma'].tolist(),
        climate['text_lemma'].tolist(),
        top_n=25
    )
    print_top_terms(results, top_n=25)

    # 5. VADER sentiment analysis
    print("\nCalculating average VADER sentiment scores...")
    covid_sentiment = avg_vader(covid['text'].tolist())
    climate_sentiment = avg_vader(climate['text'].tolist())
    sentiment_results = {
        'covid': {
            'neg': covid_sentiment[0],
            'neu': covid_sentiment[1],
            'pos': covid_sentiment[2],
            'compound': covid_sentiment[3]
        },
        'climate': {
            'neg': climate_sentiment[0],
            'neu': climate_sentiment[1],
            'pos': climate_sentiment[2],
            'compound': climate_sentiment[3]
        }
    }
    covid_vader_df = vader_series(covid['text'].tolist())
    climate_vader_df = vader_series(climate['text'].tolist())
    covid = pd.concat([covid.reset_index(drop=True), covid_vader_df], axis=1)
    climate = pd.concat([climate.reset_index(drop=True), climate_vader_df], axis=1)
    covid_vader_df.to_csv(f'{ANALYSIS_DIR}/covid_vader.csv', index=False)
    climate_vader_df.to_csv(f'{ANALYSIS_DIR}/climate_vader.csv', index=False)
    print("  Saved per-row VADER sentiment scores.")

    print("\nAverage VADER Sentiment Scores:")
    for domain, scores in sentiment_results.items():
        print(f"  {domain.capitalize()}:")
        print(f"    Neg: {scores['neg']:.4f}")
        print(f"    Neu: {scores['neu']:.4f}")
        print(f"    Pos: {scores['pos']:.4f}")
        print(f"    Compound: {scores['compound']:.4f}")


    stat, p = mannwhitneyu(
        covid_vader_df['compound'],
        climate_vader_df['compound'],
        alternative='two-sided'
    )
    print(f"\nVADER compound Mann-Whitney U: p={p:.4e}")
    vader_results = {
        'compound_mwu_p': p,
        'covid_compound_mean': covid_vader_df['compound'].mean(),
        'climate_compound_mean': climate_vader_df['compound'].mean()
    }
    with open(f'{ANALYSIS_DIR}/vader_stats.json', 'w') as f:
        json.dump(vader_results, f, indent=2)

    # 6. rhetoric analysis
    # print("\nRunning rhetoric analysis...")
    # combined = pd.concat([covid, climate], ignore_index=True)
    # rhetoric_features = extract_rhetoric_features(combined)
    # rhetoric_stats = run_rhetoric_tests(rhetoric_features)

    # rhetoric_features.to_csv(f'{ANALYSIS_DIR}/rhetoric_features.csv', index=False)
    # rhetoric_stats.to_csv(f'{ANALYSIS_DIR}/rhetoric_stats.csv', index=False)
    # print(f"Saved → {ANALYSIS_DIR}/rhetoric_features.csv")
    # print(f"Saved → {ANALYSIS_DIR}/rhetoric_stats.csv")

    # 7. save
    tfidf_serializable = {
        domain: [(term, float(score)) for term, score in terms]
        for domain, terms in results.items()
    }
    with open(f'{ANALYSIS_DIR}/top_tfidf_terms.json', 'w') as f:
        json.dump(tfidf_serializable, f, indent=2)
    print(f"\nSaved → {ANALYSIS_DIR}/top_tfidf_terms.json")

    with open(f'{ANALYSIS_DIR}/vader_sentiment.json', 'w') as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"Saved → {ANALYSIS_DIR}/vader_sentiment.json")

    print("\nBuilding TF-IDF context summaries...")
    run_tfidf_context_analysis(covid, climate, vectorizer, tfidf_matrix, results)

    print("\nRunning statistical comparisons...")
    run_stats(
        covid_path=f'{ANALYSIS_DIR}/covid_emotions.csv',
        climate_path=f'{ANALYSIS_DIR}/climate_emotions.csv',
    )

    print("\nRendering TF-IDF context figure...")
    plot_tfidf_context(
        context_path=str(TFIDF_CONTEXT_PATH),
        output_path=f'{ANALYSIS_DIR}/tfidf_context_comparison.png',
        show=False,
    )
    
if __name__ == '__main__':
    run_pipeline()
