from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def build_tf_idf(texts, max_features=5000, ngram_range=(1,2)):
    """
    Fit a TF-IDF vectorizer on a list of texts.
    Returns the fitted vectorizer and the sparse matrix.
    ngram_range=(1,2) captures both unigrams and bigrams - 
    useful for catching phrases like 'climate change' or 'herd immunity'.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=ngram_range,
        stop_words='english',  # remove common English stop words
        min_df=5,  # only include terms that appear in at least 5 documents to reduce noise
        max_df=0.95,  # ignore terms that appear in more than 95% of documents (too common to be informative)
        sublinear_tf=True, # use sublinear term frequency scaling (1 + log(tf)) to dampen the effect of very common terms
        token_pattern=r'[a-zA-Z]{2,}'
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def top_terms_per_domain(covid_texts, climate_texts, top_n=25):
    """
    Fit a TF-IDF vectorizer on the combined texts and then extract the top N terms for each domain.
    """
    all_texts = covid_texts + climate_texts
    domain_labels = ['covid'] * len(covid_texts) + ['climate'] * len(climate_texts)
    vectorizer, tfidf_matrix = build_tf_idf(all_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results = {}
    for domain in ['covid', 'climate']:
        indices = [i for i, label in enumerate(domain_labels) if label == domain]
        avg_scores = tfidf_matrix[indices].mean(axis=0).A1  # average TF-IDF score for each term in this domain
        top_idx = avg_scores.argsort()[-top_n:][::-1]  # indices of top N terms
        results[domain] = list(zip(feature_names[top_idx], avg_scores[top_idx]))

    return vectorizer, tfidf_matrix, results

def print_top_terms(results, top_n=25):
    for domain, terms in results.items():
        print(f"\nTop {top_n} terms for {domain} domain:")
        for term, score in terms:
            print(f"{term}: {score:.4f}")





