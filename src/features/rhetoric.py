"""
Urgency, authority, and doubt language detection.
Three distinct rhetorical strategies in misinformation:
  - Urgency:    creates panic, pressure to share/act
  - Authority:  invokes science/institutions to appear credible or to attack them
  - Doubt:      undermines trust in mainstream sources
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
import re

URGENCY_WORDS = [
    'immediately', 'urgent', 'breaking', 'alert', 'warning', 'danger', 'critical',
    'must', 'act now', 'share', 'spread the word', 'tell everyone', 'before it',
    'too late', 'running out', 'crisis', 'emergency', 'deadly', 'kills', 'killing',
    'die', 'death', 'dying', 'dead', 'fatal', 'catastrophic', 'explosive', 'shocking'
]

AUTHORITY_WORDS = [
    'scientists', 'doctors', 'experts', 'study', 'studies', 'research', 'proven',
    'confirmed', 'official', 'government', 'who', 'cdc', 'fda', 'nih', 'ipcc',
    'university', 'harvard', 'stanford', 'published', 'journal', 'data shows',
    'according to', 'report', 'evidence', 'fact', 'truth', 'real', 'actual',
    'peer reviewed', 'peer-reviewed', 'nasa', 'noaa', 'scientists say'
]

DOUBT_WORDS = [
    'fake', 'hoax', 'lie', 'lies', 'conspiracy', 'cover up', 'coverup',
    'they dont want', 'hidden', 'suppressed', 'censored', 'banned', 'silenced',
    'wake up', 'mainstream media', 'msm', 'big pharma', 'deep state', 'agenda',
    'corrupt', 'follow the money', 'do your research', 'question', 'debunked',
    'misleading', 'manipulated', 'propaganda', 'narrative', 'so called'
]

def count_rhetoric_words(text, word_list):
    text = str(text).lower()
    return sum(1 for word in word_list 
               if re.search(r'\b' + re.escape(word) + r'\b', text))

def extract_rhetoric_features(df):
    """
    Extract urgency, authority, and doubt features from the text column of the dataframe for each row.
    """
    texts = df['text'].fillna('').astype(str)

    features = pd.DataFrame()
    features['domain'] = df['domain'].values

    for name, word_list in [('urgency', URGENCY_WORDS), 
                            ('authority', AUTHORITY_WORDS), 
                            ('doubt', DOUBT_WORDS)]:
        
        features[f"{name}_count"] = texts.apply(lambda t: count_rhetoric_words(t, word_list))
        features[f"has_{name}"] = (features[f"{name}_count"] > 0).astype(int)

    features['authority_to_doubt_ratio'] = (features['authority_count'] / (features['doubt_count'] + 1))
    return features

def run_rhetoric_tests(features_df):
    """
    Compare urgency/authority/doubt between COVID and climate domains.
    Continuous counts → Mann-Whitney U
    Binary presence   → Chi-squared
    """
    covid = features_df[features_df['domain'] == 'covid']
    climate = features_df[features_df['domain'] == 'climate']

    print(f"\n{'='*65}")
    print(f"Rhetoric feature comparison: COVID ({len(covid)}) vs Climate ({len(climate)})")
    print(f"{'='*65}")

    print(f"\n{'Feature':<28} {'COVID mean':>12} {'Climate mean':>14} {'p-value':>12} Sig")
    print('-' * 72)

    continuous_features = ['urgency_count', 'authority_count', 'doubt_count', 'authority_to_doubt_ratio']
    results = []

    for feature in continuous_features:
        stat, p = mannwhitneyu(covid[feature], climate[feature], alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{feature:<28} {covid[feature].mean():>12.4f} {climate[feature].mean():>14.4f} {p:>12.4e} {sig}")
        results.append({'feature': feature, 'covid_mean': covid[feature].mean(), 'climate_mean': climate[feature].mean(), 'p_value': p, 'test': 'mann-whitney'})

    print(f"\n{'Binary Feature':<28} {'COVID %':>12} {'Climate %':>14} {'p-value':>12} Sig")
    print('-' * 72)

    binary = ['has_urgency', 'has_authority', 'has_doubt']
    for feature in binary:
        cv_yes, cl_yes = covid[feature].sum(), climate[feature].sum()
        cv_no, cl_no = len(covid) - cv_yes, len(climate) - cl_yes
        chi2, p, _, _ = chi2_contingency([[cv_yes, cv_no], [cl_yes, cl_no]])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        cv_pct = cv_yes / len(covid) * 100
        cl_pct = cl_yes / len(climate) * 100
        print(f"{feature:<28} {cv_pct:>9.1f}% {cl_pct:>11.1f}% {p:>12.4e} {sig}")
        results.append({'feature': feature, 'covid_mean': cv_pct, 'climate_mean': cl_pct, 'p_value': p, 'test': 'chi-squared'})

    return pd.DataFrame(results).sort_values('p_value')



        
