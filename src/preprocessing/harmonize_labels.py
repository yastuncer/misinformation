"""
Mapping the messy labels into a shared relationship across teh datasets
into two clean values
"""

import pandas as pd


# harmonize_labels.py

def harmonize_cdl_labels(df):
    label_map = {
        # binary_label values
        '1': 'misinformation',
        '0': 'credible',
        1: 'misinformation',
        0: 'credible',
        # veracity / tweet_label values
        'false': 'misinformation',
        'fake': 'misinformation',
        'misinformation': 'misinformation',
        'true': 'credible',
        'real': 'credible',
        'credible': 'credible',
        # climate_fever specific labels
        'refutes': 'misinformation',
        'supports': 'credible',
    }
    df['label'] = df['label'].astype(str).str.lower().replace(label_map)

    # Drop anything that didn't map cleanly (e.g. 'unknown', 'unverified')
    df = df[df['label'].isin(['misinformation', 'credible'])].reset_index(drop=True)

    return df

def harmonize_labels(df, source):
    if source == 'cdl': # sends the dataset to a harmonizer
        return harmonize_cdl_labels(df)
    elif source == 'princeton':
        return df # this has already been normalized 
    else:
        raise ValueError(f"Unknown source: {source}")