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
    # Route both cdl and climate_fever_direct to the CDL harmonizer
    if source == 'cdl' or source == 'climate_fever_direct': 
        return harmonize_cdl_labels(df)
        
    elif source == 'princeton':
        return df # this has already been normalized 
        
    # NEW: Add routing for quotaclimat
    elif source == 'quotaclimat':
        # Quotaclimat labels are specific denial taxonomies (e.g., '1_not_happening')
        # We can harmonize all of these as 'misinformation'
        df['label'] = 'misinformation'
        return df
        
    else:
        raise ValueError(f"Unknown source: {source}")