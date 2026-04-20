"""
Mapping the messy labels into a shared relationship across teh datasets
into two clean values
"""

import pandas as pd


def harmonize_cdl_labels(df):
    # mapping the binary labels to the unified schema
    label_map = {
        '1': 'misinformation',
        '0': 'credible'
    }
    df['label'] = df['label'].replace(label_map) # adding the mapping to the label column
    df = df.dropna(subset=['label']) # dop unmapped values

    return df

def harmonize_labels(df, source):
    if source == 'cdl': # sends the dataset to a harmonizer
        return harmonize_cdl_labels(df)
    elif source == 'princeton':
        return df # this has already been normalized 
    else:
        raise ValueError(f"Uknown source: {source}")