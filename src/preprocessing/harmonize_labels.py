"""
Mapping the messy labels into a shared relationship across the datasets
into two clean values
"""

import pandas as pd


def harmonize_cdl_labels(df):
    label_map = {
        '1': 'misinformation',
        '0': 'credible',
        'false': 'misinformation',
        'fake': 'misinformation',
        'misinformation': 'misinformation',
        'true': 'credible',
        'real': 'credible',
        'credible': 'credible',
        'refutes': 'misinformation',
        'supports': 'credible',
    }
    df['label'] = df['label'].astype(str).str.lower().replace(label_map)
    df = df[df['label'].isin(['misinformation', 'credible'])].reset_index(drop=True)
    return df

def harmonize_quotaclimat_labels(df):
    label_map = {
        '0_not_relevant': 'credible',
        '1_not_happening': 'misinformation',
        '2_not_human': 'misinformation',
        '3_not_bad': 'misinformation',
        '4_solutions_harmful_unnecessary': 'misinformation',
        '5_science_unreliable': 'misinformation',
        '6_proponents_biased': 'misinformation',
        '7_fossil_fuels_needed': 'misinformation'
    }
    df['label'] = df['label'].replace(label_map)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    return df

def harmonize_climate_fever_direct_labels(df):
    # 0 = SUPPORTS (credible), 1 = REFUTES (misinformation)
    # 2 = NOT_ENOUGH_INFO, 3 = DISPUTED — both dropped
    label_map = {
        0: 'credible',
        1: 'misinformation',
        2: None,
        3: None
    }
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    return df

def harmonize_climatecheck_labels(df):
    df['label'] = df['label'].apply(
        lambda x: 'credible' if str(x).startswith('0_') else 'misinformation'
    )
    df = df.dropna(subset=['label']).reset_index(drop=True)
    return df

def harmonize_labels(df, source):
    if source == 'cdl': # sends the dataset to a harmonizer
        return harmonize_cdl_labels(df)
    elif source == 'princeton':
        return df # this has already been normalized 
    elif source == 'quotaclimat':
        return harmonize_quotaclimat_labels(df)
    elif source == 'climate_fever_direct':
        return harmonize_climate_fever_direct_labels(df)
    elif source == 'climatecheck':
        return harmonize_climatecheck_labels(df)
    else:
        raise ValueError(f"Unknown source: {source}")