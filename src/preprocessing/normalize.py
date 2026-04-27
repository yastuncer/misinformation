"""
Taking every dataset by 
grouping by renaming the columns into a unified schema
"""

import pandas as pd

def normalize_cdl(df):
    TEXT_COLS = [
        'tweet_text', 'claim', 'social_media_text',
        'article_headline', 'article_title', 'false_claim',
        'canonical_sentence', 'text'
    ]
    LABEL_COLS = [
        'binary_label', 'veracity', 'tweet_label',
        'claim_label', 'label', 'label_2', 'three_label'
    ]

    existing_text = [c for c in TEXT_COLS if c in df.columns]
    text_df = df[existing_text].replace('na', pd.NA).replace('n/a', pd.NA)
    df['text'] = text_df.bfill(axis=1).iloc[:, 0]

    existing_label = [c for c in LABEL_COLS if c in df.columns]
    label_df = df[existing_label].replace('na', pd.NA).replace('n/a', pd.NA)
    df['label'] = label_df.bfill(axis=1).iloc[:, 0]

    df['source'] = 'cdl'

    if 'dataset' not in df.columns:
        df['dataset'] = 'complexdatalab'

    return df[['text', 'label', 'source', 'dataset']]

def normalize_princeton(df):
    df = df.rename(columns={
        'Title': 'text'
    })
    df['label'] = 'misinformation'
    df['source'] = 'princeton'
    df['domain'] = 'covid'
    df['dataset'] = 'jns-covid_misinfo' 
    return df[['text', 'label', 'domain', 'source', 'dataset']]

def normalize_climate_fever(df):
    # drop existing 'label' column to avoid duplicate after renaming veracity → label
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    
    df = df.rename(columns={'claim': 'text', 'veracity': 'label'})
    df['source'] = 'cdl'
    if 'dataset' not in df.columns:
        df['dataset'] = 'climate_fever'
    return df[['text', 'label', 'source', 'dataset']]

def normalize_quotaclimat(df):
    df = df[df['language'] == 'en'].reset_index(drop=True)  # ← add this
    df = df.rename(columns={'quote': 'text'})
    df['source'] = 'quotaclimat'
    if 'dataset' not in df.columns:
        df['dataset'] = 'quotaclimat'
    return df[['text', 'label', 'source', 'dataset']]

def normalize_climate_fever_direct(df):
    df = df.rename(columns={'claim': 'text', 'claim_label': 'label'})
    df['source'] = 'climate_fever_direct'
    if 'dataset' not in df.columns:
        df['dataset'] = 'climate_fever_direct'
    return df[['text', 'label', 'source', 'dataset']]

def normalize_climatecheck(df):
    df = df.rename(columns={'claim': 'text', 'annotation': 'label'})
    df['source'] = 'climatecheck'
    df['dataset'] = 'climatecheck'
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    
    return df[['text', 'label', 'source', 'dataset']]