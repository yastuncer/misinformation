"""
Taking ever dataset by 
grouping by renaming the columns into a unified schema
"""

import pandas as pd

def normalize_cdl(df):
    
    if 'label' in df.columns: # to avoid duplicates
        df = df.drop(columns=['label'])
    # transforming exisiting DataFrame
    df = df.rename(columns= { # renaming the columns 
        'tweet_text': 'text',
        'binary_label': 'label'
    })
    df['source'] = 'cdl' # adding a source column

    return df[['text', 'label', 'source', 'dataset']] 


def normalize_princeton(df):
    df = df.rename(columns={
        'Title': 'text'
    })
    df['label'] = 'misinformation'
    df['source'] = 'princeton'
    df['domain'] = 'covid'

    return df[['text', 'label', 'domain', 'source']]