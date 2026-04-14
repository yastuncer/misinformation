"""
Preprocessing pipeline
loading raw datasets (CDL + Princeton)
normzalize columns, harmonize labels, assign domains
end results produced are 3 CVSs in data/processed
- covid.csv
- climate.csv
- general.csv
"""

import pandas as pd
from src.preprocessing.normalize import normalize_cdl, normalize_princeton
from src.preprocessing.harmonize_labels import harmonize_labels
from src.preprocessing.filter_datasets import filter_cdl, get_covid, get_climate, get_general

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

def run_pipeline():
    # load raw files
    print("Loading raw files...")
    cdl = pd.read_csv(f'{RAW_DIR}/complexdatalab.csv', low_memory=False)
    princeton = pd.read_excel(f'{RAW_DIR}/jns-covid_misinfo_2021-03-06_Final_Clean.xlsx')

    # normalize column names
    print("Normalizing columns...")
    cdl = normalize_cdl(cdl)
    princeton = normalize_princeton(princeton)

    # harmonize labels
    print("Harmonizing labels...")
    cdl = harmonize_labels(cdl, 'cdl')
    princeton = harmonize_labels(princeton, 'princeton')

    # assign domains to CDL 
    print("Filtering domains...")
    cdl = filter_cdl(cdl)

    # split CDL into three domain buckets
    covid_cdl = get_covid(cdl)
    climate = get_climate(cdl)
    general = get_general(cdl)

    # merge princeton into covid
    shared_cols = ['text', 'label', 'domain', 'source']
    covid = pd.concat([covid_cdl[shared_cols], princeton[shared_cols]], ignore_index=True)
    # save processed CSVs
    print("Saving processed files...")
    covid.to_csv(f'{PROCESSED_DIR}/covid.csv', index=False)
    climate.to_csv(f'{PROCESSED_DIR}/climate.csv', index=False)
    general.to_csv(f'{PROCESSED_DIR}/general.csv', index=False)

    print(f"Done... covid: {len(covid)} rows, climate: {len(climate)} rows, general: {len(general)} rows")

if __name__ == '__main__':
    run_pipeline()