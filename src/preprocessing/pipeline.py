"""
Preprocessing pipeline
loading raw datasets (CDL + Princeton)
normzalize columns, harmonize labels, assign domains
end results produced are 3 CVSs in data/processed
- covid.csv
- climate.csv
- general.csv
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.preprocessing.normalize import normalize_cdl, normalize_princeton, normalize_climate_fever
from src.preprocessing.harmonize_labels import harmonize_labels
from src.preprocessing.filter_datasets import filter_cdl, get_covid, get_climate, get_general

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

def run_pipeline():
    # load raw files
    print("Loading raw files...")

    cdl_general = pd.read_csv(f'{RAW_DIR}/complexdatalab.csv', low_memory=False)
    cdl_covid = pd.read_csv(f'{RAW_DIR}/covid_19_rumor.csv', low_memory=False)
    cdl_covid['dataset'] = 'covid_19_rumor'
    cdl_climate = pd.read_csv(f'{RAW_DIR}/climate_fever.csv', low_memory=False)
    cdl_climate['dataset'] = 'climate_fever'
    princeton = pd.read_excel(f'{RAW_DIR}/jns-covid_misinfo_2021-03-06_Final_Clean.xlsx')

# these print statements were helpful for debugging the messy CDL dataset, but can be commented out now that the pipeline is working end-to-end
    # print("ACTUAL CDL COLUMNS:", cdl.columns.tolist())
    # print(cdl.head(2))    

    # normalize column names
    print("Normalizing columns...")
    cdl_general = normalize_cdl(cdl_general)
    cdl_covid = normalize_cdl(cdl_covid)
    cdl_climate = normalize_cdl(cdl_climate)
    princeton = normalize_princeton(princeton)
    cdl_climate = normalize_climate_fever(cdl_climate)

    # harmonize labels
    print("Harmonizing labels...")
    cdl_general = harmonize_labels(cdl_general, 'cdl')
    cdl_covid = harmonize_labels(cdl_covid, 'cdl')
    cdl_climate = harmonize_labels(cdl_climate, 'cdl')
    princeton = harmonize_labels(princeton, 'princeton')

    # assign domains to CDL 
    print("Filtering domains...")
    cdl_general = filter_cdl(cdl_general)
    cdl_covid = filter_cdl(cdl_covid)
    cdl_climate = filter_cdl(cdl_climate)

    # split CDL into three domain buckets
    covid_cdl = get_covid(cdl_covid)
    climate = get_climate(cdl_climate)
    general = get_general(cdl_general)

    # merge princeton into covid
    shared_cols = ['text', 'label', 'domain', 'source', 'dataset']
    covid = pd.concat([covid_cdl[shared_cols], princeton[shared_cols]], ignore_index=True)
    # save processed CSVs
    print("Saving processed files...")
    covid.to_csv(f'{PROCESSED_DIR}/covid.csv', index=False)
    climate.to_csv(f'{PROCESSED_DIR}/climate.csv', index=False)
    general.to_csv(f'{PROCESSED_DIR}/general.csv', index=False)

    print(f"Done... covid: {len(covid)} rows, climate: {len(climate)} rows, general: {len(general)} rows")

if __name__ == '__main__':
    run_pipeline()