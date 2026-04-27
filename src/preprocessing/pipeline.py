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
from src.preprocessing.normalize import normalize_cdl, normalize_princeton, normalize_climate_fever, normalize_quotaclimat, normalize_climate_fever_direct, normalize_climatecheck
from src.preprocessing.harmonize_labels import harmonize_labels
from src.preprocessing.filter_datasets import filter_cdl, get_covid, get_climate, get_general
from src.preprocessing.clean import clean_series, filter_english

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

def run_pipeline():
    # load raw files
    print("Loading raw files...")

    cdl_general = pd.read_csv(f'{RAW_DIR}/complexdatalab.csv', low_memory=False)
    cdl_general['dataset'] = 'complexdatalab'
    cdl_covid = pd.read_csv(f'{RAW_DIR}/covid_19_rumor.csv', low_memory=False)
    cdl_covid['dataset'] = 'covid_19_rumor'
    cdl_climate = pd.read_csv(f'{RAW_DIR}/climate_fever.csv', low_memory=False)
    cdl_climate['dataset'] = 'climate_fever'
    cdl_quotaclimat = pd.read_csv(f'{RAW_DIR}/quotaclimat.csv', low_memory=False)
    cdl_quotaclimat['dataset'] = 'quotaclimat'
    cdl_climate_fever_direct = pd.read_csv(f'{RAW_DIR}/climate_fever_direct.csv', low_memory=False)
    cdl_climate_fever_direct['dataset'] = 'climate_fever_direct'
    princeton = pd.read_excel(f'{RAW_DIR}/jns-covid_misinfo_2021-03-06_Final_Clean.xlsx')
    cdl_climatecheck = pd.read_csv(f'{RAW_DIR}/climatecheck.csv', low_memory=False)
    cdl_climatecheck['dataset'] = 'climatecheck'

# these print statements were helpful for debugging the messy CDL dataset, but can be commented out now that the pipeline is working end-to-end
    # print("ACTUAL CDL COLUMNS:", cdl.columns.tolist())
    # print(cdl.head(2))    

    # normalize column names
    print("Normalizing columns...")
    cdl_general = normalize_cdl(cdl_general)
    cdl_covid = normalize_cdl(cdl_covid)
    cdl_climate = normalize_climate_fever(cdl_climate) 
    princeton = normalize_princeton(princeton)
    cdl_quotaclimat = normalize_quotaclimat(cdl_quotaclimat)
    cdl_climate_fever_direct = normalize_climate_fever_direct(cdl_climate_fever_direct)
    cdl_climatecheck = normalize_climatecheck(cdl_climatecheck)

    # harmonize labels
    print("Harmonizing labels...")
    cdl_general = harmonize_labels(cdl_general, 'cdl')
    cdl_covid = harmonize_labels(cdl_covid, 'cdl')
    cdl_climate = harmonize_labels(cdl_climate, 'cdl')
    princeton = harmonize_labels(princeton, 'princeton')
    cdl_quotaclimat = harmonize_labels(cdl_quotaclimat, 'quotaclimat')
    cdl_climate_fever_direct = harmonize_labels(cdl_climate_fever_direct, 'climate_fever_direct')
    cdl_climatecheck = harmonize_labels(cdl_climatecheck, 'climatecheck')

    # assign domains to CDL 
    print("Filtering domains...")
    cdl_general = filter_cdl(cdl_general)
    cdl_covid = filter_cdl(cdl_covid)
    cdl_climate = filter_cdl(cdl_climate)
    cdl_quotaclimat = filter_cdl(cdl_quotaclimat)
    cdl_climate_fever_direct = filter_cdl(cdl_climate_fever_direct)
    cdl_climatecheck = filter_cdl(cdl_climatecheck)


    # Ensuring no claims present are credible.
    cdl_general = cdl_general[cdl_general['label'] != 'credible'].reset_index(drop=True)
    cdl_covid = cdl_covid[cdl_covid['label'] != 'credible'].reset_index(drop=True)
    cdl_climate = cdl_climate[cdl_climate['label'] != 'credible'].reset_index(drop=True)
    cdl_quotaclimat = cdl_quotaclimat[cdl_quotaclimat['label'] != 'credible'].reset_index(drop=True)
    cdl_climate_fever_direct = cdl_climate_fever_direct[cdl_climate_fever_direct['label'] != 'credible'].reset_index(drop=True)
    cdl_climatecheck = cdl_climatecheck[cdl_climatecheck['label'] != 'credible'].reset_index(drop=True)

    # split CDL into three domain buckets and merge climate and covid rows into their respective domains (because some CDL datasets are general, some are covid, some are climate)
    covid_cdl = pd.concat([get_covid(cdl_covid), get_covid(cdl_general)], ignore_index=True)
    climate = pd.concat([
    get_climate(cdl_climate), get_climate(cdl_general), get_climate(cdl_quotaclimat), get_climate(cdl_climate_fever_direct), get_climate(cdl_climatecheck)], ignore_index=True)
    general   = get_general(cdl_general)

    # merge princeton into covid
    shared_cols = ['text', 'label', 'domain', 'source', 'dataset']
    covid = pd.concat([covid_cdl[shared_cols], princeton[shared_cols]], ignore_index=True)

    # clean text - removing newlines, emojis, URLs, mentions, hashtags, rt prefix, and lowercasing
    print("Cleaning text (removing newlines, emojis, URLs)...")
    covid['text'] = clean_series(covid['text'])
    climate['text'] = clean_series(climate['text'])
    general['text'] = clean_series(general['text'])


    # DROP EMPTY STRINGS
    # (Because cleaning might turn a tweet that was ONLY an emoji/URL into an empty string)
    covid = covid[covid['text'] != '']
    climate = climate[climate['text'] != '']
    general = general[general['text'] != '']

    # final check for duplicates and missing values in text column before saving
    covid = covid.drop_duplicates(subset=['text']).dropna(subset=['text']).reset_index(drop=True)
    climate = climate.drop_duplicates(subset=['text']).dropna(subset=['text']).reset_index(drop=True)
    general = general.drop_duplicates(subset=['text']).dropna(subset=['text']).reset_index(drop=True)

    # FILTER NON-ENGLISH TWEETS IN COVID
    print("Filtering non-English tweets...")
    covid = filter_english(covid, 'text')

    # save processed CSVs
    print("Saving processed files...")
    covid.to_csv(f'{PROCESSED_DIR}/covid.csv', index=False)
    climate.to_csv(f'{PROCESSED_DIR}/climate.csv', index=False)
    general.to_csv(f'{PROCESSED_DIR}/general.csv', index=False)

    print(f"Done... covid: {len(covid)} rows, climate: {len(climate)} rows, general: {len(general)} rows")

if __name__ == '__main__':
    run_pipeline()