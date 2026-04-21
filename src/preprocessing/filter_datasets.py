"""
Filtering MuMin to covid/climate tweets
"""
import pandas as pd
# dataset names for every domain
COVID_D = [
    'coaid', 'fibvid', 'cmu_miscov19', 'covid_19_rumor', 'covidfact',
    'fakecovid', 'counter-covid-19-misinformation', 'mm-covid', 'antivax',
    'nlp4if', 'esoc', 'checkcovid', 'covid-19-disinformation',
    'covid_vaccine_misinfo_mic'
]

CLIMATE_D = ['climate_fever', 'climate_fever_direct', 'quotaclimat', 'climatecheck']

def filter_cdl(df):
    df['domain'] = 'general'
    df.loc[df['dataset'].isin(COVID_D), 'domain'] = 'covid' # rows named covid are named covid
    df.loc[df['dataset'].isin(CLIMATE_D), 'domain'] = 'climate'

    return df

def get_covid(df):
    return df[df['domain'] == 'covid'].reset_index(drop= True) # return coviddomain rows

def get_climate(df):
    return df[df['domain'] == 'climate'].reset_index(drop=True) # return climate domain rows

def get_general(df):
    return df[df['domain'] == 'general'].reset_index(drop=True) #return general domain rows