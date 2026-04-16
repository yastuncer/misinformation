"""
Importing ComplexDataLab/Misinfo_Datasets - the ClimateMist (Hugging Face)
"""

from datasets import load_dataset
from openpyxl import load_workbook
import pandas as pd

RAW_DIR = "data/raw"

def download_complexdatalab():
    # loading only a split (training data) and converting into a dataframe saved as a csv file under the raw folder
    train_data = load_dataset("ComplexDataLab/Misinfo_Datasets", split = "train").to_pandas().to_csv(f"{RAW_DIR}/complexdatalab.csv", index=False)
    print("Downloaded ComplexDataLab")

def download_covid_rumor():
    dataset = load_dataset("ComplexDataLab/Misinfo_Datasets", "covid_19_rumor", split="train")
    dataset.to_pandas().to_csv(f"{RAW_DIR}/covid_19_rumor.csv", index=False)
    print("Saved to data/raw/covid_19_rumor.csv!")

def download_additional_covid_dataset():
    df = pd.read_excel(f"{RAW_DIR}/jns-covid_misinfo_2021-03-06_Final_Clean.xlsx")
    df.to_csv(f"{RAW_DIR}/additional_covid_dataset.csv", index=False)
    print(f"Success! Saved as {RAW_DIR}/additional_covid_dataset.csv")

def download_climate_fever():
    dataset = load_dataset("ComplexDataLab/Misinfo_Datasets", "climate_fever", split="train")
    dataset.to_pandas().to_csv(f"{RAW_DIR}/climate_fever.csv", index=False)
    print("Saved to data/raw/climate_fever.csv!")

if __name__ == "__main__":
    download_complexdatalab()
    download_covid_rumor()
    download_additional_covid_dataset()
    download_climate_fever()


