"""
Importing ComplexDataLab/Misinfo_Datasets - the ClimateMist (Hugging Face)
"""

from datasets import load_dataset

RAW_DIR = "data/raw"


def download_complexdatalab():
    # loading only a split (training data) and converting into a dataframe saved as a csv file under the raw folder
    train_data = load_dataset("ComplexDataLab/Misinfo_Datasets", split = "train").to_pandas().to_csv(f"{RAW_DIR}/complexdatalab.csv", index=False)
    print("Downloaded ComplexDataLab")

if __name__ == "__main__":
    download_complexdatalab()
    


