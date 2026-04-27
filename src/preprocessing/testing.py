from datasets import load_dataset
df = load_dataset("rabuahmad/climatecheck", split="train").to_pandas()
print(df['annotation'].value_counts())
print(df['narrative'].value_counts().head(10))