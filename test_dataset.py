import pandas as pd

df = pd.read_csv("arxiv_dataset.csv")

print(df.head())

print("Total papers:", len(df))

print("Categories:", df['category'].unique())