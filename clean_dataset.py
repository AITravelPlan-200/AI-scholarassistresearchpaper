import pandas as pd

df = pd.read_csv("arxiv_dataset.csv")

# remove empty rows
df = df.dropna()

# clean text
df['abstract'] = df['abstract'].str.replace('\n', ' ')
df['title'] = df['title'].str.replace('\n', ' ')

# remove duplicates
df = df.drop_duplicates()

# save cleaned dataset
df.to_csv("final_arxiv_dataset.csv", index=False)

print("Dataset cleaned successfully")
print("Total papers:", len(df))
print("Categories:", df['category'].nunique())