import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time

base_url = "http://export.arxiv.org/api/query?"

search_query = "all:machine learning"
start = 0
max_results = 100
total_papers = 2500 # total dataset size you want

papers = []

print("Downloading papers from arXiv...")

while start < total_papers:

    query = f"search_query={search_query}&start={start}&max_results={max_results}"
    url = base_url + query

    response = requests.get(url)

    root = ET.fromstring(response.content)

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):

        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text

        category = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]

        papers.append([title, abstract, category])

    start += max_results

    print("Downloaded:", start)

    time.sleep(3)   # avoid API limit

df = pd.DataFrame(papers, columns=["title", "abstract", "category"])

df.to_csv("arxiv_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total papers:", len(df))