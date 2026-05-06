"""
objective3_summarization_recommendation.py – Objective 3: Summarization & Recommendation

Provides:
  - Extractive summarization using sentence embeddings + cosine similarity
  - Recommendation engine using precomputed SBERT embeddings
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import re

from preprocess import preprocess_text

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET      = "final_arxiv_dataset.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
SBERT_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
print("[1/4] Loading dataset ...")
df = pd.read_csv(DATASET).dropna().reset_index(drop=True)
df["text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)

# ──────────────────────────────────────────────
# SBERT EMBEDDINGS
# ──────────────────────────────────────────────
print("[2/4] Loading / computing SBERT embeddings ...")
sbert = SentenceTransformer(SBERT_MODEL)
emb_path = os.path.join(MODELS_DIR, "sbert_embeddings.npz")

if os.path.exists(emb_path):
    embeddings = np.load(emb_path)["embeddings"]
    print("   Loaded cached embeddings")
    if len(embeddings) != len(df):
        print("   WARNING: Size mismatch – recomputing ...")
        embeddings = sbert.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
        np.savez_compressed(emb_path, embeddings=embeddings)
else:
    embeddings = sbert.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
    np.savez_compressed(emb_path, embeddings=embeddings)

print(f"   Embedding matrix shape: {embeddings.shape}")


# ──────────────────────────────────────────────
# EXTRACTIVE SUMMARIZER
# ──────────────────────────────────────────────
def split_sentences(text):
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extractive_summary(abstract, top_n=3):
    """
    Rank sentences by cosine similarity to the full-abstract embedding,
    return the top-N as a summary.
    """
    sentences = split_sentences(abstract)
    if len(sentences) <= top_n:
        return abstract.strip()

    # Encode full abstract + each sentence
    all_texts = [abstract] + sentences
    embs = sbert.encode(all_texts)
    abstract_emb = embs[0].reshape(1, -1)
    sent_embs = embs[1:]

    # Cosine similarity
    sims = cosine_similarity(sent_embs, abstract_emb).flatten()
    top_idx = np.argsort(sims)[-top_n:]
    top_idx = sorted(top_idx)  # keep original order

    summary = " ".join([sentences[i] for i in top_idx])
    return summary


# ──────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ──────────────────────────────────────────────
def recommend_papers(query, top_k=TOP_K):
    """
    Encode user query with SBERT, compute cosine similarity to all
    paper embeddings, return top-K results with title, summary, and score.
    """
    query_emb = sbert.encode([query]).reshape(1, -1)
    sims = cosine_similarity(query_emb, embeddings).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]

    results = []
    for i in top_idx:
        row = df.iloc[i]
        summary = extractive_summary(str(row["abstract"]))
        results.append({
            "Rank": len(results) + 1,
            "Title": row["title"],
            "Category": row["category"],
            "Similarity": round(float(sims[i]), 4),
            "Summary": summary,
        })
    return results


# ──────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[3/4] Testing extractive summarization ...")
    sample_abstract = df.iloc[0]["abstract"]
    print(f"   Original ({len(sample_abstract)} chars):")
    print(f"   {sample_abstract[:200]}...\n")
    summary = extractive_summary(sample_abstract)
    print(f"   Summary ({len(summary)} chars):")
    print(f"   {summary[:200]}...\n")

    print("[4/4] Testing recommendation engine ...")
    query = "deep learning for image classification"
    print(f"   Query: \"{query}\"\n")
    recs = recommend_papers(query, top_k=5)
    for r in recs:
        print(f"   #{r['Rank']}  [{r['Category']}]  Sim={r['Similarity']:.4f}")
        print(f"       {r['Title'][:80]}")
        print(f"       Summary: {r['Summary'][:120]}...\n")

    print("Done OK")
