"""
objective1_nlp_comparison.py – Objective 1: NLP Text Representation Comparison

Compares TF-IDF, CountVectorizer, LDA, SBERT, and Mirror-BERT for
classifying arXiv papers using Logistic Regression. Saves fitted
models/embeddings for reuse.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
import joblib

from preprocess import preprocess_text

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET       = "final_arxiv_dataset.csv"
MODELS_DIR    = "models"
RESULTS_DIR   = "results"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
MAX_FEATURES  = 5000
LDA_TOPICS    = 10
SBERT_MODEL   = "all-MiniLM-L6-v2"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD & PREPROCESS
# ──────────────────────────────────────────────
print("[1/6] Loading and preprocessing dataset ...")
df = pd.read_csv(DATASET).dropna()
df["text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)
df["processed_text"] = df["text"].apply(preprocess_text)

texts_raw   = df["text"].values
texts_proc  = df["processed_text"].values

le = LabelEncoder()
y  = le.fit_transform(df["category"].astype(str))
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
    texts_raw, y, np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train_proc = texts_proc[idx_train]
X_test_proc  = texts_proc[idx_test]

# ──────────────────────────────────────────────
# HELPER: evaluate with Logistic Regression
# ──────────────────────────────────────────────
def evaluate(X_tr, X_te):
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

results = []

# ──────────────────────────────────────────────
# 1. TF-IDF
# ──────────────────────────────────────────────
print("[2/6] TF-IDF ...")
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
tfidf.fit(X_train_proc)
X_tr = tfidf.transform(X_train_proc)
X_te = tfidf.transform(X_test_proc)
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
m = evaluate(X_tr, X_te)
results.append({"Method": "TF-IDF", **m})
print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1 Score']:.4f}")

# ──────────────────────────────────────────────
# 2. Count Vectorizer
# ──────────────────────────────────────────────
print("[3/6] Count Vectorizer ...")
count = CountVectorizer(max_features=MAX_FEATURES)
count.fit(X_train_proc)
X_tr_c = count.transform(X_train_proc)
X_te_c = count.transform(X_test_proc)
joblib.dump(count, os.path.join(MODELS_DIR, "count_vectorizer.pkl"))
m = evaluate(X_tr_c, X_te_c)
results.append({"Method": "Count Vectorizer", **m})
print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1 Score']:.4f}")

# ──────────────────────────────────────────────
# 3. LDA (on count vectors)
# ──────────────────────────────────────────────
print("[4/6] LDA ({} topics) ...".format(LDA_TOPICS))
lda = LatentDirichletAllocation(n_components=LDA_TOPICS, random_state=RANDOM_STATE)
X_tr_lda = lda.fit_transform(X_tr_c)
X_te_lda = lda.transform(X_te_c)
joblib.dump(lda, os.path.join(MODELS_DIR, "lda_model.pkl"))
m = evaluate(X_tr_lda, X_te_lda)
results.append({"Method": "LDA", **m})
print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1 Score']:.4f}")

# ──────────────────────────────────────────────
# 4. Sentence-BERT
# ──────────────────────────────────────────────
print("[5/6] Sentence-BERT ...")
sbert = SentenceTransformer(SBERT_MODEL)

sbert_emb_path = os.path.join(MODELS_DIR, "sbert_embeddings.npz")
if os.path.exists(sbert_emb_path):
    data = np.load(sbert_emb_path)
    all_sbert = data["embeddings"]
    print("   (loaded cached embeddings)")
else:
    all_sbert = sbert.encode(list(texts_raw), show_progress_bar=True, batch_size=64)
    np.savez_compressed(sbert_emb_path, embeddings=all_sbert)

X_tr_sb = all_sbert[idx_train]
X_te_sb = all_sbert[idx_test]
m = evaluate(X_tr_sb, X_te_sb)
results.append({"Method": "Sentence-BERT", **m})
print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1 Score']:.4f}")

# ──────────────────────────────────────────────
# 5. Mirror-BERT (L2-normalised SBERT)
# ──────────────────────────────────────────────
print("[6/6] Mirror-BERT ...")
X_tr_mb = X_tr_sb / (np.linalg.norm(X_tr_sb, axis=1, keepdims=True) + 1e-10)
X_te_mb = X_te_sb / (np.linalg.norm(X_te_sb, axis=1, keepdims=True) + 1e-10)
m = evaluate(X_tr_mb, X_te_mb)
results.append({"Method": "Mirror-BERT", **m})
print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1 Score']:.4f}")

# ──────────────────────────────────────────────
# RESULTS TABLE
# ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "obj1_feature_comparison.csv"), index=False)
print("\n" + results_df.to_string(index=False))

# ──────────────────────────────────────────────
# COMPARISON BAR CHART
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.2
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=color)

ax.set_xlabel("Feature Extraction Method")
ax.set_ylabel("Score")
ax.set_title("Objective 1 – Feature Extraction Method Comparison")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df["Method"], rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "obj1_comparison_chart.png"), dpi=150)
print(f"\nChart saved -> {RESULTS_DIR}/obj1_comparison_chart.png")
print("Done.")
