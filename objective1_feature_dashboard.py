import pandas as pd
import numpy as np
import re
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("final_arxiv_dataset.csv").dropna()
df["text"] = df["title"] + " " + df["abstract"]
texts = df["text"].astype(str)
labels = df["category"].astype(str)

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42
)

# -----------------------------
# FEATURE MODELS
# -----------------------------
# TF-IDF / CountVectorizer on raw text (no stopword removal)
tfidf = TfidfVectorizer(max_features=5000)
count_vec = CountVectorizer(max_features=5000)

tfidf.fit(texts)
count_vec.fit(texts)

# LDA model
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

# SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# SBERT preprocessing only
# -----------------------------
def preprocess_for_sbert(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(X_train_vec, X_test_vec):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return acc, precision, recall, f1

# -----------------------------
# FEATURE COMPARISON
# -----------------------------
def run_feature_comparison():
    results = []

    # TF-IDF
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    acc,p,r,f = evaluate(X_train_tfidf,X_test_tfidf)
    results.append(["TF-IDF",acc,p,r,f])

    # Count Vectorizer
    X_train_count = count_vec.transform(X_train)
    X_test_count = count_vec.transform(X_test)
    acc,p,r,f = evaluate(X_train_count,X_test_count)
    results.append(["Count Vectorizer",acc,p,r,f])

    # LDA
    X_train_lda = lda_model.fit_transform(X_train_count)
    X_test_lda = lda_model.transform(X_test_count)
    acc,p,r,f = evaluate(X_train_lda,X_test_lda)
    results.append(["LDA",acc,p,r,f])

    # Sentence-BERT
    X_train_sbert = sbert.encode(list(X_train))
    X_test_sbert = sbert.encode(list(X_test))
    acc,p,r,f = evaluate(X_train_sbert,X_test_sbert)
    results.append(["Sentence-BERT",acc,p,r,f])

    # Mirror-BERT
    X_train_m = sbert.encode(list(X_train))
    X_test_m = sbert.encode(list(X_test))
    X_train_m = X_train_m / np.linalg.norm(X_train_m,axis=1,keepdims=True)
    X_test_m = X_test_m / np.linalg.norm(X_test_m,axis=1,keepdims=True)
    acc,p,r,f = evaluate(X_train_m,X_test_m)
    results.append(["Mirror-BERT",acc,p,r,f])

    results_df = pd.DataFrame(
        results,
        columns=["Feature Method","Accuracy","Precision","Recall","F1 Score"]
    )

    # Graph
    plt.figure()
    plt.bar(results_df["Feature Method"], results_df["Accuracy"])
    plt.title("Feature Extraction Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.tight_layout()

    return results_df, plt

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(title, abstract, method):
    text = title + " " + abstract

    if method == "TF-IDF":
        vec = tfidf.transform([text]).toarray()
    elif method == "Count Vectorizer":
        vec = count_vec.transform([text]).toarray()
    elif method == "Sentence-BERT":
        vec = sbert.encode([preprocess_for_sbert(text)])
    elif method == "Mirror-BERT":
        vec = sbert.encode([preprocess_for_sbert(text)])
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    else:
        return "Method not supported"

    # Safety fallback if all zeros
    if np.all(vec == 0):
        vec = np.ones_like(vec) * 1e-6

    return str(vec[0][:20])

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Research Paper Feature Extraction Dashboard")
    gr.Markdown("Objective 1: Comparison of Feature Extraction Methods")

    with gr.Tab("Feature Extraction"):
        title = gr.Textbox(label="Paper Title")
        abstract = gr.Textbox(lines=6,label="Abstract")
        method = gr.Dropdown(
            ["TF-IDF","Count Vectorizer","Sentence-BERT","Mirror-BERT"],
            label="Feature Method"
        )
        btn = gr.Button("Extract Features")
        output = gr.Textbox(label="Feature Vector (First 20 values)")
        btn.click(extract_features,inputs=[title,abstract,method],outputs=output)

    with gr.Tab("Feature Comparison"):
        run = gr.Button("Run Evaluation")
        table = gr.Dataframe()
        graph = gr.Plot()
        run.click(run_feature_comparison,outputs=[table,graph])

demo.launch()