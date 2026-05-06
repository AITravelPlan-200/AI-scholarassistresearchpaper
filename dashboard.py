"""
dashboard.py – Unified Gradio Dashboard for all three objectives.

Tabs:
  1. Feature Comparison 
  2. Model Evaluation   
  3. Search & Recommend  
  4. Visualizations      (embeddings projection, topic distributions)
  5. Generate Summary
  6. AI Paper Analysis   (multi-agent LangGraph pipeline)
"""

import os, warnings, re, tempfile, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gradio as gr


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib

from preprocess import preprocess_text
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# MULTI-AGENT PIPELINE (from src/)
# ──────────────────────────────────────────────
agent_pipeline_ready = False
try:
    from src.utils import extract_text_from_pdf, split_text, upload_to_pinecone
    from src.graph import app as graph_app
    from main import get_paper_context
    agent_pipeline_ready = True
    print("✅ Multi-agent pipeline loaded successfully.")
except Exception as e:
    print(f"⚠ Multi-agent pipeline not available: {e}")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET      = "final_arxiv_dataset.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
SBERT_MODEL  = "all-MiniLM-L6-v2"
MAX_FEATURES = 5000
LDA_TOPICS   = 10
RANDOM_STATE = 42
TEST_SIZE    = 0.2

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD DATA ONCE
# ──────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATASET).dropna().reset_index(drop=True)
df["text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)
df["processed_text"] = df["text"].apply(preprocess_text)

# Filter categories to match Objective 2 (>=10 samples)
cat_counts = df["category"].value_counts()
top_cats = cat_counts[cat_counts >= 10].index.tolist()
df = df[df["category"].isin(top_cats)].copy().reset_index(drop=True)
print(f"Filtered dataset: {len(df)} papers in {len(top_cats)} categories.")

le = LabelEncoder()
y  = le.fit_transform(df["category"].astype(str))

texts_raw  = df["text"].to_numpy(dtype=str)
texts_proc = df["processed_text"].to_numpy(dtype=str)

X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
    texts_raw, y, np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train_proc = texts_proc[idx_train]
X_test_proc  = texts_proc[idx_test]

# Fit vectorizers
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
tfidf.fit(X_train_proc)
count = CountVectorizer(max_features=MAX_FEATURES)
count.fit(X_train_proc)
lda = LatentDirichletAllocation(n_components=LDA_TOPICS, random_state=RANDOM_STATE)
lda.fit(count.transform(X_train_proc))

# Load / compute SBERT embeddings
print("Loading SBERT model ...")
sbert = SentenceTransformer(SBERT_MODEL)
emb_path = os.path.join(MODELS_DIR, "sbert_embeddings.npz")
if os.path.exists(emb_path):
    all_sbert_emb = np.load(emb_path)["embeddings"]
    if len(all_sbert_emb) != len(df):
        all_sbert_emb = sbert.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
        np.savez_compressed(emb_path, embeddings=all_sbert_emb)
else:
    all_sbert_emb = sbert.encode(df["text"].tolist(), show_progress_bar=True, batch_size=64)
    np.savez_compressed(emb_path, embeddings=all_sbert_emb)

# ──────────────────────────────────────────────
# LOAD INFERENCE MODELS (DEPLOYMENT)
# ──────────────────────────────────────────────
print("Loading saved inference models for deployment ...")
models_cache = {}
vectorizers_cache = {}
backup_models = {} # For on-the-fly trained models

def load_deployment_assets():
    try:
        # Load Vectorizers
        vectorizers_cache["TF-IDF"] = joblib.load(os.path.join(MODELS_DIR, "tfidf_obj2.pkl"))
        vectorizers_cache["Count Vectorizer"] = joblib.load(os.path.join(MODELS_DIR, "count_vectorizer.pkl"))
        vectorizers_cache["LDA"] = joblib.load(os.path.join(MODELS_DIR, "lda_model.pkl"))
        
        # Load Classification Models
        model_files = {
            "SVM (LinearSVC)": "obj2_svm_(linearsvc).pkl",
            "Logistic Regression": "obj2_logistic_regression.pkl",
            "Random Forest": "obj2_random_forest.pkl",
            "Naïve Bayes": "obj2_naïve_bayes.pkl"
        }
        
        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                models_cache[name] = joblib.load(path)
            else:
                print(f"  ⚠ Model {name} not found at {path}")

        # Label Binarizer for consistent classes
        global inf_lb
        inf_lb = joblib.load(os.path.join(MODELS_DIR, "label_binarizer.pkl"))
        return True
    except Exception as e:
        print(f"  ⚠ Deployment assets error: {e}")
        return False

inference_ready = load_deployment_assets()




# ──────────────────────────────────────────────
# HELPER: evaluate Logistic Regression
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


# ══════════════════════════════════════════════
# NEW TAB: Classify New Paper (Deployment Inference)
# ══════════════════════════════════════════════
def classify_paper(title, abstract, model_name="SVM (LinearSVC)", feature_method="TF-IDF"):
    if not inference_ready:
        return "Models not loaded. Run Objective 2 pipeline first.", pd.DataFrame()
    
    text = str(title) + " " + str(abstract)
    if not text.strip():
        return "Please enter text.", pd.DataFrame()
        
    proc_text = preprocess_text(text)
    
    # Identify which vectorizer and model to use
    # Note: Objective 2 models were trained on TF-IDF. 
    # If a different feature is used, we might need a backup Logistic Regression or similar.
    # For now, we use the selected model.
    
    # 1. Feature Extraction
    if feature_method == "TF-IDF":
        vec = vectorizers_cache["TF-IDF"].transform([proc_text])
    elif feature_method == "Count Vectorizer":
        vec = vectorizers_cache["Count Vectorizer"].transform([proc_text])
    elif feature_method == "Sentence-BERT":
        vec = sbert.encode([text])
    elif feature_method == "Mirror-BERT":
        vec = sbert.encode([text])
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10)
    elif feature_method == "LDA":
        count_vec = vectorizers_cache["Count Vectorizer"].transform([proc_text])
        vec = vectorizers_cache["LDA"].transform(count_vec)
    else:
        return f"Feature method {feature_method} not supported yet for classification.", pd.DataFrame()

    # 2. Prediction
    if model_name not in models_cache:
        # Fallback to Logistic Regression if specific model not found
        if "Logistic Regression" in models_cache:
            model_name = "Logistic Regression"
        else:
            return f"Model {model_name} not available.", pd.DataFrame()
            
    clf = models_cache[model_name]
    
    # Check if we need to train a backup model for this feature method
    # Pretrained models expect 5000 features (TF-IDF/Count)
    # SBERT is 384, LDA is 10.
    if feature_method in ["Sentence-BERT", "Mirror-BERT", "LDA"]:
        cache_key = f"{model_name}_{feature_method}"
        if cache_key not in backup_models:
            print(f"Training backup model for {cache_key} ...")
            try:
                # Get all features for training
                if feature_method == "Sentence-BERT":
                    X_train_backup = all_sbert_emb[idx_train]
                elif feature_method == "Mirror-BERT":
                    X_tr_sb = all_sbert_emb[idx_train]
                    X_train_backup = X_tr_sb / (np.linalg.norm(X_tr_sb, axis=1, keepdims=True) + 1e-10)
                elif feature_method == "LDA":
                    X_tr_c = vectorizers_cache["Count Vectorizer"].transform(X_train_proc)
                    X_train_backup = vectorizers_cache["LDA"].transform(X_tr_c)
                
                # Use string labels for training to avoid index issues
                y_train_str = df.iloc[idx_train]["category"].values
                
                from sklearn.linear_model import LogisticRegression
                backup_clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
                backup_clf.fit(X_train_backup, y_train_str)
                backup_models[cache_key] = backup_clf
            except Exception as e:
                return f"Error training backup model: {e}", pd.DataFrame()
        
        clf = backup_models[cache_key]
    
    try:
        # Prediction logic
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(vec)[0]
        else:
            scores = clf.decision_function(vec)[0]
            probs = np.exp(scores) / np.sum(np.exp(scores))
            
        # Determine classes based on which model was used
        if feature_method in ["Sentence-BERT", "Mirror-BERT", "LDA"]:
            # Backup models were trained on string labels
            classes = clf.classes_
        else:
            # Pretrained models use LabelBinarizer classes
            classes = inf_lb.classes_

        top_indices = np.argsort(probs)[-3:][::-1]
        
        results = []
        for i in top_indices:
            cat_name = classes[i]
            results.append({
                "Category": cat_name,
                "Confidence": f"{probs[i]*100:.1f}%"
            })
        
        best_cat = classes[top_indices[0]]
        best_conf = probs[top_indices[0]] * 100
        msg = f"Model: **{model_name}** | Features: **{feature_method}**\n\nPredicted Category: **{best_cat}** (Confidence: {best_conf:.1f}%)"
        
        return msg, pd.DataFrame(results)
    except Exception as e:
        return f"Prediction Error: {str(e)}", pd.DataFrame()

# ══════════════════════════════════════════════
# TAB 1: Feature Comparison 
# ══════════════════════════════════════════════
def run_feature_comparison():
    results = []

    # TF-IDF
    X_tr = tfidf.transform(X_train_proc)
    X_te = tfidf.transform(X_test_proc)
    m = evaluate(X_tr, X_te)
    results.append({"Method": "TF-IDF", **m})

    # Count Vectorizer
    X_tr_c = count.transform(X_train_proc)
    X_te_c = count.transform(X_test_proc)
    m = evaluate(X_tr_c, X_te_c)
    results.append({"Method": "Count Vectorizer", **m})

    # LDA
    X_tr_lda = lda.transform(X_tr_c)
    X_te_lda = lda.transform(X_te_c)
    m = evaluate(X_tr_lda, X_te_lda)
    results.append({"Method": "LDA", **m})

    # SBERT
    X_tr_sb = all_sbert_emb[idx_train]
    X_te_sb = all_sbert_emb[idx_test]
    m = evaluate(X_tr_sb, X_te_sb)
    results.append({"Method": "Sentence-BERT", **m})

    # Mirror-BERT
    X_tr_mb = X_tr_sb / (np.linalg.norm(X_tr_sb, axis=1, keepdims=True) + 1e-10)
    X_te_mb = X_te_sb / (np.linalg.norm(X_te_sb, axis=1, keepdims=True) + 1e-10)
    m = evaluate(X_tr_mb, X_te_mb)
    results.append({"Method": "Mirror-BERT", **m})

    rdf = pd.DataFrame(results)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rdf))
    w = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (met, col) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * w, rdf[met], w, label=met, color=col)
    ax.set_xlabel("Feature Method")
    ax.set_ylabel("Score")
    ax.set_title("Feature Extraction Comparison")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(rdf["Method"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return rdf, fig


# ══════════════════════════════════════════════
# TAB 2: Feature Extraction (single paper)
# ══════════════════════════════════════════════
def extract_features(title, abstract, method):
    text = str(title) + " " + str(abstract)
    proc = preprocess_text(text)
    if method == "TF-IDF":
        vec = tfidf.transform([proc]).toarray()[0]
    elif method == "Count Vectorizer":
        vec = count.transform([proc]).toarray()[0]
    elif method == "Sentence-BERT":
        vec = sbert.encode([text])[0]
    elif method == "Mirror-BERT":
        vec = sbert.encode([text])[0]
        vec = vec / (np.linalg.norm(vec) + 1e-10)
    else:
        return "Unsupported method", ""
        
    # Create human-readable explanation
    explanation = f"### How the system 'sees' this paper ({method}):\n"
    if method in ["TF-IDF", "Count Vectorizer"]:
        vocab = tfidf.get_feature_names_out() if method == "TF-IDF" else count.get_feature_names_out()
        top_indices = np.argsort(vec)[-5:][::-1]
        keywords = [vocab[i] for i in top_indices if vec[i] > 0]
        explanation += f"This method looks for specific keywords. For this paper, it heavily weighted terms like: **{', '.join(keywords)}**.\n\n"
        explanation += "**Generalization:** The system will recognize any future paper that uses these or similar technical terms."
    else:
        explanation += "This method transforms the entire meaning into a 'Semantic Fingerprint'. Instead of just looking for words, it understands the *context* and *concepts*.\n\n"
        explanation += "**Generalization:** Even if a new paper uses different words (e.g., 'deep learning' instead of 'neural networks'), the system will still understand it because their 'fingerprints' will be mathematically similar."
        
    vector_str = f"Raw Numerical Signature (First 20 dimensions):\n{list(np.round(vec[:20], 4))}"
    return vector_str, explanation


# ══════════════════════════════════════════════
# TAB 3: Model Evaluation (Obj 2) – show saved results
# ══════════════════════════════════════════════
def show_model_evaluation():
    csv_path = os.path.join(RESULTS_DIR, "obj2_model_comparison.csv")
    if os.path.exists(csv_path):
        rdf = pd.read_csv(csv_path)
    else:
        rdf = pd.DataFrame({"Info": ["Run objective2_model_evaluation.py first"]})

    # Load saved images
    imgs = {}
    for name, fname in [
        ("Confusion Matrix", "obj2_confusion_matrix.png"),
        ("ROC Curves", "obj2_roc_curves.png"),
        ("Algorithm Comparison", "obj2_algorithm_comparison.png"),
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        imgs[name] = path if os.path.exists(path) else None

    return rdf, imgs.get("Confusion Matrix"), imgs.get("ROC Curves"), imgs.get("Algorithm Comparison")


# ══════════════════════════════════════════════
# TAB 4: Search & Recommend (Obj 3)
# ══════════════════════════════════════════════
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extractive_summary(abstract, top_n=3):
    sentences = split_sentences(str(abstract))
    if len(sentences) <= top_n:
        return str(abstract).strip()
    all_texts = [str(abstract)] + sentences
    embs = sbert.encode(all_texts)
    sims = cosine_similarity(embs[1:], embs[0].reshape(1, -1)).flatten()
    top_idx = sorted(np.argsort(sims)[-top_n:])
    return " ".join([sentences[i] for i in top_idx])


def search_papers(query, top_k=5):
    if not query.strip():
        return pd.DataFrame()
    top_k = int(top_k)
    query_emb = sbert.encode([query]).reshape(1, -1)
    sims = cosine_similarity(query_emb, all_sbert_emb).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]

    rows = []
    for rank, i in enumerate(top_idx, 1):
        row = df.iloc[i]
        summary = extractive_summary(row["abstract"])
        rows.append({
            "Rank": rank,
            "Title": row["title"],
            "Category": row["category"],
            "Similarity": round(float(sims[i]), 4),
            "Summary": summary,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# TAB 5: Visualizations (UMAP + topic dist)
# ══════════════════════════════════════════════
umap_cache = {}

def generate_umap_plot():
    if "fig" in umap_cache:
        return umap_cache["fig"]
    try:
        import umap
        print("Computing UMAP projection ...")
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        emb_2d = reducer.fit_transform(all_sbert_emb)

        # Colour by top categories
        cat_counts = df["category"].value_counts()
        top10 = cat_counts.head(10).index.tolist()
        colours = []
        cmap = cm.get_cmap("tab10", 10)
        for cat in df["category"]:
            if cat in top10:
                colours.append(cmap(top10.index(cat)))
            else:
                colours.append((0.8, 0.8, 0.8, 0.4))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colours, s=8, alpha=0.6)
        for i, cat in enumerate(top10):
            ax.scatter([], [], c=[cmap(i)], label=cat, s=30)
        ax.legend(fontsize=7, loc="upper right", markerscale=2)
        ax.set_title("UMAP Projection of Paper Embeddings (SBERT)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        umap_cache["fig"] = fig
        return fig
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "umap-learn not installed", ha="center")
        return fig


def generate_topic_distribution():
    count_all = count.transform(texts_proc)
    topic_dist = lda.transform(count_all)
    mean_dist = topic_dist.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(LDA_TOPICS), mean_dist, color="#7C4DFF")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Average Weight")
    ax.set_title("LDA Topic Distribution Across All Papers")
    ax.set_xticks(range(LDA_TOPICS))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def generate_lda_top_words():
    feature_names = count.get_feature_names_out()
    rows = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        rows.append({"Topic": f"Topic {idx}", "Top Words": ", ".join(top_words)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# BUILD GRADIO UI
# ══════════════════════════════════════════════
print("Building dashboard ...")

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Research Paper Classification & Recommendation",
) as demo:
    gr.Markdown(
        "# 🔬 ScholarAssist- Research Paper Intelligence System\n"
        "Automated NLP-based system for classifying and recommending arXiv papers."
    )



    # ── Tab 0: Classify New Paper (Deployment) ──
    with gr.Tab("🚀 Classify New Paper"):
        gr.Markdown("### Live Classification Inference\n"
                    "Input a new paper's title and abstract to predict its category using the trained SVM model.")
        with gr.Row():
            with gr.Column(scale=2):
                inf_title = gr.Textbox(label="Title", placeholder="e.g. Attention Is All You Need")
                inf_abstract = gr.Textbox(label="Abstract", lines=6, placeholder="Paste abstract here ...")
                with gr.Row():
                    inf_model = gr.Dropdown(
                        ["SVM (LinearSVC)", "Logistic Regression", "Random Forest", "Naïve Bayes"],
                        label="Classification Model", value="SVM (LinearSVC)"
                    )
                    inf_feat = gr.Dropdown(
                        ["TF-IDF", "Count Vectorizer", "Sentence-BERT", "Mirror-BERT", "LDA"],
                        label="Feature Method", value="TF-IDF"
                    )
                inf_btn = gr.Button("Classify Paper", variant="primary")
            with gr.Column(scale=1):
                inf_result = gr.Markdown("### Prediction will appear here")
                inf_table = gr.Dataframe(label="Top Predictions")
        inf_btn.click(classify_paper, inputs=[inf_title, inf_abstract, inf_model, inf_feat], outputs=[inf_result, inf_table])

    # ── Tab 1: Feature Comparison ──
    with gr.Tab("📊 Feature Comparison "):
        gr.Markdown("### Compare NLP feature extraction methods\n"
                     "Trains Logistic Regression on each representation and reports metrics.")
        btn1 = gr.Button("▶ Run Feature Comparison", variant="primary")
        tbl1 = gr.Dataframe(label="Results")
        plt1 = gr.Plot(label="Comparison Chart")
        btn1.click(run_feature_comparison, outputs=[tbl1, plt1])

    # ── Tab 2: Feature Extraction ──
    with gr.Tab("🧬 Feature Extraction"):
        gr.Markdown("### Extract features from a single paper")
        with gr.Row():
            inp_title = gr.Textbox(label="Paper Title", placeholder="e.g. Deep Learning for NLP")
            inp_method = gr.Dropdown(
                ["TF-IDF", "Count Vectorizer", "Sentence-BERT", "Mirror-BERT"],
                label="Method", value="TF-IDF"
            )
            with gr.Column():
                inp_model = gr.Dropdown(
                    ["SVM (LinearSVC)", "Logistic Regression", "Random Forest", "Naïve Bayes"],
                    label="Classification Model", value="SVM (LinearSVC)"
                )
                inp_feat = gr.Dropdown(
                    ["TF-IDF", "Count Vectorizer", "Sentence-BERT", "Mirror-BERT", "LDA"],
                    label="Feature Method (Classification)", value="TF-IDF"
                )
        inp_abstract = gr.Textbox(label="Abstract", lines=5,
                                   placeholder="Paste the paper abstract here ...")
        with gr.Row():
            btn_ext = gr.Button("Extract Features & Analyze", variant="primary")
            btn_quick_cls = gr.Button("Quick Classify (SVM)", variant="secondary")
            
        with gr.Row():
            out_ext = gr.Textbox(label="Numerical Feature Vector", lines=2)
            out_explain = gr.Markdown(label="Generalization Analysis (AI Powered)")
            
        with gr.Accordion("Quick Classify Result", open=False) as quick_acc:
            out_q_msg = gr.Markdown("Prediction will appear here...")
            out_q_tbl = gr.Dataframe(label="Top Predictions")
            
        btn_ext.click(extract_features, inputs=[inp_title, inp_abstract, inp_method], outputs=[out_ext, out_explain])
        btn_quick_cls.click(lambda: gr.update(open=True), outputs=quick_acc)
        btn_quick_cls.click(classify_paper, inputs=[inp_title, inp_abstract, inp_model, inp_feat], outputs=[out_q_msg, out_q_tbl])


    # ── Tab 3: Model Evaluation ──
    with gr.Tab("🏆 Model Evaluation "):
        gr.Markdown("### Multi-label classification benchmark\n"
                     "Shows results from `objective2_model_evaluation.py`. Run that script first.")
        btn2 = gr.Button("📄 Load Results", variant="primary")
        tbl2 = gr.Dataframe(label="Algorithm Comparison")
        with gr.Row():
            img_cm  = gr.Image(label="Confusion Matrix", type="filepath")
            img_roc = gr.Image(label="ROC Curves", type="filepath")
        img_comp = gr.Image(label="Algorithm Comparison Chart", type="filepath")
        btn2.click(show_model_evaluation, outputs=[tbl2, img_cm, img_roc, img_comp])

    # ── Tab 4: Search & Recommend ──
    with gr.Tab("🔍 Search & Recommend "):
        gr.Markdown("### Find relevant papers with extractive summaries\n"
                     "Enter a query and get top-K recommended papers.")
        with gr.Row():
            inp_query = gr.Textbox(label="Search Query", placeholder="e.g. reinforcement learning robotics",
                                    scale=4)
            inp_topk  = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        btn3 = gr.Button("🔎 Search", variant="primary")
        tbl3 = gr.Dataframe(label="Recommended Papers", wrap=True)
        btn3.click(search_papers, inputs=[inp_query, inp_topk], outputs=tbl3)

    # ── Tab 5: Visualizations ──
    with gr.Tab("📈 Visualizations"):
        gr.Markdown("### Embedding projections and topic distributions")

        with gr.Accordion("UMAP Embeddings Projection", open=True):
            btn_umap = gr.Button("Generate UMAP Plot (may take ~30s)", variant="secondary")
            plt_umap = gr.Plot(label="UMAP")
            btn_umap.click(generate_umap_plot, outputs=plt_umap)

        with gr.Accordion("LDA Topic Distribution", open=True):
            btn_topic = gr.Button("Show Topic Distribution", variant="secondary")
            plt_topic = gr.Plot(label="Topic Distribution")
            tbl_words = gr.Dataframe(label="Top Words per Topic")
            btn_topic.click(generate_topic_distribution, outputs=plt_topic)
            btn_topic.click(generate_lda_top_words, outputs=tbl_words)

    # ── Tab 6: Generate Summary ──
    with gr.Tab("📝 Generate Summary"):
        gr.Markdown("### Extractive Summarization\n"
                     "Paste the abstract or text of a research paper to generate a concise summary.")
        with gr.Row():
            with gr.Column():
                sum_input = gr.Textbox(label="Input Text (Abstract/Paper)", lines=10, placeholder="Paste text here...")
                sum_top_n = gr.Slider(1, 10, value=3, step=1, label="Number of Sentences")
                sum_btn = gr.Button("Generate Summary", variant="primary")
            with gr.Column():
                sum_output = gr.Textbox(label="Generated Summary", lines=10)
                
        def generate_paper_summary_wrapper(text, n):
            if not text or not str(text).strip():
                return "Please enter text to summarize."
            return extractive_summary(text, int(n))
            
        sum_btn.click(generate_paper_summary_wrapper, inputs=[sum_input, sum_top_n], outputs=sum_output)

    # ── Tab 7: AI Paper Analysis (Multi-Agent) ──
    with gr.Tab("🤖 AI Paper Analysis"):
        gr.Markdown(
            "### Multi-Agent Research Paper Audit\n"
            "Upload a **PDF** or paste the paper text below. "
            "Four specialized AI agents (Research Analyst, Methodology Reviewer, Citation Analyst, Trend Intelligence) "
            "analyze the paper **in parallel** via Groq (Llama 3.3) and produce a synthesized Executive Audit report."
        )

        if not agent_pipeline_ready:
            gr.Markdown(
                "> ⚠️ **Multi-agent pipeline is not available.** "
                "Make sure `langchain-groq`, `langgraph`, `pinecone`, `PyPDF2`, and `langchain-huggingface` are installed, "
                "and that valid API keys are set in your `.env` file."
            )
        else:
            with gr.Row():
                with gr.Column(scale=2):
                    ai_pdf = gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"])
                    ai_text = gr.Textbox(
                        label="Or Paste Paper Text Directly",
                        lines=8,
                        placeholder="Paste the full text of a research paper here (title + abstract + body) ...\n\nIf you upload a PDF above, this field is ignored."
                    )
                with gr.Column(scale=1):
                    ai_index = gr.Textbox(
                        label="Pinecone Index Name",
                        value=os.getenv("PINECONE_INDEX_NAME", "research-scholar"),
                        info="The Pinecone index to store/retrieve paper chunks."
                    )
                    ai_btn = gr.Button("🚀 Run Multi-Agent Analysis", variant="primary")
                    ai_status = gr.Markdown("Status: Ready")

            with gr.Accordion("🏁 Executive Audit", open=True):
                ai_report = gr.Markdown("*The synthesized report will appear here after analysis.*")

            with gr.Accordion("🔬 Detailed Specialist Reports", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📊 Research Analyst")
                        ai_analyst = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("#### 🔍 Methodology Reviewer")
                        ai_method = gr.Markdown("")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📚 Citation Analyst")
                        ai_citation = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("#### 🚀 Trend Intelligence (2026)")
                        ai_trend = gr.Markdown("")

            def run_ai_analysis(pdf_file, raw_text, index_name):
                """Run the full multi-agent analysis pipeline."""
                try:
                    # Step 1: Get the paper text
                    if pdf_file is not None:
                        paper_text = extract_text_from_pdf(pdf_file.name)
                    elif raw_text and raw_text.strip():
                        paper_text = raw_text.strip()
                    else:
                        return (
                            "⚠️ Status: No input provided",
                            "*Please upload a PDF or paste text.*",
                            "", "", "", ""
                        )

                    if len(paper_text) < 100:
                        return (
                            "⚠️ Status: Text too short",
                            "*The extracted text is too short for meaningful analysis. Please provide a longer paper.*",
                            "", "", "", ""
                        )

                    # Step 2: Chunk and upload to Pinecone
                    status_msg = "🔄 Processing: Chunking text and uploading to Pinecone..."
                    chunks = split_text(paper_text)
                    upload_to_pinecone(chunks, index_name)

                    # Step 3: Retrieve context from Pinecone
                    context = get_paper_context(index_name)

                    # Step 4: Run multi-agent graph
                    initial_state = {
                        "paper_context": context,
                        "research_summary": [],
                        "methodology_critique": [],
                        "citation_audit": [],
                        "trend_analysis": [],
                        "final_compiled_report": ""
                    }

                    final_output = graph_app.invoke(initial_state)

                    # Step 5: Extract results
                    executive_report = final_output.get("final_compiled_report", "No report generated.")
                    analyst = "\n".join(final_output.get("research_summary", ["No data."]))
                    methodology = "\n".join(final_output.get("methodology_critique", ["No data."]))
                    citations = "\n".join(final_output.get("citation_audit", ["No data."]))
                    trends = "\n".join(final_output.get("trend_analysis", ["No data."]))

                    return (
                        "✅ Status: Analysis Complete!",
                        executive_report,
                        analyst,
                        methodology,
                        citations,
                        trends
                    )
                except Exception as e:
                    return (
                        f"❌ Status: Error – {str(e)}",
                        f"*An error occurred during analysis:*\n\n```\n{str(e)}\n```",
                        "", "", "", ""
                    )

            ai_btn.click(
                run_ai_analysis,
                inputs=[ai_pdf, ai_text, ai_index],
                outputs=[ai_status, ai_report, ai_analyst, ai_method, ai_citation, ai_trend]
            )

print("Launching dashboard ...")
demo.launch(share=False)
