"""
objective2_model_evaluation.py – Objective 2: Model Evaluation with Standard Metrics

Benchmarks Logistic Regression, SVM, Random Forest, and Naïve Bayes
using OneVsRestClassifier on TF-IDF features. Performs GridSearchCV with
10-fold CV, then evaluates on held-out test set. Generates confusion
matrix, ROC curves, and algorithm comparison charts.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
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
CV_FOLDS      = 10

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# LOAD & PREPROCESS
# ──────────────────────────────────────────────
print("[1/5] Loading dataset ...")
df = pd.read_csv(DATASET).dropna()
df["text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)
df["processed_text"] = df["text"].apply(preprocess_text)

# Keep top categories that have enough samples for meaningful evaluation
cat_counts = df["category"].value_counts()
top_cats = cat_counts[cat_counts >= 10].index.tolist()
df_filtered = df[df["category"].isin(top_cats)].copy()
print(f"   Using {len(top_cats)} categories with >=10 samples -> {len(df_filtered)} papers")

texts = df_filtered["processed_text"].values
labels = df_filtered["category"].values

# Binarize labels for OneVsRest
lb = LabelBinarizer()
Y = lb.fit_transform(labels)
classes = lb.classes_
joblib.dump(lb, os.path.join(MODELS_DIR, "label_binarizer.pkl"))
print(f"   Classes: {list(classes)}")

# ──────────────────────────────────────────────
# TF-IDF FEATURES
# ──────────────────────────────────────────────
print("[2/5] Vectorizing with TF-IDF ...")
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
X = tfidf.fit_transform(texts)
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_obj2.pkl"))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ──────────────────────────────────────────────
# DEFINE ALGORITHMS & PARAM GRIDS
# ──────────────────────────────────────────────
algorithms = {
    "Logistic Regression": {
        "estimator": OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "params": {"estimator__C": [0.1, 1, 10]},
    },
    "SVM (LinearSVC)": {
        "estimator": OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE), cv=3)
        ),
        "params": {"estimator__estimator__C": [0.1, 1, 10]},
    },
    "Random Forest": {
        "estimator": OneVsRestClassifier(RandomForestClassifier(random_state=RANDOM_STATE)),
        "params": {"estimator__n_estimators": [100, 200], "estimator__max_depth": [None, 20]},
    },
    "Naïve Bayes": {
        "estimator": OneVsRestClassifier(MultinomialNB()),
        "params": {"estimator__alpha": [0.1, 0.5, 1.0]},
    },
}

# ──────────────────────────────────────────────
# TRAIN, TUNE & EVALUATE
# ──────────────────────────────────────────────
print("[3/5] Training and tuning models (10-fold CV) ...")
all_results = []
best_models = {}

for name, cfg in algorithms.items():
    print(f"\n   > {name}")
    gs = GridSearchCV(
        cfg["estimator"], cfg["params"],
        cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, Y_train)
    best = gs.best_estimator_
    best_models[name] = best
    print(f"     Best params: {gs.best_params_}")

    Y_pred = best.predict(X_test)

    # Compute ROC-AUC (needs predict_proba or decision_function)
    try:
        Y_score = best.predict_proba(X_test)
        roc = roc_auc_score(Y_test, Y_score, average="weighted", multi_class="ovr")
    except Exception:
        try:
            Y_score = best.decision_function(X_test)
            roc = roc_auc_score(Y_test, Y_score, average="weighted", multi_class="ovr")
        except Exception:
            roc = float("nan")
            Y_score = None

    acc  = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average="weighted", zero_division=0)
    rec  = recall_score(Y_test, Y_pred, average="weighted", zero_division=0)
    f1   = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)
    print(f"     Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  ROC-AUC={roc:.4f}")

    all_results.append({
        "Algorithm": name, "Accuracy": acc,
        "Precision": prec, "Recall": rec,
        "F1 Score": f1, "ROC-AUC": roc,
    })

    # Save model
    joblib.dump(best, os.path.join(MODELS_DIR, f"obj2_{name.replace(' ', '_').lower()}.pkl"))

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "obj2_model_comparison.csv"), index=False)
print("\n" + results_df.to_string(index=False))

# ──────────────────────────────────────────────
# VISUALISATION 1 : Confusion Matrix (best model)
# ──────────────────────────────────────────────
print("\n[4/5] Generating visualizations ...")

# Pick best model by F1
best_name = results_df.loc[results_df["F1 Score"].idxmax(), "Algorithm"]
best_clf  = best_models[best_name]
Y_pred_best = best_clf.predict(X_test)

# Multi-label -> single label for confusion matrix
y_test_lbl = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test
y_pred_lbl = np.argmax(Y_pred_best, axis=1) if Y_pred_best.ndim > 1 else Y_pred_best

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
cm_matrix = confusion_matrix(y_test_lbl, y_pred_lbl)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=classes)
disp.plot(ax=ax_cm, cmap="Blues", values_format="d", xticks_rotation=45)
ax_cm.set_title(f"Confusion Matrix – {best_name}")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(RESULTS_DIR, "obj2_confusion_matrix.png"), dpi=150)
print(f"   Confusion matrix saved")

# ──────────────────────────────────────────────
# VISUALISATION 2 : ROC Curves
# ──────────────────────────────────────────────
try:
    Y_score_best = best_clf.predict_proba(X_test)
except Exception:
    try:
        Y_score_best = best_clf.decision_function(X_test)
    except Exception:
        Y_score_best = None

if Y_score_best is not None:
    n_classes = Y_test.shape[1] if Y_test.ndim > 1 else len(classes)
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    cmap_colors = cm.get_cmap("tab10", n_classes)

    for i in range(min(n_classes, 10)):
        y_true_i = Y_test[:, i] if Y_test.ndim > 1 else (Y_test == i).astype(int)
        y_score_i = Y_score_best[:, i] if Y_score_best.ndim > 1 else Y_score_best
        fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
        auc_val = roc_auc_score(y_true_i, y_score_i) if y_true_i.sum() > 0 else 0
        ax_roc.plot(fpr, tpr, color=cmap_colors(i), label=f"{classes[i]} (AUC={auc_val:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves – {best_name}")
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(RESULTS_DIR, "obj2_roc_curves.png"), dpi=150)
    print("   ROC curves saved")

# ──────────────────────────────────────────────
# VISUALISATION 3 : Algorithm Comparison
# ──────────────────────────────────────────────
fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.15
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = results_df[metric].fillna(0).values
    ax_comp.bar(x + i * width, vals, width, label=metric, color=color)

ax_comp.set_xlabel("Algorithm")
ax_comp.set_ylabel("Score")
ax_comp.set_title("Objective 2 – Algorithm Performance Comparison")
ax_comp.set_xticks(x + width * 2)
ax_comp.set_xticklabels(results_df["Algorithm"], rotation=15, ha="right")
ax_comp.set_ylim(0, 1.05)
ax_comp.legend()
ax_comp.grid(axis="y", alpha=0.3)
fig_comp.tight_layout()
fig_comp.savefig(os.path.join(RESULTS_DIR, "obj2_algorithm_comparison.png"), dpi=150)
print("   Algorithm comparison chart saved")

print("\n[5/5] All done OK")
print(f"Results saved in: {RESULTS_DIR}/")
print(f"Models  saved in: {MODELS_DIR}/")
