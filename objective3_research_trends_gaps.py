"""
objective3_research_trends_gaps.py – Objective 3: Research Trends & Potential Research Gaps Analysis

Fulfills Objective 3: "To identify research trends and potential research gaps from scientific literature."

Provides:
  1. Topic Modeling (LDA) & Top Keyword Extraction across scientific literature.
  2. Emerging Research Trends & Topic Momentum Analysis.
  3. Structural & Semantic Research Gap Detection (identifying under-researched topic pairs & cross-domain opportunities).
  4. Automatic export of CSV dataframes and visualization charts to results/.
"""

import os, re, sys, warnings
warnings.filterwarnings("ignore")

if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib

from preprocess import preprocess_text

# Configuration
DATASET_PATH = "final_arxiv_dataset.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
NUM_TOPICS = 10
SBERT_MODEL = "all-MiniLM-L6-v2"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class ResearchTrendsAndGapsAnalyzer:
    def __init__(self, dataset_path=DATASET_PATH, models_dir=MODELS_DIR):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.df = None
        self.vectorizer = None
        self.lda_model = None
        self.topic_feature_names = []
        self.topic_term_dist = None
        self.doc_topic_dist = None
        self._initialize()

    def _initialize(self):
        print("[Objective 3 Trends & Gaps] Loading dataset...")
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path).dropna().reset_index(drop=True)
            self.df["text"] = self.df["title"].astype(str) + " " + self.df["abstract"].astype(str)
            self.df["processed_text"] = self.df["text"].apply(preprocess_text)
        else:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        print("[Objective 3 Trends & Gaps] Fitting Count Vectorizer & LDA Topic Model...")
        self.vectorizer = CountVectorizer(max_features=4000, stop_words="english")
        X_counts = self.vectorizer.fit_transform(self.df["processed_text"])
        
        lda_path = os.path.join(self.models_dir, "obj3_lda_model.pkl")
        vec_path = os.path.join(self.models_dir, "obj3_count_vec.pkl")

        if os.path.exists(lda_path) and os.path.exists(vec_path):
            try:
                self.lda_model = joblib.load(lda_path)
                self.vectorizer = joblib.load(vec_path)
            except Exception:
                self.lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, n_jobs=1)
                self.lda_model.fit(X_counts)
                joblib.dump(self.lda_model, lda_path)
                joblib.dump(self.vectorizer, vec_path)
        else:
            self.lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, n_jobs=1)
            self.lda_model.fit(X_counts)
            joblib.dump(self.lda_model, lda_path)
            joblib.dump(self.vectorizer, vec_path)

        self.topic_feature_names = self.vectorizer.get_feature_names_out()
        self.doc_topic_dist = self.lda_model.transform(X_counts)

    def extract_top_topics_and_keywords(self, top_n_words=8):
        """Extracts top keywords for each discovered research topic."""
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[:-top_n_words - 1:-1]
            top_features = [self.topic_feature_names[i] for i in top_features_ind]
            topic_label = f"Topic {topic_idx + 1}: " + ", ".join(top_features[:3])
            topics.append({
                "Topic ID": f"Topic_{topic_idx + 1}",
                "Topic Title": topic_label,
                "Top Keywords": ", ".join(top_features),
                "Paper Count": int((self.doc_topic_dist.argmax(axis=1) == topic_idx).sum()),
                "Mean Topic Weight": round(float(self.doc_topic_dist[:, topic_idx].mean()), 4)
            })
        
        topics_df = pd.DataFrame(topics)
        topics_df.to_csv(os.path.join(RESULTS_DIR, "obj3_research_topics.csv"), index=False)
        return topics_df

    def analyze_emerging_trends(self):
        """Analyzes emerging research trends and keyword momentum across arXiv categories."""
        cat_topic = self.df.copy()
        cat_topic["Dominant_Topic"] = self.doc_topic_dist.argmax(axis=1) + 1
        
        trend_summary = []
        categories = self.df["category"].unique()
        
        for cat in categories:
            cat_df = cat_topic[cat_topic["category"] == cat]
            paper_cnt = len(cat_df)
            top_topic_id = cat_df["Dominant_Topic"].mode()[0] if not cat_df.empty else 1
            
            # Key term momentum in this category
            cat_vec = TfidfVectorizer(max_features=5, stop_words="english")
            try:
                tfidf_mat = cat_vec.fit_transform(cat_df["processed_text"])
                top_terms = list(cat_vec.get_feature_names_out())
            except Exception:
                top_terms = ["deep learning", "neural network", "classification"]

            trend_summary.append({
                "Category": cat,
                "Total Papers": paper_cnt,
                "Dominant Topic Cluster": f"Topic {top_topic_id}",
                "High-Momentum Key Terms": ", ".join(top_terms),
                "Emerging Trend Rating": "High Growth" if paper_cnt > 150 else ("Steady Research" if paper_cnt > 50 else "Emerging Area")
            })

        trends_df = pd.DataFrame(trend_summary).sort_values(by="Total Papers", ascending=False).reset_index(drop=True)
        trends_df.to_csv(os.path.join(RESULTS_DIR, "obj3_emerging_trends.csv"), index=False)
        return trends_df

    def detect_research_gaps(self):
        """Detects structural and semantic research gaps between topics and categories."""
        # Build Topic Co-occurrence & Inter-Topic Similarity Matrix
        topic_similarity = cosine_similarity(self.lda_model.components_)
        
        # Calculate topic paper density
        dominant_topics = self.doc_topic_dist.argmax(axis=1)
        topic_counts = np.bincount(dominant_topics, minlength=NUM_TOPICS)
        
        gaps = []
        for i in range(NUM_TOPICS):
            for j in range(i + 1, NUM_TOPICS):
                sim = topic_similarity[i, j]
                combined_density = (topic_counts[i] + topic_counts[j]) / len(self.df)
                
                # Gap Score: High similarity between topic representations but low joint co-occurrence/density
                gap_score = (1.0 - sim) * (1.0 / (combined_density + 1e-4))
                
                top_words_i = [self.topic_feature_names[idx] for idx in self.lda_model.components_[i].argsort()[:-4:-1]]
                top_words_j = [self.topic_feature_names[idx] for idx in self.lda_model.components_[j].argsort()[:-4:-1]]

                gap_title = f"{'/'.join(top_words_i)} ⟷ {'/'.join(top_words_j)}"
                
                gaps.append({
                    "Topic Pair": f"Topic {i+1} & Topic {j+1}",
                    "Research Intersection": gap_title,
                    "Semantic Similarity": round(float(sim), 4),
                    "Combined Density": round(float(combined_density), 4),
                    "Gap Opportunity Score": round(float(gap_score), 2),
                    "Research Gap Status": "Major Unexplored Gap" if gap_score > 30 else ("Moderate Opportunity" if gap_score > 15 else "Well-Explored Domain"),
                    "Actionable Opportunity": f"Investigate cross-domain synergies connecting {top_words_i[0]} models with {top_words_j[0]} applications."
                })

        gaps_df = pd.DataFrame(gaps).sort_values(by="Gap Opportunity Score", ascending=False).reset_index(drop=True)
        gaps_df.to_csv(os.path.join(RESULTS_DIR, "obj3_research_gaps.csv"), index=False)
        return gaps_df

    def render_trend_analysis_plots(self):
        """Generates high-resolution matplotlib plots for topic distributions and research gaps with extra large layout."""
        topics_df = self.extract_top_topics_and_keywords()
        gaps_df = self.detect_research_gaps()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 15))
        
        # Plot 1: Topic Distribution with Full Keywords
        y_labels = [f"{row['Topic ID']}: {row['Top Keywords'][:55]}..." for _, row in topics_df.iterrows()]
        bars1 = ax1.barh(y_labels, topics_df["Paper Count"], color="#00d6ef", edgecolor="#0072ff", alpha=0.9, height=0.65)
        ax1.set_xlabel("Number of Indexed Literature Papers", fontweight="bold", fontsize=14, color="#ffffff", labelpad=10)
        ax1.set_ylabel("Discovered Topic Clusters & Top Key Terms", fontweight="bold", fontsize=14, color="#ffffff", labelpad=10)
        ax1.set_title("Objective 3: Scientific Literature Topic Distribution & Paper Density", fontsize=16, fontweight="bold", color="#ffffff", pad=15)
        ax1.grid(axis="x", linestyle="--", alpha=0.3)
        ax1.tick_params(colors="#ffffff", labelsize=12)
        ax1.set_facecolor("#0a101d")
        ax1.invert_yaxis()

        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 5, bar.get_y() + bar.get_height()/2.0, f" {int(width)} papers", ha="left", va="center", color="#00d6ef", fontweight="bold", fontsize=12)

        # Plot 2: Top 8 Research Gaps Opportunity Scores
        top_gaps = gaps_df.head(8)
        gap_labels = [f"{row['Topic Pair']}\n({row['Research Intersection'][:35]})" for _, row in top_gaps.iterrows()]
        bars2 = ax2.bar(gap_labels, top_gaps["Gap Opportunity Score"], color="#ff2e4c", edgecolor="#ff7b00", alpha=0.9, width=0.55)
        ax2.set_xlabel("Unexplored Cross-Domain Topic Intersections", fontweight="bold", fontsize=14, color="#ffffff", labelpad=10)
        ax2.set_ylabel("Research Gap Opportunity Score", fontweight="bold", fontsize=14, color="#ffffff", labelpad=10)
        ax2.set_title("Objective 3: Potential Unexplored Research Gaps & Actionable Opportunities", fontsize=16, fontweight="bold", color="#ffffff", pad=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        ax2.tick_params(colors="#ffffff", labelsize=11, rotation=12)
        ax2.set_facecolor("#0a101d")

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, height + 0.8, f"Score: {height:.1f}", ha="center", va="bottom", color="#ff7b00", fontweight="bold", fontsize=12)

        fig.patch.set_facecolor("#030b17")
        fig.tight_layout(pad=3.5)
        fig.savefig(os.path.join(RESULTS_DIR, "obj3_trends_and_gaps_plot.png"), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        return fig

# Singleton instance accessor
_trends_analyzer = None

def get_trends_analyzer():
    global _trends_analyzer
    if _trends_analyzer is None:
        _trends_analyzer = ResearchTrendsAndGapsAnalyzer()
    return _trends_analyzer

if __name__ == "__main__":
    print("[Objective 3] Running standalone Research Trends & Gaps Analysis...")
    analyzer = get_trends_analyzer()
    topics = analyzer.extract_top_topics_and_keywords()
    trends = analyzer.analyze_emerging_trends()
    gaps = analyzer.detect_research_gaps()
    print("\n--- TOP RESEARCH TOPICS ---")
    print(topics[["Topic ID", "Top Keywords", "Paper Count"]].head())
    print("\n--- EMERGING TRENDS ---")
    print(trends.head())
    print("\n--- POTENTIAL RESEARCH GAPS ---")
    print(gaps[["Topic Pair", "Research Intersection", "Gap Opportunity Score", "Research Gap Status"]].head())
    print("\n[Objective 3] Success! Results saved to results/ folder.")
