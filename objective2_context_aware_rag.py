"""
objective2_context_aware_rag.py – Objective 2: Context-Aware RAG Engine for Conversational Literature Retrieval

Provides a Context-Aware RAG engine that:
1. Performs context-aware multi-turn conversational query reformulation.
2. Combines SBERT dense vector retrieval + Citational Knowledge Graph (GraphRAG) subgraph context.
3. Integrates LLM (Groq Llama 3.3 / Llama 3) for conversational academic synthesis or fallback RAG synthesis.
4. Generates explicit literature citation references, PageRank authority scores, and context prompt blocks.
"""

import os, re, json, sys, warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib

from preprocess import preprocess_text
from objective1_citational_graph import CitationalKnowledgeGraph, get_or_build_citational_graph

# Configuration
DATASET_PATH = "final_arxiv_dataset.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
SBERT_MODEL = "all-MiniLM-L6-v2"

from urllib.parse import quote_plus

class ContextAwareRAGEngine:
    def __init__(self, dataset_path=DATASET_PATH, models_dir=MODELS_DIR):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.df = None
        self.sbert = None
        self.embeddings = None
        self.ckg = None
        self.llm_ready = False
        self.llm_client = None
        self._initialize_resources()

    def _initialize_resources(self):
        print("[Objective 2 RAG] Initializing Context-Aware RAG Engine...")
        # Load Dataset
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path).dropna().reset_index(drop=True)
            cat_counts = self.df["category"].value_counts()
            top_cats = cat_counts[cat_counts >= 10].index.tolist()
            self.df = self.df[self.df["category"].isin(top_cats)].copy().reset_index(drop=True)
            self.df["paper_id"] = [f"P_{i+1:04d}" for i in range(len(self.df))]
            self.df["text"] = self.df["title"].astype(str) + " " + self.df["abstract"].astype(str)
        else:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # Load SBERT Model & Embeddings
        print("[Objective 2 RAG] Loading Sentence-BERT embeddings...")
        self.sbert = SentenceTransformer(SBERT_MODEL)
        emb_path = os.path.join(self.models_dir, "sbert_embeddings.npz")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)["embeddings"]
            if len(self.embeddings) != len(self.df):
                self.embeddings = self.sbert.encode(self.df["text"].tolist(), show_progress_bar=False, batch_size=64)
                np.savez_compressed(emb_path, embeddings=self.embeddings)
        else:
            self.embeddings = self.sbert.encode(self.df["text"].tolist(), show_progress_bar=False, batch_size=64)
            np.savez_compressed(emb_path, embeddings=self.embeddings)

        # Load / Build Citational Knowledge Graph (GraphRAG)
        print("[Objective 2 RAG] Loading Citational Knowledge Graph...")
        ckg_path = os.path.join(self.models_dir, "citational_knowledge_graph.pkl")
        if os.path.exists(ckg_path):
            try:
                self.ckg = joblib.load(ckg_path)
            except Exception:
                self.ckg = get_or_build_citational_graph(max_papers=2500)
                joblib.dump(self.ckg, ckg_path)
        else:
            self.ckg = get_or_build_citational_graph(max_papers=2500)
            joblib.dump(self.ckg, ckg_path)

        # Response Cache for ultra-fast instant execution
        self._chat_cache = {}

        # Check Groq LLM availability with active high-intelligence model fallback
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                for fast_model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
                    try:
                        self.llm_client = ChatGroq(
                            temperature=0.2,
                            model_name=fast_model,
                            max_tokens=1500,
                            groq_api_key=groq_api_key
                        )
                        self.llm_ready = True
                        print(f"[Objective 2 RAG] Ultra-High-Intelligence Groq LLM initialized ({fast_model}).")
                        break
                    except Exception as err:
                        print(f"  [Model Init Skip] {fast_model}: {err}")
                        continue
            except Exception as e:
                print(f"[Objective 2 RAG] Groq LLM init warning: {e}")

    def reformulate_context_query(self, user_query, conversation_history):
        """Reformulates a follow-up user query using conversation history for context awareness."""
        if not conversation_history:
            return user_query

        recent_turns = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        context_summary = []
        for turn in recent_turns:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                q, a = turn[0], turn[1]
                if q: context_summary.append(f"User asked: {q}")
                if a: context_summary.append(f"System answered: {str(a)[:80]}...")

        context_str = " | ".join(context_summary)
        reformulated = f"{user_query} (Context: {context_str})"
        return reformulated

    def retrieve(self, user_query, conversation_history=None, top_k=5, mode="Hybrid GraphRAG + Vector Search"):
        """Performs Context-Aware Hybrid Retrieval (Vector Similarity + Knowledge Graph Subgraph)."""
        top_k = int(top_k)
        
        # 1. Context-aware query reformulation
        effective_query = self.reformulate_context_query(user_query, conversation_history)
        
        # 2. Vector Similarity Search
        query_emb = self.sbert.encode([effective_query]).reshape(1, -1)
        sims = cosine_similarity(query_emb, self.embeddings).flatten()
        top_indices = np.argsort(sims)[-top_k:][::-1]

        retrieved_rows = []
        context_blocks = []

        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            pid = row["paper_id"]
            title = row["title"]
            abstract = row["abstract"]
            cat = row["category"]
            sim_score = float(sims[idx])

            pr_val = 0.0
            if hasattr(self.ckg, 'metrics_df') and self.ckg.metrics_df is not None and not self.ckg.metrics_df.empty:
                pr_match = self.ckg.metrics_df[self.ckg.metrics_df["Paper_ID"] == pid]
                if not pr_match.empty and "PageRank" in pr_match.columns:
                    pr_val = float(pr_match.iloc[0]["PageRank"])

            clean_title_query = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
            paper_url = f"https://arxiv.org/search/?query={quote_plus(clean_title_query)}&searchtype=title"

            retrieved_rows.append({
                "Rank": rank,
                "Paper ID": pid,
                "Title": title,
                "Category": cat,
                "Similarity Score": round(sim_score, 4),
                "PageRank Score": round(pr_val, 6),
                "Paper / PDF Link": paper_url,
                "Abstract Snippet": abstract[:180] + "..."
            })

            context_blocks.append(
                f"[{rank}] Paper ID: {pid} | Category: {cat} | Similarity: {sim_score:.4f} | PageRank: {pr_val:.6f}\n"
                f"    Title: {title}\n"
                f"    Paper PDF Link: {paper_url}\n"
                f"    Abstract: {abstract}\n"
            )

        # 3. GraphRAG Subgraph Retrieval
        graphrag_prompt_block = ""
        if "GraphRAG" in mode or mode == "Knowledge Graph Context Only":
            try:
                top_pid = retrieved_rows[0]["Paper ID"] if retrieved_rows else user_query
                graphrag_ctx = self.ckg.retrieve_graphrag_context(top_pid)
                graphrag_prompt_block = graphrag_ctx.get("graphrag_prompt_block", "")
            except Exception as e:
                graphrag_prompt_block = f"GraphRAG Context Notice: {e}"

        retrieved_df = pd.DataFrame(retrieved_rows)
        full_context_text = "\n".join(context_blocks)

        return {
            "effective_query": effective_query,
            "retrieved_df": retrieved_df,
            "vector_context": full_context_text,
            "graphrag_context": graphrag_prompt_block,
            "combined_context": f"### RETRIEVED LITERATURE (AVAILABLE INDEXED PAPERS ONLY):\n{full_context_text}\n\n### CITATIONAL KNOWLEDGE GRAPH CONTEXT:\n{graphrag_prompt_block}"
        }

    def chat(self, message, history, mode="Hybrid GraphRAG + Vector Search", top_k=5, temperature=0.3):
        """Generates a multi-turn conversational response with context-aware RAG literature citations."""
        if not message or not str(message).strip():
            return history, pd.DataFrame(), ""

        if history is None:
            history = []

        cache_key = (message.strip().lower(), str(mode), int(top_k), float(temperature))
        if hasattr(self, "_chat_cache") and cache_key in self._chat_cache:
            cached_full_resp, cached_df, cached_combined_ctx = self._chat_cache[cache_key]
            history.append((message, cached_full_resp))
            return history, cached_df, cached_combined_ctx

        # Step 1: Context-Aware Retrieval
        retrieval_res = self.retrieve(message, history, top_k=top_k, mode=mode)
        retrieved_df = retrieval_res["retrieved_df"]
        combined_context = retrieval_res["combined_context"]

        # Step 2: Strict In-Context Grounded Response Generation (No Google / No External ChatGPT Knowledge)
        system_prompt = (
            "You are ScholarAssist AI, a High-Precision Context-Aware Academic Literature RAG Engine. "
            "STRICT GROUNDING DIRECTIVE: You MUST answer the user's question STRICTLY AND SOLELY using the provided literature context from the available indexed research papers below. "
            "Do NOT use external web search, Google, Bing, or ungrounded general knowledge. "
            "Always cite explicit paper IDs (e.g., [P_0001], [P_0005]), paper titles, and arXiv paper/PDF links from the retrieved literature context below. "
            "If the provided indexed papers do not contain enough information for a specific point, explicitly state that based on the available indexed papers. "
            "Structure your response clearly with:\n"
            "• Executive Summary & Direct Answer (strictly based on available papers)\n"
            "• Technical & Methodological Deep Dive (citing Paper IDs & Paper/PDF links)\n"
            "• Comparative Literature Insights & Citation Evidence\n"
            "• Indexed Papers Summary & PDF/Paper References\n\n"
            f"RETRIEVED LITERATURE CONTEXT (AVAILABLE INDEXED PAPERS):\n{combined_context}"
        )

        if self.llm_ready:
            try:
                messages = [
                    ("system", system_prompt),
                ]
                for turn in history[-2:]:
                    if isinstance(turn, (list, tuple)) and len(turn) == 2:
                        if turn[0]: messages.append(("user", turn[0]))
                        if turn[1]: messages.append(("assistant", turn[1][:300]))

                messages.append(("user", message))
                llm_response = self.llm_client.invoke(messages)
                answer_text = llm_response.content
            except Exception as e:
                answer_text = self._synthesize_fallback_academic_response(message, retrieved_df)
        else:
            answer_text = self._synthesize_fallback_academic_response(message, retrieved_df)

        # Append citations block with clickable PDF/Paper links to answer
        citations_markdown = "\n\n---\n### 📖 Available Indexed Papers & PDF Links:\n"
        for idx, row in retrieved_df.iterrows():
            citations_markdown += f"- 📄 **[{row['Paper ID']}]** `{row['Category']}` — **{row['Title']}** | 🔗 [Read Paper / PDF on arXiv]({row['Paper / PDF Link']}) *(Similarity: {row['Similarity Score']}, PageRank: {row['PageRank Score']})*\n"

        full_response = answer_text + citations_markdown
        history.append((message, full_response))

        if hasattr(self, "_chat_cache"):
            self._chat_cache[cache_key] = (full_response, retrieved_df, combined_context)

        return history, retrieved_df, combined_context

    def _synthesize_fallback_academic_response(self, message, retrieved_df):
        """Synthesizes structured academic RAG response when external LLM is offline."""
        if retrieved_df.empty:
            return "No relevant literature could be retrieved for your query from the available dataset."

        top_paper = retrieved_df.iloc[0]
        categories_found = list(retrieved_df["Category"].unique())

        response = f"### 🔬 Context-Aware Academic RAG Analysis (Strictly Available Dataset Papers)\n\n"
        response += f"**Query:** *\"{message}\"*\n\n"
        response += f"Based strictly on available indexed papers across **{len(retrieved_df)} retrieved papers** in categories `{', '.join(categories_found)}` (No Google/ChatGPT external knowledge used):\n\n"
        
        response += f"#### 📌 Primary Findings ([{top_paper['Paper ID']}]):\n"
        response += f"The top matching paper is **[{top_paper['Paper ID']}] {top_paper['Title']}** (Category: `{top_paper['Category']}`, Similarity: **{top_paper['Similarity Score']}**).\n"
        response += f"🔗 **Paper/PDF Link:** [Read Paper / PDF on arXiv]({top_paper['Paper / PDF Link']})\n"
        response += f"> *{top_paper['Abstract Snippet']}*\n\n"

        response += "#### 📚 Related Literature & Cross-Category Insights:\n"
        for idx, row in retrieved_df.iloc[1:4].iterrows():
            response += f"- **[{row['Paper ID']}]** (`{row['Category']}`): **{row['Title']}** | 🔗 [Read Paper / PDF]({row['Paper / PDF Link']}) — Similarity: {row['Similarity Score']}\n"

        response += "\n#### 🕸️ GraphRAG Citation Topology Insights:\n"
        response += f"The literature graph connects these papers through structural citation hubs with maximum PageRank authority of **{retrieved_df['PageRank Score'].max()}**."

        return response


# Global singleton instance for dashboard integration
_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = ContextAwareRAGEngine()
    return _rag_engine
