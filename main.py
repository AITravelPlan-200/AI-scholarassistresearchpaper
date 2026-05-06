import os
import warnings
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from src.graph import app
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
# 1. Configuration & Silence Noise
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
load_dotenv()

def get_paper_context(index_name):
    """Retrieves relevant text chunks from the Milestone 1 Pinecone index."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Must match the 384-dimension model used in utils.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Search for the core of the research
    query_vec = embeddings.embed_query("Summarize main findings, methodology, and citations")
    results = index.query(vector=query_vec, top_k=10, include_metadata=True)
    
    return [res['metadata']['text'] for res in results['matches']]

def run_analysis():
    print("\n" + "="*60)
    print("🚀 RESEARCH SCHOLAR AI: STARTING MULTI-AGENT ANALYSIS")
    print("="*60)

    # 1. Retrieve Data from Milestone 1
    index_name = os.getenv("PINECONE_INDEX_NAME", "research-scholar")
    try:
        context = get_paper_context(index_name)
        print(f"📡 Data retrieved from Pinecone index: '{index_name}'")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return

    # 2. Initialize the Shared Whiteboard (State)
    initial_state = {
        "paper_context": context,
        "research_summary": [],
        "methodology_critique": [],
        "citation_audit": [],
        "trend_analysis": [],
        "final_compiled_report": ""
    }

    # 3. Trigger Parallel Agents via Groq (Fan-Out/Fan-In)
    print("🧠 Triggering 4 Specialized Agents in Parallel...")
    print("🤖 Synthesizing results via the Aggregator node...")
    
    try:
        final_output = app.invoke(initial_state)
    except Exception as e:
        print(f"❌ Error during Agent execution: {e}")
        return

    # 4. Print the Synthesized Executive Audit
    print("\n" + "🎓 " + "="*20 + " EXECUTIVE AUDIT " + "="*20 + " 🎓")
    if final_output.get('final_compiled_report'):
        print(final_output['final_compiled_report'])
    else:
        print("⚠️ Aggregator failed to produce a final report.")
    
    # 5. Print the Individual Specialist Reports (Appendix)
    print("\n" + "📜 " + "-"*20 + " DETAILED SPECIALIST NOTES " + "-"*20 + " 📜")
    
    sections = {
        "RESEARCH ANALYST": "research_summary",
        "METHODOLOGY REVIEWER": "methodology_critique",
        "CITATION ANALYST": "citation_audit",
        "2026 TREND INTELLIGENCE": "trend_analysis"
    }

    for title, key in sections.items():
        print(f"\n--- {title} ---")
        content = "\n".join(final_output.get(key, ["No data returned."]))
        print(content)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")

if __name__ == "__main__":
    run_analysis()