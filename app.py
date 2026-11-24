"""
app.py – Hugging Face Spaces & Production Entrypoint for Scholar Assist AI Dashboard

Launches the unified Scholar Assist AI research assistant platform with:
- Objective 1: Citational Knowledge Graph (GraphRAG)
- Objective 2: Context-Aware RAG Engine (with strict dataset grounding & PDF citations)
- Objective 3: Research Trends & Gaps Analysis
- Objective 4: Dynamic System Evaluation Benchmark
"""

import os
import sys

# Ensure current directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dashboard import demo, get_rag_engine

if __name__ == "__main__":
    print("Pre-initializing Context-Aware RAG Engine ...", flush=True)
    try:
        get_rag_engine()
    except Exception as e:
        print(f"[Notice] RAG Engine pre-init notice: {e}", flush=True)

    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

    print(f"Launching Scholar Assist AI Dashboard on {server_name}:{server_port} ...", flush=True)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False
    )
