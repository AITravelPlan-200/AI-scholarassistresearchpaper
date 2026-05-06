import streamlit as st
import os
import tempfile
from src.utils import upload_to_pinecone, extract_text_from_pdf, split_text
from src.graph import app as graph_app

st.set_page_config(page_title="ResearchScholar_AI", page_icon="🎓", layout="wide")

st.title("🎓 ResearchScholar_AI")
st.markdown("### Multi-Agent Parallel Research Audit")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    index_name = st.text_input("Pinecone Index", value="research-scholar")
    if st.button("Clear Session"):
        st.rerun()

# --- File Upload Section ---
uploaded_file = st.file_uploader("Drag and drop your research paper (PDF)", type="pdf")

if uploaded_file is not None:
    # 1. Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🚀 Analyze Paper"):
        with st.status("Processing Paper...", expanded=True) as status:
            # Milestone 1: Ingestion
            st.write("Reading PDF and Vectorizing...")
            text = extract_text_from_pdf(tmp_path)
            chunks = split_text(text)
            upload_to_pinecone(chunks, index_name)
            
            # Milestone 2 & 3: Multi-Agent Analysis
            st.write("Triggering 4 Parallel Agents via Groq...")
            # We fetch context just like in main.py but for the UI
            from main import get_paper_context
            context = get_paper_context(index_name)
            
            initial_state = {
                "paper_context": context,
                "research_summary": [],
                "methodology_critique": [],
                "citation_audit": [],
                "trend_analysis": [],
                "final_compiled_report": ""
            }
            
            final_output = graph_app.invoke(initial_state)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- Display Results ---
        st.success("Analysis Finished!")
        
        # Display Executive Audit prominently
        st.subheader("🏁 Executive Audit")
        st.info(final_output['final_compiled_report'])
        
        # Display Specialist tabs
        st.subheader("🔬 Detailed Specialist Reports")
        tabs = st.tabs(["Analyst", "Methodology", "Citations", "Trends"])
        
        with tabs[0]:
            st.write(final_output['research_summary'][0])
        with tabs[1]:
            st.write(final_output['methodology_critique'][0])
        with tabs[2]:
            st.write(final_output['citation_audit'][0])
        with tabs[3]:
            st.write(final_output['trend_analysis'][0])

    # Cleanup temp file
    os.remove(tmp_path)