import os
import time
import PyPDF2
import logging
import warnings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Silence noise
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
os.environ["HF_HUB_VERBOSITY"] = "error"

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def extract_text_from_pdf(file_path):
    """Step 1: Just get the raw text."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text):
    """Step 2: Turn raw text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def upload_to_pinecone(chunks, index_name):
    """Step 3: Embed and send to Pinecone."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Ensure index exists first
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embeddings.embed_query(chunk)
        vectors.append({
            "id": f"chunk_{i}_{int(time.time())}", # Added time to avoid ID collisions
            "values": vec,
            "metadata": {"text": chunk}
        })
        
        if len(vectors) == 50 or i == len(chunks) - 1:
            index.upsert(vectors=vectors)
            vectors = []
    return True