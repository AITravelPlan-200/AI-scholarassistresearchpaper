import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pinecone import Pinecone

def test_environment():
    load_dotenv()
    print("--- 1. Testing Environment Variables ---")
    groq_key = os.getenv("GROQ_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if groq_key and pinecone_key:
        print("✅ API Keys loaded successfully.")
    else:
        print("❌ Missing API Keys in .env file.")

def test_groq_connection():
    print("\n--- 2. Testing Groq (Llama 3) Connection ---")
    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        response = llm.invoke("Hello, ResearchScholar_AI is online. Confirm receipt.")
        print(f"✅ Groq Response: {response.content}")
    except Exception as e:
        print(f"❌ Groq Error: {e}")

def test_pinecone_connection():
    print("\n--- 3. Testing Pinecone Connection ---")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "research-scholar-index")
        
        # List indexes to verify connection
        indexes = pc.list_indexes().names()
        print(f"✅ Pinecone Connected. Available indexes: {indexes}")
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")

if __name__ == "__main__":
    test_environment()
    test_groq_connection()
    test_pinecone_connection()