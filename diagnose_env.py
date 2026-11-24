import sys
import os

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name.replace("-", "_")
    try:
        __import__(import_name)
        print(f"✅ {name} is installed and importable.")
        return True
    except ImportError as e:
        print(f"❌ {name} is NOT installed or fails to import: {e}")
        return False

print("--- Checking Dependencies for AI Paper Analysis ---")
packages = [
    ("langchain-groq", "langchain_groq"),
    ("langgraph", "langgraph"),
    ("pinecone-client", "pinecone"),
    ("PyPDF2", "PyPDF2"),
    ("langchain-huggingface", "langchain_huggingface"),
    ("sentence-transformers", "sentence_transformers"),
]

all_ok = True
for pkg, imp in packages:
    if not check_package(pkg, imp):
        all_ok = False

print("\n--- Checking API Keys ---")
from dotenv import load_dotenv
load_dotenv()

keys = ["GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
for key in keys:
    val = os.getenv(key)
    if val:
        # Masked key
        masked = val[:5] + "..." + val[-5:] if len(val) > 10 else "***"
        print(f"✅ {key} is set: {masked}")
    else:
        print(f"❌ {key} is MISSING from .env")
        all_ok = False

if all_ok:
    print("\n🚀 Environment seems ready! Try running 'python dashboard.py' again.")
else:
    print("\n⚠️ Please fix the missing dependencies or API keys above.")
