"""
preprocess.py – Shared NLP preprocessing utilities for the research paper system.

Provides:
  - clean_text: lowercasing, punctuation/digit removal
  - tokenize_and_lemmatize: NLTK tokenizer + stopword removal + lemmatization
  - preprocess_pipeline: applies full pipeline to a DataFrame column
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent if already present)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip newlines, remove punctuation and digits."""
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-z\s]", "", text)          # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize. Returns a joined string."""
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> lemmatize."""
    return tokenize_and_lemmatize(clean_text(text))


def preprocess_dataframe(df, text_col="abstract", new_col="processed_text"):
    """Apply preprocessing to an entire DataFrame column."""
    df = df.copy()
    df[new_col] = df[text_col].apply(preprocess_text)
    return df


if __name__ == "__main__":
    sample = "Machine Learning (ML) is a subset of AI.  It enables systems to learn from data-sets, 123!"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
    print("Full    :", preprocess_text(sample))
