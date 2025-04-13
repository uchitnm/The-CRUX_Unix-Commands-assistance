import faiss
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path="faiss_index.bin", metadata_path="faiss_index_metadata.csv", model_name="all-MiniLM-L6-v2"):
    """
    Load FAISS index from file or create new if it doesn't exist.
    
    Args:
        index_path: Path to the FAISS index file
        metadata_path: Path to the FAISS index metadata CSV
        model_name: Name of the SentenceTransformer model
        
    Returns:
        tuple: (index, embedding_model, metadata_df)
    """
    embedding_model = SentenceTransformer(model_name)
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print(f"\033[1;33mLoading existing FAISS index from {index_path}...\033[0m")
        # Load FAISS index
        index = faiss.read_index(index_path)
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        print(f"\033[1;32mLoaded FAISS index with {index.ntotal} vectors\033[0m")
        return index, embedding_model, metadata_df
    else:
        print("\033[1;31mFAISS index files not found. Please run indexing first.\033[0m")
        return None, embedding_model, None