import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import ssl
import nltk
import sys

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data for sentence tokenization
print("Checking and downloading NLTK tokenizer resources...")
try:
    # Download both punkt and punkt_tab if needed
    nltk.download('punkt')
except Exception as e:
    print(f"Warning: Error downloading nltk resources: {e}")
    print("Will use a simple regex-based tokenizer as fallback.")

def load_csv(file_path):
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def chunk_text(text, chunk_size=3, overlap=1):
    """
    Split text into overlapping chunks of sentences.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Number of sentences per chunk
        overlap (int): Number of overlapping sentences between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text or not isinstance(text, str):
        return [""]
    
    # Use a simple regex-based sentence tokenizer as it's more reliable
    # across different NLTK installations
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(sentences) - chunk_size + 1, chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    
    # Handle the last chunk if there are remaining sentences
    if i + chunk_size < len(sentences):
        chunks.append(" ".join(sentences[-(chunk_size):]))
    
    return chunks

def build_and_save_faiss_index(df, column_name, index_path, chunk_size=3, overlap=1, model_name="all-MiniLM-L6-v2"):
    """Build a FAISS index from the given column with text chunking and save it to disk."""
    # Initialize lists to store chunks and their metadata
    all_chunks = []
    chunk_metadata = []
    
    print(f"Processing {len(df)} entries...")
    
    for idx, row in df.iterrows():
        text = str(row[column_name]).strip()
        if not text:
            continue
            
        # Chunk the text
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        # Store each chunk with its metadata
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(chunk)
            # Store metadata for this chunk (original row index, chunk number, and command)
            chunk_metadata.append({
                'original_idx': idx,
                'chunk_idx': i,
                'command': row.get('Command', ''),  # Assuming 'Command' column exists
                'chunk_text': chunk
            })
        
        # Print progress every 100 entries
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} entries...")
    
    print(f"Creating embeddings for {len(all_chunks)} chunks...")
    # Create embeddings for all chunks
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)
    
    print("Building FAISS index...")
    # Build and save FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    
    # Save chunk metadata
    metadata_df = pd.DataFrame(chunk_metadata)
    metadata_path = os.path.splitext(index_path)[0] + "_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"FAISS index saved to {index_path}")
    print(f"Chunk metadata saved to {metadata_path}")
    print(f"Created {len(all_chunks)} chunks from {len(df)} original entries")

def main():
    # Get file paths from command line or use defaults
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "linux_commands_tokenized.csv"
    
    if len(sys.argv) > 2:
        index_path = sys.argv[2]
    else:
        index_path = "faiss_index.bin"
        
    # Make sure CSV file exists
    if not os.path.exists(csv_file):
        csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file)
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found: {csv_file}")
            print("Please provide the correct path to the CSV file.")
            sys.exit(1)
            
    column_name = "DESCRIPTION"  # Column for FAISS indexing
    
    # Chunking parameters
    chunk_size = 3  # Number of sentences per chunk
    overlap = 1     # Overlap between chunks

    print(f"Loading CSV file: {csv_file}")
    df = load_csv(csv_file)
    print(f"Loaded {len(df)} entries")
    
    build_and_save_faiss_index(df, column_name, index_path, chunk_size, overlap)

if __name__ == "__main__":
    main()
