import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_csv(file_path):
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def build_and_save_faiss_index(df, column_name, index_path, model_name="all-MiniLM-L6-v2"):
    """Build a FAISS index from the given column and save it to disk."""
    descriptions = df[column_name].astype(str).fillna("").tolist()
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(descriptions, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)  # Save the FAISS index to disk
    print(f"FAISS index saved to {index_path}")

def main():
    csv_file = "/Users/uchitnm/Downloads/linux_commands_tokenized.csv"  # Adjust to your file path
    index_path = "/Users/uchitnm/Downloads/faiss_index.bin"  # Path to save FAISS index
    column_name = "DESCRIPTION"  # Column for FAISS indexing

    df = load_csv(csv_file)
    build_and_save_faiss_index(df, column_name, index_path)

if __name__ == "__main__":
    main()
