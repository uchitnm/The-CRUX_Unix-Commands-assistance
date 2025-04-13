import os
import pandas as pd
import numpy as np
import faiss
import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from agent_system import AgentSystem

# Data loading functions
def load_command_data(file_path="linux_commands_tokenized.csv"):
    """Load command data from CSV file."""
    print(f"Loading command data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} commands")
    return df

def prepare_chunks(df, chunk_size=200, overlap=50):
    """
    Prepare chunks of the command data for more precise retrieval.
    """
    print(f"Preparing chunks with size={chunk_size}, overlap={overlap}...")
    chunks = []
    chunk_metadata = []
    
    for idx, row in df.iterrows():
        command = row['Command']
        description = str(row.get('DESCRIPTION', ''))
        examples = str(row.get('EXAMPLES', ''))
        options = str(row.get('OPTIONS', ''))
        
        # Combine all text for this command
        full_text = f"Command: {command}\nDescription: {description}\nExamples: {examples}\nOptions: {options}"
        
        # Create chunks with overlap
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk_text = full_text[i:i + chunk_size]
            if len(chunk_text) < 50:  # Skip very small chunks
                continue
                
            chunks.append(chunk_text)
            chunk_metadata.append({
                'original_idx': idx,
                'command': command,
                'chunk_idx': len(chunks) - 1,
                'text': chunk_text
            })
    
    chunks_df = pd.DataFrame(chunk_metadata)
    print(f"Created {len(chunks)} chunks")
    return chunks_df, chunks

def create_embedding_index(texts, model_name="all-MiniLM-L6-v2"):
    """
    Create FAISS index from text embeddings.
    """
    print(f"Creating embeddings with model {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"Created embeddings with dimension {dimension}")
    return index, embedding_model

def load_faiss_index(index_path="faiss_index.bin", metadata_path="faiss_index_metadata.csv", model_name="all-MiniLM-L6-v2"):
    """
    Load FAISS index from file or create new if it doesn't exist.
    """
    embedding_model = SentenceTransformer(model_name)
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print(f"Loading existing FAISS index from {index_path}...")
        # Load FAISS index
        index = faiss.read_index(index_path)
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, embedding_model, metadata_df
    else:
        print("FAISS index files not found. Please run indexing first.")
        return None, embedding_model, None

def process_query(user_query):
    """
    Process a query using the agent system with query optimization.
    """
    # Load data
    df = load_command_data()
    
    # Try to load existing FAISS index
    index, embedding_model, chunk_metadata = load_faiss_index()
    
    # If no index is loaded, create one
    if index is None:
        # Prepare chunked data for more precise retrieval
        chunk_metadata, chunks = prepare_chunks(df)
        
        # Create embeddings and index
        index, embedding_model = create_embedding_index(chunks)
        
        # Save metadata for future use
        chunk_metadata.to_csv("faiss_index_metadata.csv", index=False)
        faiss.write_index(index, "faiss_index.bin")
    
    # Initialize agent system
    agent_system = AgentSystem()
    
    # Step 1: Analyze and optimize the query
    print("Analyzing and optimizing query...")
    analysis = agent_system.query_analyzer_agent(
        user_query, 
        df=df,
        index=index, 
        embedding_model=embedding_model, 
        chunk_metadata=chunk_metadata
    )
    
    # Log optimization if it occurred
    if 'original_query' in analysis and analysis['original_query'] != analysis.get('reformulated_query'):
        print(f"Query optimized from: {analysis['original_query']}")
        print(f"To: {analysis.get('reformulated_query', user_query)}")
    
    # Step 2: Retrieve relevant commands with chunk support
    print("Retrieving relevant commands...")
    retrieved_commands, context = agent_system.retrieval_agent(
        df, 
        analysis.get('reformulated_query', user_query),
        analysis, 
        index, 
        embedding_model, 
        chunk_metadata
    )
    
    if not retrieved_commands:
        return "No relevant commands found. Try rephrasing your query."
    
    # Step 3: Generate response
    print("Generating response...")
    return agent_system.response_generator_agent(user_query, analysis, context)

# Command-line interface
if __name__ == "__main__":
    print("=== UNIX Command Assistant ===")
    print("Ask about any UNIX command or task. Type 'exit' to quit.")
    
    while True:
        print("\nEnter your query:")
        user_input = input("> ")
        
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
            
        try:
            response = process_query(user_input)
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a different query.")