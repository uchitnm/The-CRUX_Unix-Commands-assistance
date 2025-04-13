from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from agent_system import AgentSystem

app = Flask(__name__)

# Global variables to store loaded resources
df = None
index = None
embedding_model = None
chunk_metadata = None
agent_system = None

def load_command_data(file_path="linux_commands_tokenized.csv"):
    """Load command data from CSV file."""
    print(f"Loading command data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} commands")
    return df

def load_faiss_index(index_path="faiss_index.bin", metadata_path="faiss_index_metadata.csv", model_name="all-MiniLM-L6-v2"):
    """Load FAISS index from file or create new if it doesn't exist."""
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
        print("FAISS index files not found.")
        return None, embedding_model, None

def initialize_resources():
    """Initialize all resources needed for the application."""
    global df, index, embedding_model, chunk_metadata, agent_system
    
    # Load data
    df = load_command_data()
    
    # Try to load existing FAISS index
    index, embedding_model, chunk_metadata = load_faiss_index()
    
    if index is None:
        return False
    
    # Initialize agent system
    agent_system = AgentSystem()
    
    return True

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint to process user queries."""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Step 1: Analyze and optimize the query
        analysis = agent_system.query_analyzer_agent(
            user_query, 
            df=df,
            index=index, 
            embedding_model=embedding_model, 
            chunk_metadata=chunk_metadata
        )
        
        # Step 2: Retrieve relevant commands
        retrieved_commands, context = agent_system.retrieval_agent(
            df, 
            analysis.get('reformulated_query', user_query),
            analysis, 
            index, 
            embedding_model, 
            chunk_metadata
        )
        
        if not retrieved_commands:
            return jsonify({'response': 'No relevant commands found. Try rephrasing your query.'})
        
        # Step 3: Generate response
        response = agent_system.response_generator_agent(user_query, analysis, context)
        
        # Return the response
        return jsonify({
            'response': response,
            'analysis': analysis,
            'commands': [cmd.get('Command', '') for cmd in retrieved_commands]
        })
    
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if initialize_resources():
        print("Resources initialized successfully!")
        app.run(debug=True)
    else:
        print("Failed to initialize resources. Make sure all required files exist.")
