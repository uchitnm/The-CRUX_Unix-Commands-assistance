from flask import Blueprint, request, jsonify, render_template, send_file
from app.core.agent import AgentSystem
from app.core.data_manager import DataManager
from app.core.embeddings import EmbeddingManager
import json
import datetime
import os
from pathlib import Path

# Create blueprint
api = Blueprint('api', __name__)

# Initialize managers
data_manager = DataManager()
embedding_manager = EmbeddingManager()
agent_system = None

def initialize_resources():
    """Initialize all resources needed for the application."""
    global agent_system
    
    # Load FAISS index or create new one
    if not embedding_manager.load_index():
        # Prepare chunks and create new index
        chunk_metadata, chunks = data_manager.prepare_chunks()
        embedding_manager.create_index(chunks)
        
        # Save resources
        embedding_manager.save_index()
        data_manager.save_chunk_metadata()
    else:
        # Load chunk metadata
        data_manager.load_chunk_metadata()
    
    # Initialize agent system
    agent_system = AgentSystem()
    
    return True

@api.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@api.route('/api/query', methods=['POST'])
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
            embedding_manager=embedding_manager,
            data_manager=data_manager
        )
        
        # Step 2: Retrieve relevant commands
        retrieved_commands, context = agent_system.retrieval_agent(
            analysis.get('reformulated_query', user_query),
            analysis,
            embedding_manager=embedding_manager,
            data_manager=data_manager
        )
        
        if not retrieved_commands:
            return jsonify({'response': 'No relevant commands found. Try rephrasing your query.'})
        
        # Step 3: Generate response
        response = agent_system.response_generator_agent(user_query, analysis, context)
        
        # Create detailed JSON result
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_result = {
            "metadata": {
                "timestamp": timestamp,
                "query_id": f"query_{timestamp}",
                "original_query": user_query,
                "optimized_query": analysis.get('reformulated_query', user_query),
                "query_intent": analysis.get('intent', 'unknown'),
                "keywords": analysis.get('keywords', []),
                "optimization_metrics": analysis.get('optimization_metrics', {})
            },
            "retrieved_commands": [
                {
                    "command": cmd.get('Command', ''),
                    "description": cmd.get('DESCRIPTION', ''),
                    "examples": cmd.get('EXAMPLES', ''),
                    "options": cmd.get('OPTIONS', '')
                } for cmd in retrieved_commands
            ],
            "context": context,
            "response": response,
            "analysis": {
                "query_analysis": analysis,
                "command_relevance": [
                    {
                        "command": cmd.get('Command', ''),
                        "relevance_score": cmd.get('relevance_score', 0)
                    } for cmd in retrieved_commands
                ]
            }
        }
        
        # Save to JSON file
        results_dir = Path("query_results")
        results_dir.mkdir(exist_ok=True)
        json_file = results_dir / f"query_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        # Save to text file for backward compatibility
        with open("query_results.txt", "a") as f:
            f.write(f"[{timestamp}] Query: {user_query} | Response: {response}\n\n")
        
        # Return the response with additional metadata
        return jsonify({
            'response': response,
            'analysis': analysis,
            'commands': [cmd.get('Command', '') for cmd in retrieved_commands],
            'query_id': f"query_{timestamp}",
            'json_file': f"query_{timestamp}.json"
        })
    
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/api/download/<query_id>', methods=['GET'])
def download_query_result(query_id):
    """Download the JSON result for a specific query."""
    try:
        json_file = Path("query_results") / f"{query_id}.json"
        if not json_file.exists():
            return jsonify({'error': 'Query result not found'}), 404
        
        return send_file(
            json_file,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"{query_id}.json"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500 