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
    
    # Initialize optimizer and command graph
    agent_system.initialize_optimizer(embedding_manager, data_manager)
    agent_system.initialize_command_graph(data_manager)
    
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
        
        # Step 4: Get command chain recommendations if available
        chain_recommendations = None
        if agent_system.command_graph and retrieved_commands:
            # Use the first command as the starting point
            primary_command = retrieved_commands[0].get('Command', '')
            if primary_command:
                chain_recommendations = agent_system.get_command_chain_recommendations(
                    primary_command,
                    task_description=user_query
                )
        
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
        
        # Add chain recommendations if available
        if chain_recommendations:
            json_result["command_chains"] = chain_recommendations
        
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
        response_data = {
            'response': response,
            'analysis': analysis,
            'commands': [cmd.get('Command', '') for cmd in retrieved_commands],
            'query_id': f"query_{timestamp}",
            'json_file': f"query_{timestamp}.json"
        }
        
        # Add chain recommendations to response if available
        if chain_recommendations:
            response_data['command_chains'] = chain_recommendations
            
        return jsonify(response_data)
    
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
    
# Add a new route to visualize the command graph
@api.route('/api/command_graph', methods=['GET'])
def get_command_graph():
    """Generate and return a visualization of the command graph."""
    if not agent_system or not agent_system.command_graph:
        return jsonify({'error': 'Command graph not initialized'}), 500
        
    # Generate visualization
    graph_file = Path("command_graph.png")
    if agent_system.command_graph.visualize_graph(str(graph_file)):
        return send_file(
            graph_file,
            mimetype='image/png',
            as_attachment=False
        )
    else:
        return jsonify({'error': 'Failed to generate graph visualization'}), 500
        
# Add a new route to get command chain recommendations
@api.route('/api/command_chains/<command>', methods=['GET'])
def get_command_chains(command):
    """Get command chain recommendations for a specific command."""
    if not agent_system or not agent_system.command_graph:
        return jsonify({'error': 'Command graph not initialized'}), 500
        
    task_description = request.args.get('task', '')
    
    recommendations = agent_system.get_command_chain_recommendations(
        command,
        task_description=task_description if task_description else None
    )
    
    if not recommendations:
        return jsonify({'error': 'No recommendations found for this command'}), 404
        
    return jsonify(recommendations)