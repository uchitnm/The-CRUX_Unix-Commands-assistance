import os
import shutil

def create_directory_structure():
    """Create the necessary directory structure for the Flask application."""
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    # Check if required files exist
    required_files = [
        'agent_system.py',
        'query_optimization_algorithm.py',
        'linux_commands_tokenized.csv',
        'faiss_index.bin',
        'faiss_index_metadata.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"WARNING: Required file {file} does not exist")
    
    print("Directory structure setup complete!")

if __name__ == "__main__":
    create_directory_structure()
