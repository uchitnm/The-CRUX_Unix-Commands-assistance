import os
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def load_command_data(file_path="linux_commands_tokenized.csv"):
    """Load command data from CSV file."""
    print(f"Loading command data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} commands")
    return df

def initialize_genai():
    """Initialize the Google Generative AI with the correct API."""
    # Get API key
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Configure the library with the API key
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Test the API
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, what can you tell me about Linux commands?")
    print("API test successful. Sample response:")
    print(response.text[:100] + "...")
    
    return model

def main():
    # Test loading data
    df = load_command_data()
    
    # Test initializing the genAI
    model = initialize_genai()
    
    print("\nSetup test complete. You can now modify the main.py file to use the correct API.")

if __name__ == "__main__":
    main()