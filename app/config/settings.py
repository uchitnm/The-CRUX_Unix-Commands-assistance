import os
from pathlib import Path
from google import genai
from google.genai import types
import subprocess

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent.parent

# Data files
DATA_PATH = BASE_DIR / "linux_commands_tokenized.csv"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = BASE_DIR / "faiss_index_metadata.csv"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_CONFIG = types.GenerateContentConfig(
    response_mime_type="text/plain",
)

# Chunking parameters
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Search parameters
TOP_N_RESULTS = 5

# Load environment variables
from dotenv import load_dotenv
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys - Get from system environment variables first, then fallback to .env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")

# Flask settings
FLASK_DEBUG = True
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5050