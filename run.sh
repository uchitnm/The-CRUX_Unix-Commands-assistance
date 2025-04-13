#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Check for GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    echo "Please run: export GEMINI_API_KEY=\"your_api_key_here\""
    exit 1
fi

# Run the program
python main.py