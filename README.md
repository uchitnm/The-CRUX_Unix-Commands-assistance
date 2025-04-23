# Unix Commands Assistance

## Introduction

The Unix Commands Assistance is an intelligent command-line assistant that helps users find, understand, and utilize Unix commands through natural language queries. This application leverages advanced natural language processing and Generative AI techniques to interpret user queries, retrieve relevant Unix commands, and generate optimized command pipelines. Whether you're a beginner learning Unix commands or an experienced user looking for efficient command combinations, this tool provides contextual recommendations and explanations tailored to your specific tasks.

## Features

### Natural Language Query Processing
- Intelligent analysis of user queries to understand intent and extract key concepts
- Query reformulation and optimization for better command retrieval
- Context-aware interpretation of task requirements

### Semantic Command Search
- Powerful vector-based search using sentence transformers and FAISS indexing
- High-precision retrieval of relevant commands based on descriptions, examples, and options
- Efficient chunking of command documentation for granular search capabilities

### Command Chain Recommendations
- Graph-based command relationship modeling that understands I/O compatibility between commands
- Automatic generation of efficient command pipelines for complex tasks
- Optimization of command chains to eliminate redundancy and improve performance
- Intelligent recommendations for subsequent commands based on current context

### Command Pipeline Optimization
- Automatic detection and application of command shorthand patterns
- Replacement of multi-command sequences with more efficient alternatives
- Context-aware recommendations for flags and options that enhance efficiency

### Comprehensive Command Knowledge
- Detailed information on command syntax, options, and examples
- Category-based organization of commands (file operations, text processing, system info, etc.)
- Command compatibility analysis for effective command chaining

## Installation

Clone the repository:
```
git clone https://github.com/uchitnm/Unix-Commands-assistance.git
cd Unix-Commands-assistance
```

Set up a virtual environment:
```
python -m venv venv
source venv/bin/activate

# On Windows, use:
venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Download NLTK data:
```
python -m nltk.downloader all
```

Set up your API key:
```
export GEMINI_API_KEY="your api key"

# On Windows, use:
set GEMINI_API_KEY=your_api_key
```
Alternatively, create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_api_key
```

## Technology Stack

- Flask: Web framework
- Google Gemini AI: Language model integration
- NLTK: Natural language processing
- scikit-learn & NumPy: Machine learning and data handling
- FAISS: Fast similarity search
- Sentence Transformers: Text embeddings

## Usage
Run the application:
```
python run.py
```
Access the web interface at http://localhost:5000 (or the port indicated in the console).

## Requirements

- Python 3.7+
- FAISS-CPU (for MacOS) or FAISS-GPU (for systems with compatible GPUs)
- Google Gemini API key

## License

The CRUX is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1). This means:

- You are free to use, modify, and distribute this software
- If you modify the software, you must distribute your modifications under the LGPL-2.1
- You can link to this software from other programs, including proprietary ones
- The complete license text can be found in the LICENSE file

For more details, see the [GNU Lesser General Public License v2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html).
