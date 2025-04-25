# The CRUX (Command Reference for Unix & Windows eXecution)

## Introduction

The CRUX is an intelligent command-line assistant that helps users find, understand, and utilize both Unix and Windows commands through natural language queries. This application leverages advanced natural language processing and Generative AI techniques to interpret user queries, retrieve relevant commands, and generate optimized solutions. Whether you're a beginner learning command-line operations or an experienced user looking for efficient command combinations, this tool provides contextual recommendations and explanations tailored to your specific tasks.

## Features

### Cross-Platform Command Support
- Comprehensive Unix command documentation and usage guidance
- Windows command support with CMD and PowerShell compatibility information
- Intelligent detection of platform-specific queries
- Properly formatted command syntax for both Unix and Windows environments

### Natural Language Query Processing
- Intelligent analysis of user queries to understand intent and extract key concepts
- Query reformulation and optimization for better command retrieval
- Context-aware interpretation of task requirements
- Platform-specific query routing (Unix vs Windows)

### Semantic Command Search
- Powerful vector-based search using sentence transformers and FAISS indexing
- High-precision retrieval of relevant commands based on descriptions, examples, and options
- Efficient chunking of command documentation for granular search capabilities
- Platform-aware search results

### Command Chain Recommendations
- Graph-based command relationship modeling that understands I/O compatibility between commands
- Automatic generation of efficient command pipelines for complex tasks
- Optimization of command chains to eliminate redundancy and improve performance
- Intelligent recommendations for subsequent commands based on current context

### Modern Web Interface
- Clean, responsive design with dark mode
- Syntax-highlighted code blocks for both Unix and Windows commands
- Copy-to-clipboard functionality for command examples
- Query analysis visualization
- Downloadable JSON results for each query
- Welcome animation and smooth transitions

### Command Pipeline Optimization
- Automatic detection and application of command shorthand patterns
- Replacement of multi-command sequences with more efficient alternatives
- Context-aware recommendations for flags and options that enhance efficiency
- Platform-specific optimizations

### Comprehensive Command Knowledge
- Detailed information on command syntax, options, and examples
- Category-based organization of commands
- Command compatibility analysis for effective command chaining
- Platform-specific command variations and alternatives

## Installation

Clone the repository:
```bash
git clone https://github.com/uchitnm/The-CRUX.git
cd The-CRUX
```

Set up a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download NLTK data:
```bash
python -m nltk.downloader all
```

Set up your API key:
```bash
# On Unix/macOS:
export GEMINI_API_KEY="your_api_key"

# On Windows:
set GEMINI_API_KEY=your_api_key
```
Alternatively, create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_api_key
```

## Technology Stack

- Flask: Web framework
- Google Gemini AI: Advanced language model for command processing
- NLTK: Natural language processing
- scikit-learn & NumPy: Machine learning and data handling
- FAISS: Fast similarity search
- Sentence Transformers: Text embeddings
- Bootstrap & Custom CSS: Modern UI components
- Prism.js: Syntax highlighting for both Unix and Windows commands

## Usage

Run the application:
```bash
python run.py
```
Access the web interface at http://localhost:5050

### Example Queries

Unix commands:
- "How do I find files modified in the last 24 hours?"
- "Show disk usage in human-readable format"
- "Search for text in multiple files recursively"

Windows commands:
- "How do I list files in Windows command prompt?"
- "Create a new directory in PowerShell"
- "Copy files with progress bar in CMD"

## Requirements

- Python 3.7+
- FAISS-CPU (for MacOS/Linux) or FAISS-GPU (for systems with compatible GPUs)
- Google Gemini API key
- Modern web browser with JavaScript enabled

## License

The CRUX is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1). This means:

- You are free to use, modify, and distribute this software
- If you modify the software, you must distribute your modifications under the LGPL-2.1
- You can link to this software from other programs, including proprietary ones
- The complete license text can be found in the LICENSE file

For more details, see the [GNU Lesser General Public License v2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html).

## Authors

- Uchit N M
- Lakshmi Kamath
- Mahima N R
- Ajay Arya Gupta

PES University RR Campus, Bangalore
