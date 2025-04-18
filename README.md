# UNIX Command Assistant

## Overview
The UNIX Command Assistant is a web-based application designed to help users query and understand UNIX commands. It provides detailed explanations, examples, and options for various UNIX commands, making it easier for users to perform tasks efficiently.

## Features
- **Query UNIX Commands**: Enter a query to get detailed information about UNIX commands.
- **Command Analysis**: View analysis of the query, including intent and keywords.
- **Download Results**: Download query results in JSON format for offline use.
- **Interactive UI**: A user-friendly interface with enhanced styling for better readability.

## Project Structure
```
GenAI_with_GemnAI/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── data_manager.py
│   │   ├── embeddings.py
│   │   ├── query_optimization_algorithm.py
│   ├── templates/
│   │   ├── index.html
│   ├── utils/
│       ├── __init__.py
├── query_results/
├── faiss_index_metadata.csv
├── faiss_index.bin
├── linux_commands_tokenized.csv
├── query_results.txt
├── README.md
├── run.py
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd GenAI_with_GemnAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```env
     GEMINI_API_KEY=<your-api-key>
     ```
4. Run the application:
   ```bash
   python run.py
   ```

## Usage
1. Open the application in your browser at `http://127.0.0.1:5000`.
2. Enter a query in the input box (e.g., "how to find files modified in the last 24 hours").
3. View the detailed response, including command descriptions, examples, and options.
4. Download the query results if needed.

## Configuration
- **Data Files**: Located in the root directory (`linux_commands_tokenized.csv`, `faiss_index.bin`, etc.).
- **Settings**: Modify `app/config/settings.py` for custom configurations like model settings, chunking parameters, and search parameters.

## Development
### Adding New Features
1. Add new routes in `app/api/routes.py`.
2. Update the frontend in `app/templates/index.html`.
3. Modify backend logic in `app/core/` as needed.

### Testing
- Use the `query_results/` directory to test query outputs.
- Ensure all changes are tested locally before deployment.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.


## Techstack
- **FAISS**: For efficient similarity search and clustering.
- **Flask**: For the web framework.
- **Google GenAI**: For advanced AI capabilities.