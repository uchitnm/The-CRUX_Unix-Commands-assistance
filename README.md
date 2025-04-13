# UNIX Command Assistant with Query Optimization

This application helps users find and understand UNIX commands by processing natural language queries. It includes an advanced query optimization system that improves search results by reformulating queries for better command retrieval.

## Features

- **Natural Language Understanding**: Process user queries about UNIX commands
- **Query Optimization**: Automatically reformulate queries to improve search accuracy
- **Chunked Retrieval**: Break command documentation into chunks for precise retrieval
- **Context-Aware Responses**: Generate responses tailored to the user's specific need

## Query Optimization System

The system uses a combination of metrics to evaluate and optimize queries:

1. **Command Match Score**: How well the query retrieves the most appropriate command
2. **Example Relevance**: How applicable the retrieved examples are to the user's specific use case  
3. **Response Conciseness**: How efficiently the command is explained without unnecessary information

The optimizer generates multiple variations of the original query and selects the one that maximizes these metrics.

## System Components

- `query_optimizer.py`: The query optimization algorithm
- `agent_system.py`: The core agent system that manages query analysis, retrieval, and response generation
- `main.py`: The main application that ties everything together
- `benchmark.py`: Script to evaluate the query optimization system
- `requirements.txt`: Required dependencies

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your API key:
   ```
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. Run the application:
   ```
   python main.py
   ```

4. To benchmark the query optimization system:
   ```
   python benchmark.py
   ```

## Example

Input:
```
> how to find files modified in the last 24 hours
```

The system will:
1. Analyze the query's intent
2. Generate variations of the query to optimize for precision
3. Retrieve the most relevant commands (in this case, likely `find`)
4. Generate a detailed response with examples specific to finding files modified in the last 24 hours

## How Query Optimization Works

1. **Query Variation Generation**:
   - Add task-focused prefixes ("command for", "unix command to")
   - Replace key terms with synonyms
   - Add category-specific context
   - Add command-specific terminology

2. **Query Evaluation**:
   - Measure retrieval precision with multiple metrics
   - Calculate query specificity score
   - Assess command diversity in results
   - Factor in retrieval time

3. **Selection of Best Query**:
   - Choose the query variation that achieves the highest overall score
   - Use this optimal query for command retrieval
   - Include both optimization metrics and analysis in the final response
