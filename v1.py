import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
from markdown import markdown
import re
import datetime
import json

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext while preserving command examples """
    # Preserve code blocks before conversion
    code_blocks = re.findall(r'```(.*?)```', markdown_string, re.DOTALL)
    code_placeholders = {}
    for i, block in enumerate(code_blocks):
        placeholder = f"CODE_BLOCK_{i}"
        code_placeholders[placeholder] = block
        markdown_string = markdown_string.replace(f"```{block}```", placeholder)

    # Preserve inline code before conversion
    inline_codes = re.findall(r'`(.*?)`', markdown_string)
    inline_placeholders = {}
    for i, code in enumerate(inline_codes):
        placeholder = f"INLINE_CODE_{i}"
        inline_placeholders[placeholder] = code
        markdown_string = markdown_string.replace(f"`{code}`", placeholder)

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)
    
    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(string=True))  # Updated to use 'string' instead of 'text'
    
    # Restore code blocks
    for placeholder, block in code_placeholders.items():
        text = text.replace(placeholder, block)
    
    # Restore inline code
    for placeholder, code in inline_placeholders.items():
        text = text.replace(placeholder, code)
        
    return text

# Load FAISS index and embedding model
def load_faiss_index(index_path, metadata_path=None, model_name="all-MiniLM-L6-v2"):
    """Load a pre-built FAISS index, metadata, and initialize the embedding model."""
    index = faiss.read_index(index_path)
    embedding_model = SentenceTransformer(model_name)
    
    # Load chunk metadata if provided
    chunk_metadata = None
    if metadata_path and os.path.exists(metadata_path):
        chunk_metadata = pd.read_csv(metadata_path)
    
    return index, embedding_model, chunk_metadata

# Retrieve relevant commands
def retrieve_relevant_data(df, query, index, embedding_model, chunk_metadata=None, top_n=5):
    """Retrieve the most relevant commands based on FAISS similarity search with chunk handling."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_n)

    if indices.size == 0:
        return [], ""

    # Handle chunked data if metadata is available
    if chunk_metadata is not None:
        # Get metadata for matched chunks
        matched_chunks = chunk_metadata.iloc[indices[0]].to_dict(orient="records")
        
        # Get unique original document indices
        original_indices = {chunk['original_idx'] for chunk in matched_chunks}
        
        # Get results from the original dataframe
        results = df.iloc[list(original_indices)].to_dict(orient="records")
        
        # Add chunk context to each result
        for result in results:
            result_chunks = [chunk for chunk in matched_chunks 
                            if chunk['original_idx'] == df[df['Command'] == result['Command']].index[0]]
            if result_chunks:
                result['matched_chunks'] = [chunk['chunk_text'] for chunk in result_chunks]
    else:
        # Original behavior if no chunking
        results = df.iloc[indices[0]].to_dict(orient="records")
    
    # Format retrieved data
    context = "\n".join([f"Command: {item['Command']}\nDescription: {item['DESCRIPTION']}" 
                        for item in results])
    
    return results, context

# Define agent roles and functions
class AgentSystem:
    def __init__(self, client, model="gemini-2.0-flash"):
        self.client = client
        self.model = model
    
    def query_analyzer_agent(self, query):
        """Agent responsible for understanding the user's intent and reformulating the query."""
        prompt = f"""
        You are a Query Analyzer Agent. Your role is to:
        1. Understand the true intent behind a user's UNIX command query
        2. Identify key concepts and command requirements
        3. Reformulate or expand the query if needed for better retrieval
        4. Return a structured analysis that will help with command retrieval
        
        User Query: {query}
        
        Respond with a JSON object with these fields:
        - intent: the main purpose of the query
        - keywords: key terms for retrieval
        - reformulated_query: an improved version of the original query
        """
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = self.client.models.generate_content(model=self.model, contents=contents).text
        
        # Extract JSON from response
        try:
            analysis = json.loads(response.strip())
            return analysis
        except:
            # Fallback if JSON parsing fails
            return {"intent": "unknown", "keywords": [query], "reformulated_query": query}
    
    def retrieval_agent(self, df, query, analysis, index, embedding_model, chunk_metadata=None, top_n=5):
        """Agent responsible for retrieving and ranking relevant commands."""
        # Use both original query and reformulated query for retrieval
        original_results, original_context = retrieve_relevant_data(df, query, index, embedding_model, chunk_metadata, top_n)
        reformed_results, reformed_context = retrieve_relevant_data(df, analysis["reformulated_query"], index, embedding_model, chunk_metadata, top_n)
        
        # Combine and deduplicate results
        all_commands = {}
        for result in original_results + reformed_results:
            if result["Command"] not in all_commands:
                all_commands[result["Command"]] = result
        
        # Let the agent rank and filter these results
        command_list = list(all_commands.values())
        command_json = json.dumps(command_list)
        
        prompt = f"""
        You are a Retrieval Agent for UNIX commands. Review these retrieved commands and the user's intent.
        
        User Query: {query}
        User Intent: {analysis["intent"]}
        
        Retrieved Commands:
        {command_json}
        
        Rank these commands by relevance to the query and user intent.
        Return a JSON array of the top 3-5 most relevant commands with their descriptions.
        """
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = self.client.models.generate_content(model=self.model, contents=contents).text
        
        # Extract JSON from response
        try:
            ranked_results = json.loads(response.strip())
            context = "\n".join([f"Command: {item['Command']}\nDescription: {item['DESCRIPTION']}" for item in ranked_results])
            return ranked_results, context
        except:
            # Fallback if JSON parsing fails
            return original_results[:3], original_context
    
    def response_generator_agent(self, query, analysis, context):
        """Agent responsible for generating the final structured response."""
        # This agent uses the Graph of Thoughts methodology to generate responses
        prompt = f"""
        You are a UNIX Command Response Agent. Given a user's query and retrieved commands, generate a clear, helpful response.
        Use the Graph of Thoughts reasoning method to approach this step by step.
        
        **User Query:** {query}
        **User Intent:** {analysis["intent"]}

        **Relevant Commands Found:**
        {context}

        **Graph of Thoughts Approach:**
        1. Initial Analysis: First, analyze what the user is asking for. Think about:
          a) What task they're trying to accomplish
          b) What type of command would be most helpful
          c) Which of the relevant commands best matches their need
        
        2. Command Identification: Examine each potential command:
          a) Evaluate which command is most relevant to the query
          b) Consider command functionality, common use cases, and limitations
          c) Compare alternative commands that might serve the same purpose
        
        3. Example Construction: Create a practical example:
          a) Make it clear and relevant to the user's query
          b) Show proper syntax and typical usage patterns
          c) Demonstrate the most useful flags or options
        
        **Important Guidelines:**
        - Focus on commonly used, well-established UNIX commands (like ls, find, grep, cat, etc.)
        - For file operations, prioritize commands like 'find', 'ls' or 'grep'
        - For text searching, prefer 'grep' or 'awk'
        - For file manipulation, use 'cp', 'mv', 'rm', etc.
        - Avoid suggesting obscure or non-standard commands unless explicitly required
        
        **Task:** Provide a structured response with the following format:
        - Query: <user query>
        - Command: <command name>
        - Example Usage: <specific command with all necessary flags and arguments>
        - Two-Line Description: <brief description of the command>
        - Optional Arguments or Flags: <list each flag on a new line with its description>
          
        Ensure all command examples and flags are surrounded by backticks to preserve formatting.
        For example: `-name "*.py"` rather than just -name "*.py"
        """
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")
        
        # Generate multiple candidate responses
        candidate_responses = []
        for _ in range(3):  # Generate 3 candidates
            response = ""
            for chunk in self.client.models.generate_content_stream(model=self.model, contents=contents, config=generate_content_config):
                response += chunk.text
            plain_response = markdown_to_text(response)  # Convert to plain text
            candidate_responses.append(plain_response)

        # Validate responses and filter out strange commands
        validated_responses = []
        common_unix_commands = [
            "ls", "find", "grep", "awk", "sed", "cat", "cp", "mv", "rm", "mkdir", 
            "chmod", "chown", "ps", "top", "kill", "df", "du", "tar", "gzip", 
            "ssh", "scp", "rsync", "curl", "wget", "sort", "uniq", "head", "tail",
            "less", "more", "touch", "echo", "which", "whereis", "man", "pwd"
        ]
        
        for response in candidate_responses:
            # Extract command name
            cmd_match = re.search(r"Command:\s*(\w+)", response)
            if cmd_match:
                cmd_name = cmd_match.group(1).lower()
                if cmd_name in common_unix_commands:
                    validated_responses.append(response)
                    continue
            
            # If command not in common list or not extracted, still keep the response
            # but with lower priority
            validated_responses.append(response)
            
        if not validated_responses:
            validated_responses = candidate_responses
            
        # Evaluate and select the best response
        _response_ = self.evaluate_responses(query, validated_responses)
        
        # Add colors to the response sections
        _response_ = _response_.replace("Query:", "\033[1;36mQuery:\033[0m")  # Cyan
        _response_ = _response_.replace("Command:", "\033[1;36mCommand:\033[0m")  # Cyan
        _response_ = _response_.replace("Example Usage:", "\033[1;36mExample Usage:\033[0m")  # Cyan
        _response_ = _response_.replace("Two-Line Description:", "\033[1;36mTwo-Line Description:\033[0m")  # Cyan
        _response_ = _response_.replace("Optional Arguments or Flags:", "\033[1;36mOptional Arguments or Flags:\033[0m")  # Cyan
        
        return _response_
    
    def evaluate_responses(self, query, responses):
        """Agent responsible for evaluating and selecting the best response."""
        if len(responses) == 1:
            return self.format_final_response(responses[0])
            
        responses_json = json.dumps(responses)
        
        prompt = f"""
        You are a Response Evaluation Agent. Evaluate these candidate responses to a user query and select the best one.
        
        User Query: {query}
        
        Candidate Responses:
        {responses_json}
        
        For each response, evaluate:
        1. Relevance to the user query
        2. Accuracy of command information - prefer standard UNIX commands like ls, find, grep, etc.
        3. Clarity of explanation
        4. Quality of example
        5. Completeness of command flags/options
        
        Return the index (0, 1, or 2) of the best response with a brief explanation.
        Focus on responses that use well-known UNIX commands rather than obscure ones.
        """
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = self.client.models.generate_content(model=self.model, contents=contents).text
        
        # Parse the response to get the best index
        try:
            # Look for a number in the response
            index_match = re.search(r"(\d+)", response)
            if index_match:
                best_index = int(index_match.group(1))
                if 0 <= best_index < len(responses):
                    return self.format_final_response(responses[best_index])
        except:
            pass
        
        # Default to the longest response if evaluation fails
        return self.format_final_response(max(responses, key=len))
    
    def format_final_response(self, response):
        """Format the final response to ensure all command details are preserved."""
        # Ensure all command information is properly displayed
        sections = [
            "Query:", "Command:", "Example Usage:", 
            "Two-Line Description:", "Optional Arguments or Flags:"
        ]
        
        formatted_response = response
        
        # Make sure there's no blank lines between section headers and content
        for section in sections:
            pattern = f"({section})\\s*\\n\\s*\\n"  # Fixed escape sequences
            formatted_response = re.sub(pattern, r"\1 ", formatted_response)
        
        # Ensure flags and options are properly formatted
        if "Optional Arguments or Flags:" in formatted_response:
            flags_section = formatted_response.split("Optional Arguments or Flags:")[1]
            # Replace any missing flag details
            flags_section = re.sub(r'(\\s*:)\\s*\\n', r'\1 [flag description missing]\n', flags_section)
            formatted_response = formatted_response.split("Optional Arguments or Flags:")[0] + "Optional Arguments or Flags:" + flags_section
            
        return formatted_response

def save_query_history(query, response, history_file="query_history.txt"):
    """Save the query and AI response to a history file."""
    with open(history_file, "a") as file:

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"\n[{timestamp}]\nQuery: {query}\nResponse: {response}\n{'-'*50}\n")

if __name__ == "__main__":
    # Load data and models
    history_file = "query_history.txt" 
    csv_file = "linux_commands_tokenized.csv"  # Adjust to your file path
    index_path = "faiss_index.bin"  # Path to pre-built FAISS index
    metadata_path = os.path.splitext(index_path)[0] + "_metadata.csv"  # Path to chunk metadata
    
    df = pd.read_csv(csv_file)  # Load CSV for command metadata
    index, embedding_model, chunk_metadata = load_faiss_index(index_path, metadata_path)

    # Update the agent system to pass chunk_metadata
    def process_query(user_query):
        """Process a query using the agent system."""
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        agent_system = AgentSystem(client)
        
        # Step 1: Analyze the query
        print("\033[1;33mAnalyzing query...\033[0m")
        analysis = agent_system.query_analyzer_agent(user_query)
        
        # Step 2: Retrieve relevant commands with chunk support
        print("\033[1;33mRetrieving relevant commands...\033[0m")
        retrieved_commands, context = agent_system.retrieval_agent(df, user_query, analysis, index, embedding_model, chunk_metadata)
        
        if not retrieved_commands:
            return "No relevant commands found. Try rephrasing your query."
        
        # Step 3: Generate response
        print("\033[1;33mGenerating response...\033[0m")
        return agent_system.response_generator_agent(user_query, analysis, context)

    # Interactive mode
    while True:
        user_query = input("Ask a UNIX-related question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            exit(0)
        response = process_query(user_query)
        save_query_history(user_query, response, history_file)
        print("\n\033[1;33m Ans :\033[0m", response, "\n")
