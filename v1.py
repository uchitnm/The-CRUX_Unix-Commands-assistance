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

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code>', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(string=True))  # Updated to use 'string' instead of 'text'

    return text

# Load FAISS index and embedding model
def load_faiss_index(index_path, model_name="all-MiniLM-L6-v2"):
    """Load a pre-built FAISS index and initialize the embedding model."""
    index = faiss.read_index(index_path)
    embedding_model = SentenceTransformer(model_name)
    return index, embedding_model

# Retrieve relevant commands
def retrieve_relevant_data(df, query, index, embedding_model, top_n=5):
    """Retrieve the most relevant commands based on FAISS similarity search."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_n)

    if indices.size == 0:
        return [], ""

    results = df.iloc[indices[0]].to_dict(orient="records")
    
    # Format retrieved data
    context = "\n".join([f"Command: {item['Command']}\nDescription: {item['DESCRIPTION']}" for item in results])
    
    return results, context

# Generate a response using Gemini AI with Graph of Thoughts
def generate_response(query, context):
    """Generate a structured response using Gemini AI with Graph of Thoughts."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"

    # Refined Query Template with Graph of Thoughts
    prompt = f"""
    You are a UNIX command assistant. Given a user's query, retrieve and suggest the most relevant UNIX command.
    Use the Graph of Thoughts reasoning method to approach this step by step.
    
    **User Query:** {query}

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
    
    **Example of this approach:**
    For query "how to find files by name":
    - Initial Analysis: User needs to search for files in the filesystem based on filename.
    - Command Identification: 'find' command is best suited as it's designed for filesystem searches, more powerful than 'ls' or 'grep' for this purpose.
    - Example Construction: 'find /home -name "*.txt"' demonstrates searching for all text files in the home directory.
    
    **Task:** Provide a structured response with the following format:
    - Query: <user query>
    - Command: <command name>
    - Example Usage: <example command usage>
    - Two-Line Description: <brief description of the command>
    - Optional Arguments or Flags: <list of optional arguments or flags>
    """
    
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
    ]

    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    # Generate multiple candidate responses
    candidate_responses = []
    for _ in range(3):  # Generate 3 candidates
        response = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            response += chunk.text
        plain_response = markdown_to_text(response)  # Convert to plain text
        candidate_responses.append(plain_response)

    # Evaluate and select the best response
    best_response = select_best_response(candidate_responses)

    # Add colors to the response sections
    best_response = best_response.replace("Query:", "\033[1;36mQuery:\033[0m")  # Cyan
    best_response = best_response.replace("Command:", "\033[1;36mCommand:\033[0m")  # Cyan
    best_response = best_response.replace("Example Usage:", "\033[1;36mExample Usage:\033[0m")  # Cyan
    best_response = best_response.replace("Two-Line Description:", "\033[1;36mTwo-Line Description:\033[0m")  # Cyan
    best_response = best_response.replace("Optional Arguments or Flags:", "\033[1;36mOptional Arguments or Flags:\033[0m")  # Cyan

    return best_response

def select_best_response(responses):
    """Select the best response from multiple candidates."""
    # Placeholder logic: Select the longest response as the best one
    # You can replace this with more sophisticated evaluation criteria
    return max(responses, key=len)

# Save query history
def save_query_history(query, response, history_file="query_history.txt"):
    """Save the query and AI response to a history file."""
    with open(history_file, "a") as file:

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"\n[{timestamp}]\nQuery: {query}\nResponse: {response}\n{'-'*50}\n")

def process_query(user_query):
    """Process a query and return the AI response."""
    retrieved_commands, context = retrieve_relevant_data(df, user_query, index, embedding_model)
    if not retrieved_commands:
        return "No relevant commands found. Try rephrasing your query."
    return generate_response(user_query, context)

if __name__ == "__main__":
    # Load data and models
    history_file = "query_history.txt" 
    csv_file = "linux_commands_tokenized.csv"  # Adjust to your file path
    index_path = "faiss_index.bin"  # Path to pre-built FAISS index
    df = pd.read_csv(csv_file)  # Load CSV for command metadata
    index, embedding_model = load_faiss_index(index_path)

    # Interactive mode
    while True:
        user_query = input("Ask a UNIX-related question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            exit(0)
        response = process_query(user_query)
        save_query_history(user_query, response, history_file)
        print("\n\033[1;33m Ans :\033[0m", response, "\n")
