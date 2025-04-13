import json
import os
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from google.generativeai import types
from sentence_transformers import SentenceTransformer
from query_optimization_algorithm import QueryOptimizer

class AgentSystem:
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initialize the agent system.
        
        Args:
            model_name: Model identifier to use for generation
        """
        # Configure the genAI library
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        self.query_optimizer = None  # Will be initialized when needed
    
    def initialize_optimizer(self, df, index, embedding_model, chunk_metadata=None):
        """Initialize the query optimizer with necessary components."""
        if self.query_optimizer is None:
            self.query_optimizer = QueryOptimizer(
                embedding_model=embedding_model,
                index=index,
                df=df,
                chunk_metadata=chunk_metadata
            )
    
    def query_analyzer_agent(self, query, df=None, index=None, embedding_model=None, chunk_metadata=None):
        """
        Agent responsible for understanding the user's intent and reformulating the query.
        """
        # Initialize optimizer if needed and components are provided
        if df is not None and index is not None and embedding_model is not None:
            self.initialize_optimizer(df, index, embedding_model, chunk_metadata)
        
        # If optimizer is available, use it to improve the query
        optimized_query = query
        optimization_metrics = None
        all_results = None
        
        if self.query_optimizer is not None:
            optimized_query, optimization_metrics, all_results = self.query_optimizer.optimize_query(query)
            print(f"Optimized query: {optimized_query}")
            
            # For debugging - output metrics
            if optimization_metrics:
                print(f"Optimization metrics:")
                print(f"  - Overall score: {optimization_metrics['overall_score']:.2f}")
                print(f"  - Query specificity: {optimization_metrics['query_specificity']:.2f}")
                print(f"  - Command count: {optimization_metrics['command_count']}")
        
        # Original LLM-based analysis
        prompt = f"""
        You are a Query Analyzer Agent. Your role is to:
        1. Understand the true intent behind a user's UNIX command query
        2. Identify key concepts and command requirements
        3. Reformulate or expand the query if needed for better retrieval
        4. Return a structured analysis that will help with command retrieval
        
        User Query: {optimized_query}
        
        Respond with a JSON object with these fields:
        - intent: the main purpose of the query
        - keywords: key terms for retrieval
        - reformulated_query: an improved version of the original query
        """
        
        response = self.model.generate_content(prompt)
        
        # Extract JSON from response
        try:
            # Remove any markdown code blocks if present
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis = json.loads(response_text.strip())
            # Add optimization information if available
            if optimization_metrics:
                analysis['original_query'] = query
                analysis['optimized_query'] = optimized_query
                analysis['optimization_metrics'] = optimization_metrics
            return analysis
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            # Fallback if JSON parsing fails
            return {
                "intent": "unknown", 
                "keywords": [optimized_query], 
                "reformulated_query": optimized_query,
                "original_query": query
            }
    
    def retrieval_agent(self, df, query, analysis, index, embedding_model, chunk_metadata=None, top_n=5):
        """
        Agent responsible for retrieving relevant commands based on the query.
        """
        print(f"Retrieving commands for: {query}")
        
        # Get keywords from analysis if available
        keywords = analysis.get("keywords", [query])
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Prefer optimized query if available
        if "optimized_query" in analysis and analysis["optimized_query"] != query:
            main_query = analysis["optimized_query"]
        else:
            main_query = query
            
        # Encode query
        query_embedding = embedding_model.encode([main_query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_n)

        if indices.size == 0:
            return [], ""

        # Process results based on chunk metadata if available
        if chunk_metadata is not None:
            # Get metadata for matched chunks
            matched_chunks = chunk_metadata.iloc[indices[0]].to_dict(orient="records")
            
            # Get unique original document indices
            original_indices = {chunk['original_idx'] for chunk in matched_chunks}
            
            # Get results from the original dataframe
            results = df.iloc[list(original_indices)].to_dict(orient="records")
            
            # Build detailed context from chunks
            context = "Retrieved commands:\n\n"
            for cmd in results:
                context += f"Command: {cmd.get('Command', '')}\n"
                context += f"Description: {cmd.get('DESCRIPTION', '')}\n"
                
                # Add matched chunks for this command
                try:
                    cmd_chunks = [chunk for chunk in matched_chunks 
                                if chunk['original_idx'] == df[df['Command'] == cmd.get('Command', '')].index[0]]
                    
                    if cmd_chunks:
                        context += "Relevant sections:\n"
                        for chunk in cmd_chunks:
                            context += f"- {chunk.get('text', '')}\n"
                except Exception as e:
                    print(f"Error processing chunks for command {cmd.get('Command', '')}: {e}")
                
                context += "\n---\n\n"
        else:
            # Original behavior if no chunking
            results = df.iloc[indices[0]].to_dict(orient="records")
            
            # Build context from complete command entries
            context = "Retrieved commands:\n\n"
            for cmd in results:
                context += f"Command: {cmd.get('Command', '')}\n"
                context += f"Description: {cmd.get('DESCRIPTION', '')}\n"
                
                if "EXAMPLES" in cmd and cmd["EXAMPLES"]:
                    context += f"Examples: {cmd.get('EXAMPLES', '')}\n"
                if "OPTIONS" in cmd and cmd["OPTIONS"]:
                    context += f"Options: {cmd.get('OPTIONS', '')}\n"
                
                context += "\n---\n\n"
        
        print(f"Retrieved {len(results)} commands")
        return results, context
    
    def response_generator_agent(self, query, analysis, context):
        """
        Agent responsible for generating the final response to the user.
        """
        prompt = f"""
        You are a UNIX Command Assistant. Your role is to:
        1. Provide clear, helpful explanations of UNIX commands
        2. Respond directly to the user's query using the provided context
        3. Focus on the most relevant command(s) for the user's need
        4. Include practical examples that address the specific use case
        
        User Query: {query}
        
        Query Analysis: {json.dumps(analysis)}
        
        Context Information:
        {context}
        
        Based on the above information, provide a clear, concise response that directly answers the user's query. Include:
        1. The most appropriate command(s) for their need
        2. A brief explanation of how the command works
        3. 1-2 specific examples tailored to their use case
        4. Any relevant flags or options they should know about
        
        Format your response in an easy-to-read way with markdown formatting.
        """
        
        response = self.model.generate_content(prompt)
        return response.text