import json
from google import genai
from google.genai import types
from app.config import settings
from app.core.query_optimization_algorithm import QueryOptimizer

class AgentSystem:
    def __init__(self):
        """Initialize the agent system."""
        try:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            # Test the API connection
            self.test_connection()
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API: {str(e)}")
            
        self.query_optimizer = None
    
    def test_connection(self):
        """Test the API connection with a simple request."""
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="test")],
            ),
        ]
        self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=settings.GEMINI_CONFIG,
        )
    
    def initialize_optimizer(self, embedding_manager, data_manager):
        """Initialize the query optimizer with necessary components."""
        if self.query_optimizer is None:
            self.query_optimizer = QueryOptimizer(
                embedding_model=embedding_manager.model,
                index=embedding_manager.index,
                df=data_manager.commands,
                chunk_metadata=data_manager.metadata
            )
    
    def query_analyzer_agent(self, query, embedding_manager, data_manager):
        """Agent responsible for understanding the user's intent and reformulating the query."""
        # Initialize optimizer if needed
        self.initialize_optimizer(embedding_manager, data_manager)
        
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
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=settings.GEMINI_CONFIG,
        )
        
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
            return {
                "intent": "unknown", 
                "keywords": [optimized_query], 
                "reformulated_query": optimized_query,
                "original_query": query
            }
    
    def retrieval_agent(self, query, analysis, embedding_manager, data_manager):
        """Agent responsible for retrieving relevant commands based on the query."""
        print(f"Retrieving commands for: {query}")
        
        # Get keywords and intent from analysis
        keywords = analysis.get("keywords", [query])
        intent = analysis.get("intent", "")
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Prefer optimized query if available
        if "optimized_query" in analysis and analysis["optimized_query"] != query:
            main_query = analysis["optimized_query"]
        else:
            main_query = query
            
        # Search for relevant commands with increased top_n for better filtering
        distances, indices = embedding_manager.search(main_query, top_n=20)
        
        if len(indices) == 0:
            return [], ""
            
        # Process results based on chunk metadata
        if data_manager.metadata is not None:
            # Get metadata for matched chunks
            matched_chunks = data_manager.metadata.iloc[indices].to_dict(orient="records")
            
            # Get unique original document indices
            original_indices = {chunk['original_idx'] for chunk in matched_chunks}
            
            # Get results from the original dataframe
            results = data_manager.commands.iloc[list(original_indices)].to_dict(orient="records")
            
            # Score and filter results based on relevance
            scored_results = []
            for cmd in results:
                # Calculate relevance score based on multiple factors
                score = 0
                
                # 1. Command name relevance
                cmd_name = cmd.get('Command', '').lower()
                if any(keyword.lower() in cmd_name for keyword in keywords):
                    score += 0.3
                
                # 2. Description relevance
                description = cmd.get('DESCRIPTION', '').lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in description)
                score += 0.2 * (keyword_matches / len(keywords))
                
                # 3. Intent relevance
                if intent and any(word in description.lower() for word in intent.lower().split()):
                    score += 0.2
                
                # 4. Examples relevance
                examples = cmd.get('EXAMPLES', '').lower()
                if examples and any(keyword.lower() in examples for keyword in keywords):
                    score += 0.15
                
                # 5. Options relevance
                options = cmd.get('OPTIONS', '').lower()
                if options and any(keyword.lower() in options for keyword in keywords):
                    score += 0.15
                
                # Add the score to the command dictionary
                cmd['relevance_score'] = score
                scored_results.append(cmd)
            
            # Sort by relevance score and take top results
            scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            results = scored_results[:settings.TOP_N_RESULTS]
            
            # Build detailed context from chunks
            context = self._build_context_from_chunks(results, matched_chunks, data_manager)
        else:
            # Original behavior if no chunking
            results = data_manager.commands.iloc[indices].to_dict(orient="records")
            context = self._build_context_from_commands(results)
        
        print(f"Retrieved {len(results)} commands")
        return results, context
    
    def _build_context_from_chunks(self, results, matched_chunks, data_manager):
        """Build context string from chunked results."""
        context = "Retrieved commands:\n\n"
        for cmd in results:
            context += f"Command: {cmd.get('Command', '')}\n"
            context += f"Description: {cmd.get('DESCRIPTION', '')}\n"
            
            # Add matched chunks for this command
            try:
                cmd_chunks = [chunk for chunk in matched_chunks 
                            if chunk['original_idx'] == data_manager.commands[
                                data_manager.commands['Command'] == cmd.get('Command', '')
                            ].index[0]]
                
                if cmd_chunks:
                    context += "Relevant sections:\n"
                    for chunk in cmd_chunks:
                        context += f"- {chunk.get('text', '')}\n"
            except Exception as e:
                print(f"Error processing chunks for command {cmd.get('Command', '')}: {e}")
            
            context += "\n---\n\n"
        return context
    
    def _build_context_from_commands(self, commands):
        """Build context string from command dictionaries."""
        context_parts = []
        for cmd in commands:
            relevance_score = cmd.get('relevance_score', 0)
            context_parts.append(f"Command: {cmd.get('Command', '')} (Relevance Score: {relevance_score:.2f})")
            context_parts.append(f"Description: {cmd.get('DESCRIPTION', '')}")
            context_parts.append(f"Examples: {cmd.get('EXAMPLES', '')}")
            context_parts.append(f"Options: {cmd.get('OPTIONS', '')}")
            context_parts.append("---")
        return "\n".join(context_parts)
    
    def response_generator_agent(self, query, analysis, context):
        """Agent responsible for generating the final response to the user."""
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
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=settings.GEMINI_CONFIG,
        )
        
        return response.text 