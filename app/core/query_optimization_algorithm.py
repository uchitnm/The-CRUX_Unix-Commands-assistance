import nltk
from nltk.corpus import wordnet
import itertools
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from collections import defaultdict

class QueryOptimizer:
    def __init__(self, embedding_model, index, df, chunk_metadata=None):
        """
        Initialize the query optimizer with necessary components.
        
        Args:
            embedding_model: The SentenceTransformer model for embeddings
            index: FAISS index for retrieval
            df: DataFrame with command information
            chunk_metadata: Metadata for chunked data
        """
        self.embedding_model = embedding_model
        self.index = index
        self.df = df
        self.chunk_metadata = chunk_metadata
        self.command_categories = self.categorize_commands()
        self.query_cache = {}  # Cache for query results
        self.download_nltk_resources()
        
    def download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK resources: {e}")
    
    def categorize_commands(self):
        """Categorize commands by their primary functionality."""
        categories = defaultdict(list)
        
        # Define command categories
        file_ops = ['ls', 'find', 'cp', 'mv', 'rm', 'mkdir', 'touch', 'chmod', 'chown']
        text_processing = ['grep', 'sed', 'awk', 'cat', 'head', 'tail', 'less', 'more', 'sort', 'uniq', 'wc']
        system_info = ['ps', 'top', 'df', 'du', 'free', 'uname', 'who', 'w', 'uptime']
        compression = ['tar', 'gzip', 'zip', 'unzip', 'bzip2', 'xz']
        networking = ['ssh', 'scp', 'ping', 'curl', 'wget', 'netstat', 'ifconfig', 'ip', 'dig', 'nslookup']
        
        # Add commands to categories
        for cmd in file_ops:
            categories['file_operations'].append(cmd)
        for cmd in text_processing:
            categories['text_processing'].append(cmd)
        for cmd in system_info:
            categories['system_info'].append(cmd)
        for cmd in compression:
            categories['compression'].append(cmd)
        for cmd in networking:
            categories['networking'].append(cmd)
            
        return categories
    
    def generate_query_variations(self, query, max_variations=5):
        """Generate various reformulations of the user query."""
        variations = [query]  # Original query is always included
        
        # 1. Extract key terms and their synonyms
        words = nltk.word_tokenize(query.lower())
        pos_tags = nltk.pos_tag(words)
        
        # Focus on nouns and verbs
        key_terms = [(word, pos) for word, pos in pos_tags 
                     if pos.startswith('NN') or pos.startswith('VB')]
        
        # Get synonyms for key terms
        synonyms = {}
        for word, pos in key_terms:
            if len(word) <= 3:  # Skip short words
                continue
                
            word_synonyms = []
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ')
                    if syn != word and syn not in word_synonyms:
                        word_synonyms.append(syn)
            
            if word_synonyms:
                synonyms[word] = word_synonyms[:3]  # Limit to 3 synonyms per word
        
        # 2. Generate query variations by replacing terms with synonyms
        for word, syns in synonyms.items():
            for syn in syns:
                new_query = query.replace(word, syn)
                if new_query not in variations:
                    variations.append(new_query)
                    
        # 3. Add task-focused variations
        task_prefixes = [
            "how to", "command for", "unix command to", 
            "linux command for", "command line way to"
        ]
        
        # Remove existing prefixes if any
        clean_query = query
        for prefix in task_prefixes:
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):].strip()
                break
        
        # Add new prefixes
        for prefix in task_prefixes:
            if not query.lower().startswith(prefix):
                new_query = f"{prefix} {clean_query}"
                if new_query not in variations:
                    variations.append(new_query)
                    
        # 4. Add common UNIX command-related terms
        unix_terms = ["list", "find", "search", "display", "show", "create", "remove", "delete"]
        if not any(term in query.lower() for term in unix_terms):
            # Find the most appropriate term to add
            for term in unix_terms:
                if self.is_term_relevant(query, term):
                    new_query = f"{term} {query}"
                    if new_query not in variations:
                        variations.append(new_query)
                        break
        
        # 5. Add command category context if possible
        query_category = self.detect_query_category(query)
        if query_category:
            category_terms = {
                'file_operations': ["file", "directory", "folder"],
                'text_processing': ["text", "string", "pattern"],
                'system_info': ["system", "process", "resource"],
                'compression': ["compress", "archive", "zip"],
                'networking': ["network", "connection", "remote"]
            }
            
            terms = category_terms.get(query_category, [])
            for term in terms:
                if term not in query.lower():
                    new_query = f"{query} {term}"
                    if new_query not in variations:
                        variations.append(new_query)
        
        # Limit the number of variations
        return variations[:max_variations]
    
    def is_term_relevant(self, query, term):
        """Check if a term is relevant to add to the query."""
        # Simple relevance check based on command categories
        if term == "list" and any(w in query.lower() for w in ["file", "directory", "folder", "content"]):
            return True
        if term == "find" and any(w in query.lower() for w in ["search", "locate", "where"]):
            return True
        if term == "show" and any(w in query.lower() for w in ["display", "view", "see"]):
            return True
        if term == "create" and any(w in query.lower() for w in ["make", "new", "add"]):
            return True
        if term == "remove" and any(w in query.lower() for w in ["delete", "eliminate", "get rid"]):
            return True
        
        return False
    
    def detect_query_category(self, query):
        """Detect the likely command category for a query."""
        query = query.lower()
        
        # Define category indicators
        indicators = {
            'file_operations': ["file", "directory", "folder", "permission", "list", "find", "copy", "move", "delete"],
            'text_processing': ["text", "string", "pattern", "search", "replace", "word", "line", "content"],
            'system_info': ["system", "process", "memory", "disk", "cpu", "usage", "space"],
            'compression': ["compress", "archive", "extract", "zip", "tar", "unzip"],
            'networking': ["network", "download", "upload", "connect", "remote", "server", "ssh"]
        }
        
        # Count matches for each category
        matches = {category: 0 for category in indicators}
        for category, terms in indicators.items():
            for term in terms:
                if term in query:
                    matches[category] += 1
        
        # Find the category with most matches
        max_matches = max(matches.values())
        if max_matches > 0:
            best_categories = [c for c, m in matches.items() if m == max_matches]
            return best_categories[0]  # Return the first if there are ties
        
        return None
    
    def retrieve_commands(self, query, top_n=5):
        """Retrieve relevant commands using FAISS search."""
        # Check cache first
        if query in self.query_cache:
            return self.query_cache[query]
            
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_n)

        if indices.size == 0:
            return []

        # Process results based on chunk metadata if available
        if self.chunk_metadata is not None:
            # Get metadata for matched chunks
            matched_chunks = self.chunk_metadata.iloc[indices[0]].to_dict(orient="records")
            
            # Get unique original document indices
            original_indices = {chunk['original_idx'] for chunk in matched_chunks}
            
            # Get results from the original dataframe
            results = self.df.iloc[list(original_indices)].to_dict(orient="records")
        else:
            # Original behavior if no chunking
            results = self.df.iloc[indices[0]].to_dict(orient="records")
        
        # Store in cache
        self.query_cache[query] = results
        return results
    
    def evaluate_query(self, query, ground_truth=None):
        """
        Evaluate a query based on multiple metrics.
        
        Args:
            query: The query to evaluate
            ground_truth: Optional expected command for benchmark tests
            
        Returns:
            dict: Metrics for the query
        """
        start_time = time.time()
        retrieved_commands = self.retrieve_commands(query)
        retrieval_time = time.time() - start_time
        
        if not retrieved_commands:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'retrieval_time': retrieval_time,
                'command_count': 0,
                'diverse_categories': 0,
                'query_specificity': self.calculate_query_specificity(query),
                'overall_score': 0
            }
        
        # 1. Retrieval metrics
        command_count = len(retrieved_commands)
        
        # 2. Diversity metrics
        command_names = [cmd.get('Command', '') for cmd in retrieved_commands]
        categories_covered = set()
        for cmd in command_names:
            for category, commands in self.command_categories.items():
                if cmd in commands:
                    categories_covered.add(category)
                    break
        diverse_categories = len(categories_covered)
        
        # 3. Query specificity
        query_specificity = self.calculate_query_specificity(query)
        
        # 4. Ground truth evaluation if available
        precision = recall = f1 = 0
        if ground_truth:
            # Calculate precision, recall based on presence of ground truth command
            retrieved_set = set(command_names)
            truth_set = set([ground_truth])
            
            # Convert to binary classification problem
            y_true = [1 if cmd in truth_set else 0 for cmd in command_names]
            y_pred = [1 if cmd == ground_truth else 0 for cmd in command_names]
            
            if sum(y_true) > 0:  # Only calculate if ground truth is in retrieval set
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            # Without ground truth, use command specificity as proxy
            cmd_specificities = [len(cmd.get('DESCRIPTION', '').split()) for cmd in retrieved_commands]
            precision = min(1.0, sum(cmd_specificities) / (100 * len(cmd_specificities))) if cmd_specificities else 0
        
        # Calculate overall score - higher is better
        # Weight the factors according to importance
        overall_score = (
            (0.4 * precision if ground_truth else 0.4 * query_specificity) + 
            (0.3 * min(1.0, command_count / 5)) +  # Normalize command count
            (0.2 * min(1.0, diverse_categories / 3)) +  # Normalize diversity
            (0.1 * (1.0 - min(1.0, retrieval_time / 2)))  # Normalize time (lower is better)
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'retrieval_time': retrieval_time,
            'command_count': command_count,
            'diverse_categories': diverse_categories,
            'query_specificity': query_specificity,
            'overall_score': overall_score
        }
    
    def calculate_query_specificity(self, query):
        """Calculate how specific a query is (0-1 scale)."""
        # Count specific terms that indicate clarity in the query
        specific_terms = [
            'specific', 'exact', 'only', 'just', 'precisely',
            '.py', '.txt', '.c', '.go',  # File extensions
            '-r', '-l', '-a',  # Command options
            '--recursive', '--verbose',  # Long options
            '24 hours', 'modified', 'permission'  # Specific conditions
        ]
        
        # Count command indicators
        command_indicators = ['ls', 'find', 'grep', 'awk', 'sed', 'cat', 'rm']
        
        # Base specificity on query length and presence of specific terms
        query_words = query.lower().split()
        word_count = len(query_words)
        
        specificity = 0.3  # Base specificity
        
        # Add for length (up to a point)
        if word_count >= 3 and word_count <= 15:
            specificity += min(0.3, 0.05 * word_count)
        
        # Add for specific terms
        specific_count = sum(1 for term in specific_terms if term.lower() in query.lower())
        specificity += min(0.3, 0.1 * specific_count)
        
        # Add for command indicators
        command_count = sum(1 for cmd in command_indicators if cmd.lower() in query.lower())
        specificity += min(0.1, 0.05 * command_count)
        
        return min(1.0, specificity)
    
    def optimize_query(self, original_query, ground_truth=None, max_variations=5):
        """
        Generate and evaluate query variations to find the optimal query.
        
        Args:
            original_query: The user's original query
            ground_truth: Optional expected command for benchmark tests
            max_variations: Maximum number of query variations to test
            
        Returns:
            tuple: (best_query, evaluation_metrics, all_results)
        """
        # Generate query variations
        query_variations = self.generate_query_variations(original_query, max_variations)
        
        # Add the original query if not already included
        if original_query not in query_variations:
            query_variations.insert(0, original_query)
        
        # Evaluate each variation
        evaluation_results = {}
        for query in query_variations:
            metrics = self.evaluate_query(query, ground_truth)
            evaluation_results[query] = metrics
        
        # Find the best query based on overall score
        best_query = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])[0]
        
        return best_query, evaluation_results[best_query], evaluation_results
