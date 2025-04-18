import numpy as np
import networkx as nx
from collections import defaultdict
import pandas as pd
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandGraph:
    """
    Creates and manages a graph of UNIX commands that can be chained together.
    Provides functionality to find optimal command sequences for complex tasks.
    """

    def __init__(self, commands_df):
        """
        Initialize the command graph with a DataFrame of commands.

        Args:
            commands_df: DataFrame containing command information
        """
        if not isinstance(commands_df, pd.DataFrame):
            raise TypeError("commands_df must be a pandas DataFrame")
        
        if 'Command' not in commands_df.columns:
            raise ValueError("commands_df must contain a 'Command' column")
            
        self.commands_df = commands_df
        self.graph = nx.DiGraph()
        self.command_chains = defaultdict(int)
        self.command_metadata = {}
        
        # Pre-define command sets for faster lookups
        self.filter_commands = {'grep', 'sed', 'awk', 'tr', 'cut', 'head', 'tail', 'sort', 'uniq', 'wc', 'less', 'more', 'cat', 'tee', 'xargs'}
        self.transform_commands = {'awk', 'sed', 'tr', 'cut', 'paste', 'join', 'fmt', 'pr', 'fold'}
        self.non_stdout_commands = {'mv', 'cp', 'mkdir', 'rmdir', 'rm', 'touch', 'chmod', 'chown', 'chgrp'}
        
        # Build the graph
        self._build_command_metadata()
        self._build_initial_graph()
        logger.info(f"Initialized CommandGraph with {len(self.command_metadata)} commands")

    def _safe_lower(self, val):
        """Safely convert value to lowercase string."""
        if pd.isna(val):
            return ''
        return str(val).lower() if val is not None else ''

    def _build_command_metadata(self):
        """Extract and process metadata for each command."""
        try:
            # Convert all string columns to lowercase in advance
            string_cols = ['DESCRIPTION', 'EXAMPLES', 'OPTIONS']
            for col in string_cols:
                if col in self.commands_df.columns:
                    self.commands_df[col] = self.commands_df[col].fillna('').astype(str).str.lower()
            
            for idx, row in self.commands_df.iterrows():
                cmd_name = row.get('Command', '')
                if not cmd_name:
                    continue
                
                # Check for required columns
                for col in ['DESCRIPTION', 'EXAMPLES']:
                    if col not in row:
                        row[col] = ''
                
                # Process command metadata
                accepts_stdin = self._accepts_stdin(row)
                produces_stdout = cmd_name not in self.non_stdout_commands
                
                self.command_metadata[cmd_name] = {
                    'accepts_stdin': accepts_stdin,
                    'produces_stdout': produces_stdout,
                    'category': self._get_command_category(row),
                    'filters_data': cmd_name in self.filter_commands,
                    'transforms_data': cmd_name in self.transform_commands,
                    'examples': row.get('EXAMPLES', ''),
                    'description': row.get('DESCRIPTION', '')
                }
            
            logger.info(f"Built metadata for {len(self.command_metadata)} commands")
        except Exception as e:
            logger.error(f"Error building command metadata: {e}")
            raise

    def _accepts_stdin(self, command_row):
        """Determine if a command accepts standard input."""
        try:
            desc = self._safe_lower(command_row.get('DESCRIPTION', ''))
            examples = self._safe_lower(command_row.get('EXAMPLES', ''))
            options = self._safe_lower(command_row.get('OPTIONS', ''))
            command = self._safe_lower(command_row.get('Command', ''))
            
            # Check for pipe pattern in examples
            pipe_pattern = r'[^|]+\|\s*' + re.escape(command)
            has_pipe_examples = re.search(pipe_pattern, examples) is not None
            
            # Check for stdin indicators in descriptions
            stdin_indicators = ['standard input', 'stdin', 'pipe', 'pipeline', 'read from']
            input_indicators = any(indicator in desc or indicator in examples or indicator in options 
                                for indicator in stdin_indicators)
            
            # Return result based on multiple criteria
            return (has_pipe_examples or 
                    input_indicators or 
                    command in self.filter_commands)
                    
        except Exception as e:
            logger.warning(f"Error determining stdin acceptance for {command_row.get('Command', '')}: {e}")
            # Default to False on error
            return False

    def _get_command_category(self, command_row):
        """Categorize a command based on its description and function."""
        try:
            command = self._safe_lower(command_row.get('Command', ''))
            desc = self._safe_lower(command_row.get('DESCRIPTION', ''))
            
            # Use dictionary lookups for more efficient categorization
            category_commands = {
                'file_operations': ['ls', 'find', 'cp', 'mv', 'rm', 'mkdir', 'touch', 'chmod', 'chown'],
                'text_processing': ['grep', 'sed', 'awk', 'cat', 'head', 'tail', 'less', 'more', 'sort', 'uniq', 'wc'],
                'system_info': ['ps', 'top', 'df', 'du', 'free', 'uname', 'who', 'w', 'uptime'],
                'compression': ['tar', 'gzip', 'zip', 'unzip', 'bzip2', 'xz'],
                'networking': ['ssh', 'scp', 'ping', 'curl', 'wget', 'netstat', 'ifconfig', 'ip']
            }
            
            # Check if command belongs to a specific category
            for category, commands in category_commands.items():
                if command in commands:
                    return category
            
            # Check description for category indicators
            categories = {
                'file_operations': ['file', 'directory', 'folder', 'list', 'permission'],
                'text_processing': ['text', 'string', 'pattern', 'search', 'replace', 'word', 'line'],
                'system_info': ['system', 'process', 'memory', 'disk', 'cpu', 'usage', 'space'],
                'compression': ['compress', 'archive', 'extract', 'zip', 'tar', 'unzip'],
                'networking': ['network', 'download', 'upload', 'connect', 'remote', 'server', 'ssh']
            }
            
            for category, indicators in categories.items():
                if any(indicator in desc for indicator in indicators):
                    return category
            
            return 'other'
        except Exception as e:
            logger.warning(f"Error categorizing command {command_row.get('Command', '')}: {e}")
            return 'other'
    
    def _build_initial_graph(self):
        """Build the initial command graph based on I/O compatibility."""
        try:
            # Add nodes for all commands
            for cmd_name, metadata in self.command_metadata.items():
                self.graph.add_node(cmd_name, **metadata)
            
            # Add edges between compatible commands
            edge_count = 0
            for source_cmd, source_meta in self.command_metadata.items():
                if source_meta['produces_stdout']:
                    for target_cmd, target_meta in self.command_metadata.items():
                        if target_meta['accepts_stdin'] and source_cmd != target_cmd:
                            # Initial weight is based on compatibility
                            compatibility = self._calculate_compatibility(source_cmd, target_cmd)
                            self.graph.add_edge(source_cmd, target_cmd, weight=compatibility, count=0)
                            edge_count += 1
            
            logger.info(f"Built initial graph with {len(self.command_metadata)} nodes and {edge_count} edges")
        except Exception as e:
            logger.error(f"Error building initial graph: {e}")
            raise
    
    def _calculate_compatibility(self, source_cmd, target_cmd):
        """Calculate compatibility score between two commands."""
        try:
            if source_cmd not in self.command_metadata or target_cmd not in self.command_metadata:
                return 1.0  # Default compatibility if metadata missing
                
            source_meta = self.command_metadata[source_cmd]
            target_meta = self.command_metadata[target_cmd]
            
            # Base compatibility score
            compatibility = 1.0
            
            # Adjust based on command categories
            if source_meta['category'] == target_meta['category']:
                compatibility *= 0.8  # Same category commands may be redundant
            
            # Filter + Transform is a common pattern
            if source_meta['filters_data'] and target_meta['transforms_data']:
                compatibility *= 0.7
            elif source_meta['transforms_data'] and target_meta['filters_data']:
                compatibility *= 0.7
                
            # Initial high weights for common combinations
            common_pairs = {
                ('ls', 'grep'): 0.5, ('find', 'grep'): 0.5, ('ps', 'grep'): 0.5, 
                ('cat', 'grep'): 0.5, ('grep', 'awk'): 0.5, ('grep', 'sed'): 0.5,
                ('sort', 'uniq'): 0.5, ('sort', 'head'): 0.5, ('sort', 'tail'): 0.5,
                ('cat', 'wc'): 0.5, ('cat', 'sort'): 0.5, ('ls', 'wc'): 0.5,
            }
            
            # Use direct dictionary lookup for efficiency
            if (source_cmd, target_cmd) in common_pairs:
                compatibility *= common_pairs[(source_cmd, target_cmd)]
                
            return compatibility
        except Exception as e:
            logger.warning(f"Error calculating compatibility between {source_cmd} and {target_cmd}: {e}")
            return 1.0  # Default compatibility on error
    
    def update_graph_with_usage(self, command_chain):
        """
        Update the graph based on observed command chains.
        
        Args:
            command_chain: List of commands that were chained together
        """
        if not command_chain or len(command_chain) < 2:
            return
            
        try:
            # Update counts for each command pair in the chain
            for i in range(len(command_chain) - 1):
                source = command_chain[i]
                target = command_chain[i+1]
                
                if source in self.command_metadata and target in self.command_metadata:
                    # Increment the count for this command pair
                    pair_key = (source, target)
                    self.command_chains[pair_key] += 1
                    
                    # Update edge weight
                    if self.graph.has_edge(source, target):
                        count = self.command_chains[pair_key]
                        # Adjust weight based on usage frequency
                        # More usage = lower weight = higher preference
                        new_weight = 1.0 / (1.0 + np.log1p(count))
                        self.graph.edges[source, target]['weight'] = new_weight
                        self.graph.edges[source, target]['count'] = count
                        
            logger.debug(f"Updated graph with usage data for command chain: {' | '.join(command_chain)}")
        except Exception as e:
            logger.warning(f"Error updating graph with usage data: {e}")
    
    def _command_accomplishes_task(self, command, task_description):
        """Check if a single command can accomplish the described task."""
        if not task_description:
            return False
        
        try:    
            task_words = set(self._safe_lower(task_description).split())
            cmd_meta = self.command_metadata.get(command, {})
            
            # Check if command description and examples have good overlap with task
            description = self._safe_lower(cmd_meta.get('description', ''))
            examples = self._safe_lower(cmd_meta.get('examples', ''))
            
            desc_words = set(description.split())
            example_words = set(examples.split())
            
            # Check if task keywords are present in command description or examples
            overlap_desc = len(task_words.intersection(desc_words))
            overlap_examples = len(task_words.intersection(example_words))
            
            # Use more efficient dictionary lookup for task-specific keywords
            task_keywords = {
                'count': ['wc', 'grep -c', 'find'],
                'search': ['grep', 'find'],
                'sort': ['sort'],
                'unique': ['uniq', 'sort -u'],
                'display': ['cat', 'less', 'more', 'head', 'tail'],
                'list': ['ls', 'find'],
                'delete': ['rm'],
                'move': ['mv'],
                'copy': ['cp'],
                'create': ['touch', 'mkdir'],
                'compress': ['tar', 'gzip', 'zip'],
                'extract': ['tar', 'unzip'],
                'download': ['curl', 'wget'],
            }
            
            # Check if the task contains a keyword that matches this command's purpose
            for keyword, commands in task_keywords.items():
                if keyword in task_description.lower() and command in commands:
                    return True
            
            # Consider the command sufficient if it has strong overlap with task description
            return (overlap_desc >= 2 or overlap_examples >= 2)
            
        except Exception as e:
            logger.warning(f"Error determining if {command} accomplishes task: {e}")
            return False

    def _commands_compatible(self, source, target):
        """Check if source command output can be piped to target command."""
        try:
            if source not in self.command_metadata or target not in self.command_metadata:
                return False
                
            source_meta = self.command_metadata[source]
            target_meta = self.command_metadata[target]
            
            return source_meta.get('produces_stdout', False) and target_meta.get('accepts_stdin', False)
        except Exception as e:
            logger.warning(f"Error checking compatibility between {source} and {target}: {e}")
            return False

    def _optimize_command_chain(self, path, task_description=None):
        """Optimize a command chain by removing redundant commands and applying known patterns."""
        if not path or len(path) < 2:
            return path
        
        try:
            # Common command chain optimizations - use dictionary for O(1) lookups
            optimizations = {
                ('grep', 'wc'): ['grep -c'],  # Replace 'grep | wc -l' with 'grep -c'
                ('cat', 'grep'): ['grep'],    # Replace 'cat file | grep pattern' with 'grep pattern file'
                ('cat', 'awk'): ['awk'],      # Replace 'cat file | awk' with 'awk file'
                ('cat', 'sed'): ['sed'],      # Replace 'cat file | sed' with 'sed file'
                ('sort', 'uniq'): ['sort -u'],  # Replace 'sort | uniq' with 'sort -u'
                ('ls', 'grep'): ['ls | grep'],  # Retain, but could later optimize
                ('find', 'grep'): ['find | grep'],
                ('sort', 'head'): ['sort | head'],
                ('sort', 'tail'): ['sort | tail'],
                ('ps', 'grep'): ['pgrep'],  # Replace 'ps | grep' with 'pgrep'
                ('du', 'sort'): ['du | sort -n'],  # Typical disk usage sort
                ('df', 'grep'): ['df | grep'],
                ('cat', 'head'): ['head'],  # Replace 'cat file | head' with 'head file'
                ('cat', 'tail'): ['tail'],  # Replace 'cat file | tail' with 'tail file'
            }
    
            # Check for patterns in the chain
            i = 0
            optimized = []
            while i < len(path) - 1:
                pair = (path[i], path[i+1])
                if pair in optimizations:
                    optimized_cmds = optimizations[pair]
                    optimized.extend(optimized_cmds)
                    i += 2  # Skip both commands in the pair
                else:
                    optimized.append(path[i])
                    i += 1
            
            # Add the last command if we haven't processed it yet
            if i == len(path) - 1:
                optimized.append(path[i])
            
            # Keep unique commands and maintain order
            seen = set()
            result = []
            for cmd in optimized:
                if cmd not in seen:
                    result.append(cmd)
                    seen.add(cmd)
            
            return result
        except Exception as e:
            logger.warning(f"Error optimizing command chain: {e}")
            return path  # Return original path on error

    def find_command_chain(self, source_cmd, target_cmd=None, task_description=None, max_length=5):
        """
        Find the optimal chain of commands from source to target.
        If target is None, find a chain that best matches the task description.
        Prioritizes shorter command chains when possible.
        
        Args:
            source_cmd: The starting command
            target_cmd: The desired ending command (optional)
            task_description: Description of the task to accomplish (optional)
            max_length: Maximum chain length
            
        Returns:
            list: List of commands in the optimal chain or multiple chains
        """
        try:
            if source_cmd not in self.graph:
                logger.warning(f"Source command '{source_cmd}' not in graph")
                return []
                
            if target_cmd and target_cmd not in self.graph:
                logger.warning(f"Target command '{target_cmd}' not in graph")
                return []
                
            # First check if the task can be accomplished with just the source command
            # For simple tasks, avoid creating unnecessary chains
            if task_description and self._command_accomplishes_task(source_cmd, task_description):
                logger.info(f"Task can be accomplished with single command '{source_cmd}'")
                return [source_cmd]
                
            if target_cmd:
                try:
                    # Find shortest path from source to target
                    path = nx.shortest_path(self.graph, source=source_cmd, target=target_cmd, weight='weight')
                    # Optimize the path by removing redundant commands
                    optimized_path = self._optimize_command_chain(path, task_description)
                    logger.info(f"Found path from {source_cmd} to {target_cmd}: {' | '.join(optimized_path)}")
                    return optimized_path if len(optimized_path) <= max_length else []
                except nx.NetworkXNoPath:
                    logger.info(f"No path found from {source_cmd} to {target_cmd}")
                    return []
            elif task_description:
                # Find commands that might accomplish the task
                target_candidates = self._find_commands_for_task(task_description)
                
                # Try to find a direct command that accomplishes the task
                direct_commands = [cmd for cmd in target_candidates if self._command_accomplishes_task(cmd, task_description)]
                if direct_commands and direct_commands[0] != source_cmd:
                    # If source_cmd is needed to provide input to the direct command
                    if self._commands_compatible(source_cmd, direct_commands[0]):
                        logger.info(f"Task can be accomplished with command chain: {source_cmd} | {direct_commands[0]}")
                        return [source_cmd, direct_commands[0]]
                    else:
                        logger.info(f"Task can be accomplished with single command: {direct_commands[0]}")
                        return [direct_commands[0]]
                
                # Find paths to target candidates
                paths = []
                for target in target_candidates:
                    if target == source_cmd:
                        continue  # Skip paths to self
                        
                    try:
                        path = nx.shortest_path(self.graph, source=source_cmd, target=target, weight='weight')
                        if len(path) <= max_length:
                            # Calculate the total weight of this path
                            total_weight = sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                            paths.append((path, total_weight))
                    except nx.NetworkXNoPath:
                        continue
                        
                if not paths:
                    logger.info(f"No suitable paths found for task: {task_description}")
                    return []
                    
                # Sort paths by length first (shorter is better), then by weight
                paths.sort(key=lambda x: (len(x[0]), x[1]))
                
                # Take top 3 paths
                best_paths = [path for path, _ in paths[:3]]
                
                # Optimize each path
                optimized_paths = [self._optimize_command_chain(path, task_description) for path in best_paths]
                
                # Filter out empty paths and deduplicate
                unique_paths = []
                seen = set()
                for path in optimized_paths:
                    path_str = " | ".join(path)
                    if path and path_str not in seen:
                        unique_paths.append(path)
                        seen.add(path_str)
                
                logger.info(f"Found {len(unique_paths)} suitable paths for task: {task_description}")
                return unique_paths if len(unique_paths) > 0 else []
            else:
                # Find all possible chains starting from source
                chains = []
                for target in self.graph.nodes:
                    if target != source_cmd:
                        try:
                            path = nx.shortest_path(self.graph, source=source_cmd, target=target, weight='weight')
                            if 2 <= len(path) <= max_length:
                                # Optimize the path
                                optimized_path = self._optimize_command_chain(path, None)
                                if optimized_path and len(optimized_path) >= 2:
                                    chains.append(optimized_path)
                        except nx.NetworkXNoPath:
                            continue
                
                # Sort chains by length first, then by total weight
                chains.sort(key=lambda p: (len(p), sum(self.graph[u][v]['weight'] for u, v in zip(p[:-1], p[1:]))))
                
                # Return the top chains
                logger.info(f"Found {len(chains[:3])} general chains starting from {source_cmd}")
                return chains[:3] if chains else []
                
        except Exception as e:
            logger.error(f"Error finding command chain: {e}")
            return []
    
    def _find_commands_for_task(self, task_description):
        """Find commands that match a task description."""
        try:
            if not task_description:
                return []
                
            task_words = set(self._safe_lower(task_description).split())
            
            # Score each command based on word match in description
            scores = {}
            for cmd, metadata in self.command_metadata.items():
                desc_words = set(self._safe_lower(metadata['description']).split())
                examples_words = set(self._safe_lower(metadata['examples']).split())
                
                # Calculate word overlap with better error handling
                desc_overlap = len(task_words.intersection(desc_words)) if desc_words else 0
                examples_overlap = len(task_words.intersection(examples_words)) if examples_words else 0
                
                # Combine scores
                scores[cmd] = desc_overlap * 2 + examples_overlap
            
            # Get top matches
            top_commands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [cmd for cmd, score in top_commands[:10] if score > 0]
            
        except Exception as e:
            logger.warning(f"Error finding commands for task: {e}")
            return []
    
    def get_common_chains(self, top_n=10):
        """
        Get the most commonly used command chains.
        
        Args:
            top_n: Number of chains to return
            
        Returns:
            list: List of (command_pair, count) tuples
        """
        try:
            sorted_chains = sorted(self.command_chains.items(), key=lambda x: x[1], reverse=True)
            return sorted_chains[:top_n]
        except Exception as e:
            logger.warning(f"Error getting common chains: {e}")
            return []
    
    def recommend_next_command(self, current_command, task_description=None):
        """
        Recommend the next command to chain with the current command.
        
        Args:
            current_command: The current command
            task_description: Optional task description to guide recommendations
            
        Returns:
            list: List of recommended next commands
        """
        try:
            if current_command not in self.graph:
                logger.warning(f"Command '{current_command}' not found in graph")
                return []
                
            # Get all neighbors
            neighbors = list(self.graph.successors(current_command))
            
            if not neighbors:
                return []
                
            # Score neighbors based on edge weight and optionally task relevance
            scores = {}
            for neighbor in neighbors:
                # Base score is inverse of edge weight (lower weight = higher preference)
                edge_weight = self.graph.edges[current_command, neighbor]['weight']
                edge_count = self.graph.edges[current_command, neighbor].get('count', 0)
                
                # Start with base score from edge weight
                scores[neighbor] = 1.0 / edge_weight
                
                # Boost based on usage count
                scores[neighbor] *= (1.0 + np.log1p(edge_count))
                
                # If task description provided, adjust score based on relevance
                if task_description:
                    task_words = set(self._safe_lower(task_description).split())
                    neighbor_meta = self.command_metadata[neighbor]
                    
                    # Check word overlap with command description
                    desc_words = set(self._safe_lower(neighbor_meta['description']).split())
                    examples_words = set(self._safe_lower(neighbor_meta['examples']).split())
                    
                    # Calculate word overlap
                    desc_overlap = len(task_words.intersection(desc_words))
                    examples_overlap = len(task_words.intersection(examples_words))
                    
                    # Boost score based on relevance
                    scores[neighbor] *= (1.0 + 0.5 * desc_overlap + 0.3 * examples_overlap)
            
            # Sort by score
            sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return just the command names
            logger.info(f"Recommending {len(sorted_recommendations[:5])} commands after {current_command}")
            return [cmd for cmd, _ in sorted_recommendations[:5]]
            
        except Exception as e:
            logger.warning(f"Error recommending next command: {e}")
            return []
    
    def format_command_chain(self, chain, include_tips=True):
        """
        Format a command chain as a pipe-separated string.
        
        Args:
            chain: List of commands
            include_tips: Whether to include optimization tips
            
        Returns:
            str: Formatted command chain
        """
        try:
            if not chain or len(chain) < 2:
                return ""
            
            # For display purposes, we'll format it simply
            basic_format = " | ".join(chain)
            
            # Check for common patterns that could be optimized
            if include_tips:
                optimizations = {
                    "cat | grep": "Consider using 'grep pattern file' directly",
                    "grep | wc -l": "Consider using 'grep -c pattern file'",
                    "sort | uniq": "Consider using 'sort -u'",
                    "cat | head": "Consider using 'head file' directly",
                    "cat | tail": "Consider using 'tail file' directly",
                    "ls | grep": "Consider using 'find . -name \"*pattern*\"' for more control",
                    "ps | grep": "Consider using 'pgrep pattern' for better process filtering",
                }
                
                for pattern, note in optimizations.items():
                    if pattern in basic_format:
                        return f"{basic_format} (Optimization tip: {note})"
            
            return basic_format
            
        except Exception as e:
            logger.warning(f"Error formatting command chain: {e}")
            return " | ".join(chain) if chain else ""
    
    def visualize_graph(self, output_file="command_graph.png"):
        """
        Generate a visualization of the command graph.
        
        Args:
            output_file: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 20))
            
            # Extract subgraph of most frequently used commands
            common_chains = self.get_common_chains(top_n=30)
            common_commands = set()
            for (src, tgt), count in common_chains:
                common_commands.add(src)
                common_commands.add(tgt)
            
            if common_commands:
                subgraph = self.graph.subgraph(common_commands)
            else:
                # If no common chains yet, take most connected nodes
                degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
                top_nodes = [node for node, _ in degrees[:30]]
                subgraph = self.graph.subgraph(top_nodes)
            
            # Create the layout
            pos = nx.spring_layout(subgraph)
            
            # Draw edges with width based on count
            edge_widths = []
            for u, v in subgraph.edges():
                count = subgraph.edges[u, v].get('count', 0)
                edge_widths.append(1 + count * 0.5)
            
            # Draw the graph
            nx.draw_networkx_nodes(subgraph, pos, node_size=1000, node_color='lightblue')
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.7, edge_color='gray')
            nx.draw_networkx_labels(subgraph, pos, font_size=10)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, format='png', dpi=300)
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error generating graph visualization: {e}")
            return False