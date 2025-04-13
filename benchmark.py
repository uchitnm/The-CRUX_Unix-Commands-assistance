"""
Benchmark script to evaluate the query optimization system.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from agent_system import AgentSystem
from main import load_command_data, prepare_chunks, create_embedding_index

# Test queries with known ground truth commands
TEST_CASES = [
    {"query": "How to list files in a directory", "ground_truth": "ls"},
    {"query": "Search for text in files", "ground_truth": "grep"},
    {"query": "Show disk space usage", "ground_truth": "df"},
    {"query": "How to compress files", "ground_truth": "tar"},
    {"query": "Check running processes", "ground_truth": "ps"},
    {"query": "Display the contents of a file", "ground_truth": "cat"},
    {"query": "Download files from the internet", "ground_truth": "wget"},
    {"query": "Find files by name", "ground_truth": "find"},
    {"query": "Count lines words and characters", "ground_truth": "wc"},
    {"query": "Move files to a different location", "ground_truth": "mv"},
    {"query": "Change file permissions", "ground_truth": "chmod"},
    {"query": "Create a new directory", "ground_truth": "mkdir"},
    {"query": "Connect to a remote server", "ground_truth": "ssh"},
    {"query": "See available disk space", "ground_truth": "df"},
    {"query": "How to extract a zip file", "ground_truth": "unzip"},
]

def run_benchmark():
    """Run the benchmark on test cases."""
    # Load API key
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Initialize components
    print("Initializing components...")
    df = load_command_data()
    chunk_metadata, chunks = prepare_chunks(df)
    index, embedding_model = create_embedding_index(chunks)
    client = genai.Client(api_key=GEMINI_API_KEY)
    agent_system = AgentSystem(client)
    
    # Initialize the query optimizer
    agent_system.initialize_optimizer(df, index, embedding_model, chunk_metadata)
    
    # Run tests
    print("\nRunning benchmark...")
    results = []
    
    for test_case in tqdm(TEST_CASES):
        query = test_case["query"]
        ground_truth = test_case["ground_truth"]
        
        # Test without optimization
        start_time = time.time()
        retrieved_commands, _ = agent_system.retrieval_agent(
            df, query, {"reformulated_query": query}, index, embedding_model, chunk_metadata
        )
        baseline_time = time.time() - start_time
        
        baseline_hit = False
        for cmd in retrieved_commands:
            if cmd.get("Command") == ground_truth:
                baseline_hit = True
                break
        
        # Test with optimization
        start_time = time.time()
        optimized_query, metrics, all_results = agent_system.query_optimizer.optimize_query(
            query, ground_truth
        )
        optimization_time = time.time() - start_time
        
        # Get results with optimized query
        retrieved_commands, _ = agent_system.retrieval_agent(
            df, optimized_query, {"reformulated_query": optimized_query}, index, embedding_model, chunk_metadata
        )
        
        optimized_hit = False
        for cmd in retrieved_commands:
            if cmd.get("Command") == ground_truth:
                optimized_hit = True
                break
        
        # Store results
        results.append({
            "query": query,
            "ground_truth": ground_truth,
            "baseline_hit": baseline_hit,
            "optimized_hit": optimized_hit,
            "optimized_query": optimized_query,
            "baseline_time": baseline_time,
            "optimization_time": optimization_time,
            "metrics": metrics
        })
    
    # Calculate summary statistics
    baseline_hits = sum(1 for r in results if r["baseline_hit"])
    optimized_hits = sum(1 for r in results if r["optimized_hit"])
    
    avg_baseline_time = np.mean([r["baseline_time"] for r in results])
    avg_optimization_time = np.mean([r["optimization_time"] for r in results])
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Baseline hits: {baseline_hits}/{len(TEST_CASES)} ({baseline_hits/len(TEST_CASES)*100:.1f}%)")
    print(f"Optimized hits: {optimized_hits}/{len(TEST_CASES)} ({optimized_hits/len(TEST_CASES)*100:.1f}%)")
    print(f"Average baseline time: {avg_baseline_time:.3f}s")
    print(f"Average optimization time: {avg_optimization_time:.3f}s")
    
    # Detailed results
    print("\n=== Detailed Results ===")
    for i, result in enumerate(results):
        print(f"\nCase {i+1}: {result['query']}")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Baseline hit: {result['baseline_hit']}")
        print(f"Optimized hit: {result['optimized_hit']}")
        print(f"Optimized query: {result['optimized_query']}")
    
    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
