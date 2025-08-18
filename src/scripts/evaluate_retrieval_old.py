#!/usr/bin/env python3
"""
RAG Retrieval Evaluation Script

This script evaluates the retrieval accuracy of the RAG system by:
- Loading an evaluation dataset with queries and expected results
- Using the semantic_query function to retrieve top-k results for each query
- Computing retrieval metrics: Precision@k, Recall@k, and Hit Rate
- Displaying a summary table of results

Usage Examples:
    # Use built-in sample dataset
    python src/scripts/evaluate_retrieval.py

    # Use custom evaluation file
    python src/scripts/evaluate_retrieval.py --eval-file path/to/evaluation.json --k 10

    # Evaluate with different k values
    python src/scripts/evaluate_retrieval.py --k 3 --verbose

    # Test mode (when Pinecone is not available)
    python src/scripts/evaluate_retrieval.py --test-mode --verbose
"""

import argparse
import json
import csv
import os
import sys
from typing import List, Dict, Any, Set, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Conditional imports - only import vector store if not in test mode
semantic_query = None

def import_semantic_query():
    """Import semantic_query function when needed."""
    global semantic_query
    if semantic_query is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from src.storage.vector_store import semantic_query
        except Exception as e:
            print(f"Warning: Could not import semantic_query: {e}")
            print("Use --test-mode to run with mock data")
            raise


# Sample evaluation dataset for testing
SAMPLE_EVALUATION_DATA = [
    {
        "query": "existential meaning",
        "expected_chunks": [
            "existential philosophy and the search for meaning",
            "meaning of life in existential thought",
            "existential crisis and finding purpose"
        ],
        "expected_ids": []  # Can be empty if using text matching
    },
    {
        "query": "consciousness and awareness",
        "expected_chunks": [
            "consciousness in philosophical discourse",
            "awareness and perception",
            "conscious experience and qualia"
        ],
        "expected_ids": []
    },
    {
        "query": "ethics and morality",
        "expected_chunks": [
            "ethical frameworks and moral philosophy",
            "moral reasoning and ethical decisions",
            "virtue ethics and moral character"
        ],
        "expected_ids": []
    },
    {
        "query": "free will and determinism",
        "expected_chunks": [
            "free will versus determinism debate",
            "deterministic universe and choice",
            "libertarian free will theory"
        ],
        "expected_ids": []
    },
    {
        "query": "knowledge and epistemology",
        "expected_chunks": [
            "epistemological theories of knowledge",
            "knowledge acquisition and justification",
            "skepticism and certainty in knowledge"
        ],
        "expected_ids": []
    }
]


def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from JSON or CSV file.
    
    Expected format for JSON:
    [
        {
            "query": "search query",
            "expected_chunks": ["chunk text 1", "chunk text 2"],
            "expected_ids": ["id1", "id2"]  # optional
        }
    ]
    
    Expected format for CSV:
    query,expected_chunks,expected_ids
    "search query","chunk1;chunk2","id1;id2"
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_ext == '.csv':
        evaluation_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse semicolon-separated chunks and ids
                expected_chunks = [chunk.strip() for chunk in row['expected_chunks'].split(';') if chunk.strip()]
                expected_ids = [id.strip() for id in row.get('expected_ids', '').split(';') if id.strip()]
                
                evaluation_data.append({
                    'query': row['query'],
                    'expected_chunks': expected_chunks,
                    'expected_ids': expected_ids
                })
        return evaluation_data
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .json or .csv")


def extract_results_info(search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract relevant information from Pinecone search results.
    
    Returns list of dicts with 'id', 'score', and 'text' keys.
    """
    results = []
    
    # Handle different possible result structures from Pinecone
    if 'matches' in search_results:
        # Standard Pinecone format
        for match in search_results['matches']:
            results.append({
                'id': match.get('id', ''),
                'score': match.get('score', 0.0),
                'text': match.get('metadata', {}).get('chunk_text', '')
            })
    elif 'result' in search_results and 'hits' in search_results['result']:
        # Alternative format seen in test_search.py comment
        for hit in search_results['result']['hits']:
            results.append({
                'id': hit.get('_id', ''),
                'score': hit.get('_score', 0.0),
                'text': hit.get('fields', {}).get('chunk_text', '')
            })
    else:
        # Try to handle unknown format gracefully
        print(f"Warning: Unknown result format: {search_results}")
    
    return results


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity based on overlapping words.
    Returns a score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def find_relevant_results(retrieved_results: List[Dict[str, Any]], 
                         expected_chunks: List[str],
                         expected_ids: List[str],
                         similarity_threshold: float = 0.3) -> Set[int]:
    """
    Find which retrieved results are relevant based on expected chunks/ids.
    
    Returns set of indices of relevant results.
    """
    relevant_indices = set()
    
    for i, result in enumerate(retrieved_results):
        # Check if result ID matches expected IDs
        if expected_ids and result['id'] in expected_ids:
            relevant_indices.add(i)
            continue
        
        # Check text similarity with expected chunks
        result_text = result['text']
        for expected_chunk in expected_chunks:
            similarity = calculate_text_similarity(result_text, expected_chunk)
            if similarity >= similarity_threshold:
                relevant_indices.add(i)
                break
    
    return relevant_indices


def calculate_metrics(relevant_indices: Set[int], 
                     total_retrieved: int,
                     total_expected: int,
                     k: int) -> Dict[str, float]:
    """
    Calculate retrieval metrics.
    
    Args:
        relevant_indices: Set of indices of relevant retrieved results
        total_retrieved: Number of results retrieved (should be <= k)
        total_expected: Number of expected relevant results
        k: The k value for top-k evaluation
    
    Returns:
        Dict with precision_at_k, recall_at_k, hit_rate metrics
    """
    num_relevant_retrieved = len(relevant_indices)
    
    # Precision@k: relevant_retrieved / min(total_retrieved, k)
    precision_at_k = num_relevant_retrieved / min(total_retrieved, k) if min(total_retrieved, k) > 0 else 0.0
    
    # Recall@k: relevant_retrieved / total_expected
    recall_at_k = num_relevant_retrieved / total_expected if total_expected > 0 else 0.0
    
    # Hit Rate: 1 if any relevant result found, 0 otherwise
    hit_rate = 1.0 if num_relevant_retrieved > 0 else 0.0
    
    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'hit_rate': hit_rate,
        'relevant_retrieved': num_relevant_retrieved,
        'total_retrieved': total_retrieved,
        'total_expected': total_expected
    }


def evaluate_single_query(query: str, 
                         expected_chunks: List[str],
                         expected_ids: List[str],
                         k: int = 5,
                         similarity_threshold: float = 0.3,
                         verbose: bool = False,
                         test_mode: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single query against expected results.
    """
    if verbose:
        print(f"\nEvaluating query: '{query}'")
    
    try:
        if test_mode:
            # Generate mock results for testing - make some actually match
            mock_results = []
            for i in range(min(k, 3)):
                if i == 0 and expected_chunks:
                    # Make first result somewhat relevant
                    mock_text = f"This is about {expected_chunks[0].split()[0]} and related concepts in philosophy"
                elif i == 1 and len(expected_chunks) > 1:
                    # Make second result partially relevant  
                    mock_text = f"Discussion of {expected_chunks[1].split()[-1]} in modern philosophical thought"
                else:
                    # Make other results less relevant
                    mock_text = f'Mock result {i+1} discussing various topics related to {query.split()[0] if query.split() else "concepts"}'
                
                mock_results.append({
                    'id': f'mock_id_{i}', 
                    'score': 0.8 - i*0.1, 
                    'text': mock_text
                })
            
            retrieved_results = mock_results
            if verbose:
                print(f"[TEST MODE] Generated {len(retrieved_results)} mock results")
        else:
            # Import semantic_query function
            if semantic_query is None:
                import_semantic_query()
            
            # Retrieve results using semantic_query
            search_results = semantic_query(query)
            retrieved_results = extract_results_info(search_results)
            
            # Limit to top-k results
            retrieved_results = retrieved_results[:k]
        
        if verbose and not test_mode:
            print(f"Retrieved {len(retrieved_results)} results")
            for i, result in enumerate(retrieved_results):
                print(f"  {i+1}. Score: {result['score']:.3f}, Text: {result['text'][:100]}...")
        
        # Find relevant results
        relevant_indices = find_relevant_results(
            retrieved_results, expected_chunks, expected_ids, similarity_threshold
        )
        
        if verbose and relevant_indices:
            print(f"Found {len(relevant_indices)} relevant results at indices: {sorted(relevant_indices)}")
        
        # Calculate metrics
        metrics = calculate_metrics(
            relevant_indices, 
            len(retrieved_results),
            len(expected_chunks) + len(expected_ids),
            k
        )
        
        metrics['query'] = query
        metrics['success'] = True
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating query '{query}': {e}")
        return {
            'query': query,
            'success': False,
            'error': str(e),
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'hit_rate': 0.0,
            'relevant_retrieved': 0,
            'total_retrieved': 0,
            'total_expected': len(expected_chunks) + len(expected_ids)
        }


def print_summary_table(results: List[Dict[str, Any]], k: int):
    """
    Print a formatted summary table of evaluation results.
    """
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("No successful evaluations to summarize.")
        return
    
    # Calculate overall metrics
    total_precision = sum(r['precision_at_k'] for r in successful_results)
    total_recall = sum(r['recall_at_k'] for r in successful_results)
    total_hit_rate = sum(r['hit_rate'] for r in successful_results)
    num_queries = len(successful_results)
    
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_hit_rate = total_hit_rate / num_queries
    
    # Print header
    print(f"\n{'='*80}")
    print(f"RAG RETRIEVAL EVALUATION SUMMARY (k={k})")
    print(f"{'='*80}")
    
    # Print per-query results
    print(f"{'Query':<30} {'Precision@k':<12} {'Recall@k':<10} {'Hit Rate':<9} {'Rel/Tot':<8}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*9} {'-'*8}")
    
    for result in successful_results:
        query_short = result['query'][:28] + '..' if len(result['query']) > 30 else result['query']
        rel_tot = f"{result['relevant_retrieved']}/{result['total_retrieved']}"
        print(f"{query_short:<30} {result['precision_at_k']:<12.3f} {result['recall_at_k']:<10.3f} "
              f"{result['hit_rate']:<9.1f} {rel_tot:<8}")
    
    # Print overall metrics
    print(f"{'-'*70}")
    print(f"{'AVERAGE':<30} {avg_precision:<12.3f} {avg_recall:<10.3f} {avg_hit_rate:<9.3f}")
    
    # Print additional statistics
    print(f"\nDetailed Statistics:")
    print(f"  Total queries evaluated: {num_queries}")
    print(f"  Failed evaluations: {len(results) - num_queries}")
    print(f"  Average Precision@{k}: {avg_precision:.3f}")
    print(f"  Average Recall@{k}: {avg_recall:.3f}")
    print(f"  Average Hit Rate: {avg_hit_rate:.3f}")
    
    # Show failed evaluations if any
    failed_results = [r for r in results if not r.get('success', False)]
    if failed_results:
        print(f"\nFailed Evaluations:")
        for result in failed_results:
            print(f"  - {result['query']}: {result.get('error', 'Unknown error')}")


def save_sample_evaluation_file(file_path: str):
    """
    Save a sample evaluation dataset to a file for reference.
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_EVALUATION_DATA, f, indent=2, ensure_ascii=False)
        print(f"Sample evaluation dataset saved to: {file_path}")
    
    elif file_ext == '.csv':
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['query', 'expected_chunks', 'expected_ids'])
            for item in SAMPLE_EVALUATION_DATA:
                expected_chunks = ';'.join(item['expected_chunks'])
                expected_ids = ';'.join(item['expected_ids'])
                writer.writerow([item['query'], expected_chunks, expected_ids])
        print(f"Sample evaluation dataset (CSV) saved to: {file_path}")
    
    else:
        print(f"Unsupported format for sample file: {file_ext}. Use .json or .csv")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system retrieval accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--eval-file', '-f',
        type=str,
        help='Path to evaluation dataset file (JSON or CSV format)'
    )
    
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=5,
        help='Number of top results to evaluate (default: 5)'
    )
    
    parser.add_argument(
        '--similarity-threshold', '-t',
        type=float,
        default=0.3,
        help='Text similarity threshold for relevance (default: 0.3)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--save-sample',
        type=str,
        help='Save sample evaluation dataset to specified file and exit'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with mock results (useful when Pinecone is not available)'
    )
    
    args = parser.parse_args()
    
    # Handle save sample file option
    if args.save_sample:
        save_sample_evaluation_file(args.save_sample)
        return
    
    # Load evaluation data
    if args.eval_file:
        try:
            evaluation_data = load_evaluation_data(args.eval_file)
            print(f"Loaded {len(evaluation_data)} queries from {args.eval_file}")
        except Exception as e:
            print(f"Error loading evaluation file: {e}")
            return
    else:
        evaluation_data = SAMPLE_EVALUATION_DATA
        print(f"Using built-in sample dataset with {len(evaluation_data)} queries")
    
    if not evaluation_data:
        print("No evaluation data to process.")
        return
    
    print(f"Evaluating with k={args.k}, similarity_threshold={args.similarity_threshold}")
    if args.test_mode:
        print("Running in TEST MODE with mock results")
    
    # Evaluate each query
    results = []
    for item in evaluation_data:
        result = evaluate_single_query(
            query=item['query'],
            expected_chunks=item.get('expected_chunks', []),
            expected_ids=item.get('expected_ids', []),
            k=args.k,
            similarity_threshold=args.similarity_threshold,
            verbose=args.verbose,
            test_mode=args.test_mode
        )
        results.append(result)
    
    # Print summary
    print_summary_table(results, args.k)


if __name__ == '__main__':
    main()