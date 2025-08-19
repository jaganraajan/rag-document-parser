"""
Hybrid search implementation combining dense semantic search with sparse keyword search.

This module provides:
- Hybrid search combining dense and sparse retrieval
- Result merging and deduplication by ID
- Reranking with multiple relevance signals
- Configurable scoring weights
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
import math

# Import search functions
from src.storage.vector_store import semantic_query
from src.storage.sparse_store import sparse_store

# Scoring weights - configurable constants
W_DENSE = 0.5    # Weight for dense semantic score
W_SPARSE = 0.3   # Weight for sparse keyword score  
W_OVERLAP = 0.2  # Weight for lexical overlap score


def calculate_lexical_overlap(query: str, text: str) -> float:
    """
    Calculate lexical overlap ratio between query and text.
    
    Returns: unique query terms present / total unique query terms
    """
    if not query or not text:
        return 0.0
    
    # Tokenize using same method as sparse store
    query_tokens = set(sparse_store.tokenize(query))
    text_tokens = set(sparse_store.tokenize(text))
    
    if not query_tokens:
        return 0.0
    
    overlap = len(query_tokens.intersection(text_tokens))
    return overlap / len(query_tokens)


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range using min-max normalization.
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        List of normalized scores (0-1 range)
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle case where all scores are the same
    if max_score == min_score:
        return [1.0] * len(scores)
    
    # Min-max normalization
    normalized = []
    for score in scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append(norm_score)
    
    return normalized


def merge_results(dense_results: List[Dict[str, Any]], 
                 sparse_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Merge dense and sparse results by ID, aggregating scores.
    
    Args:
        dense_results: Results from semantic search
        sparse_results: Results from sparse search
        
    Returns:
        Dictionary mapping result ID to merged result with both scores
    """
    merged = {}
    
    # Process dense results
    for result in dense_results:
        result_id = result['id']
        merged[result_id] = {
            'id': result_id,
            'text': result['text'],
            'metadata': result.get('metadata', {}),
            'dense_score': result['score'],
            'sparse_score': 0.0,
            'source': 'dense'
        }
    
    # Process sparse results
    for result in sparse_results:
        result_id = result['id']
        if result_id in merged:
            # Update existing result with sparse score
            merged[result_id]['sparse_score'] = result['score']
            merged[result_id]['source'] = 'both'
        else:
            # Add new sparse-only result
            merged[result_id] = {
                'id': result_id,
                'text': result['text'],
                'metadata': result.get('metadata', {}),
                'dense_score': 0.0,
                'sparse_score': result['score'],
                'source': 'sparse'
            }
    
    return merged


def rerank_results(query: str, merged_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank merged results using weighted combination of multiple signals.
    
    Args:
        query: Original search query
        merged_results: Dictionary of merged results by ID
        
    Returns:
        List of reranked results with final relevance scores
    """
    if not merged_results:
        return []
    
    results_list = list(merged_results.values())
    
    # Extract and normalize dense scores
    dense_scores = [r['dense_score'] for r in results_list]
    normalized_dense = normalize_scores([s for s in dense_scores if s > 0])
    
    # Extract and normalize sparse scores  
    sparse_scores = [r['sparse_score'] for r in results_list]
    normalized_sparse = normalize_scores([s for s in sparse_scores if s > 0])
    
    # Create mappings for normalized scores
    dense_score_map = {}
    sparse_score_map = {}
    
    # Map normalized dense scores
    dense_idx = 0
    for i, result in enumerate(results_list):
        if result['dense_score'] > 0:
            dense_score_map[i] = normalized_dense[dense_idx] if normalized_dense else 0.0
            dense_idx += 1
        else:
            dense_score_map[i] = 0.0
    
    # Map normalized sparse scores
    sparse_idx = 0
    for i, result in enumerate(results_list):
        if result['sparse_score'] > 0:
            sparse_score_map[i] = normalized_sparse[sparse_idx] if normalized_sparse else 0.0
            sparse_idx += 1
        else:
            sparse_score_map[i] = 0.0
    
    # Calculate final relevance scores
    reranked_results = []
    for i, result in enumerate(results_list):
        # Get normalized component scores
        norm_dense = dense_score_map[i]
        norm_sparse = sparse_score_map[i]
        
        # Calculate lexical overlap
        overlap_score = calculate_lexical_overlap(query, result['text'])
        
        # Weighted combination
        relevance_score = (
            W_DENSE * norm_dense + 
            W_SPARSE * norm_sparse + 
            W_OVERLAP * overlap_score
        )
        
        # Create final result
        final_result = {
            'id': result['id'],
            'text': result['text'],
            'metadata': result['metadata'],
            'relevance_score': relevance_score,
            'score_breakdown': {
                'dense_score': result['dense_score'],
                'sparse_score': result['sparse_score'],
                'normalized_dense': norm_dense,
                'normalized_sparse': norm_sparse,
                'overlap_score': overlap_score,
                'final_score': relevance_score
            },
            'source': result['source']
        }
        reranked_results.append(final_result)
    
    # Sort by relevance score (descending)
    reranked_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return reranked_results


def hybrid_search(query: str, top_k_dense: int = 5, top_k_sparse: int = 20) -> List[Dict[str, Any]]:
    """
    Execute hybrid search combining dense and sparse retrieval with reranking.
    
    Args:
        query: Search query string
        top_k_dense: Number of dense results to retrieve (default: 5)
        top_k_sparse: Number of sparse results to retrieve (default: 20)
        
    Returns:
        List of reranked results with relevance scores and breakdown
    """
    print(f"Executing hybrid search for query: '{query}'")
    
    # Execute dense semantic search
    try:
        print("Executing dense semantic search...")
        dense_raw = semantic_query(query, top_k=top_k_dense)
        
        # Parse dense results - handle Pinecone response format
        dense_results = []
        hits = dense_raw.get("result", {}).get("hits", [])
        for hit in hits:
            fields = hit.get("fields", {}) or {}
            text = fields.get("chunk_text") or fields.get("text") or ""
            
            # Extract metadata
            metadata = {}
            for k, v in fields.items():
                if k.startswith("meta_"):
                    metadata[k[5:]] = v  # strip 'meta_' prefix
            
            dense_results.append({
                "id": hit.get("_id"),
                "score": hit.get("_score", 0.0),
                "text": text,
                "metadata": metadata
            })
        
        print(f"Dense search returned {len(dense_results)} results")
        
    except Exception as e:
        print(f"Dense search failed: {e}")
        dense_results = []
    
    # Execute sparse keyword search  
    try:
        print("Executing sparse keyword search...")
        sparse_results = sparse_store.sparse_query(query, top_k=top_k_sparse)
        print(f"Sparse search returned {len(sparse_results)} results")
        
    except Exception as e:
        print(f"Sparse search failed: {e}")
        sparse_results = []
    
    # Handle case where both searches failed
    if not dense_results and not sparse_results:
        print("Both dense and sparse searches failed")
        return []
    
    # Handle graceful degradation
    if not sparse_results:
        print("Sparse search unavailable, using dense results only")
        # Return dense results with relevance scores
        reranked = []
        dense_scores = [r['score'] for r in dense_results]
        normalized_dense = normalize_scores(dense_scores)
        
        for i, result in enumerate(dense_results):
            overlap_score = calculate_lexical_overlap(query, result['text'])
            relevance_score = W_DENSE * normalized_dense[i] + W_OVERLAP * overlap_score
            
            reranked.append({
                'id': result['id'],
                'text': result['text'],
                'metadata': result['metadata'],
                'relevance_score': relevance_score,
                'score_breakdown': {
                    'dense_score': result['score'],
                    'sparse_score': 0.0,
                    'normalized_dense': normalized_dense[i],
                    'normalized_sparse': 0.0,
                    'overlap_score': overlap_score,
                    'final_score': relevance_score
                },
                'source': 'dense_only'
            })
        
        return reranked
    
    if not dense_results:
        print("Dense search unavailable, using sparse results only")
        # Return sparse results with relevance scores
        reranked = []
        sparse_scores = [r['score'] for r in sparse_results]
        normalized_sparse = normalize_scores(sparse_scores)
        
        for i, result in enumerate(sparse_results):
            overlap_score = calculate_lexical_overlap(query, result['text'])
            relevance_score = W_SPARSE * normalized_sparse[i] + W_OVERLAP * overlap_score
            
            reranked.append({
                'id': result['id'],
                'text': result['text'],
                'metadata': result['metadata'],
                'relevance_score': relevance_score,
                'score_breakdown': {
                    'dense_score': 0.0,
                    'sparse_score': result['score'],
                    'normalized_dense': 0.0,
                    'normalized_sparse': normalized_sparse[i],
                    'overlap_score': overlap_score,
                    'final_score': relevance_score
                },
                'source': 'sparse_only'
            })
        
        return reranked
    
    # Merge results from both searches
    print("Merging and reranking results...")
    merged_results = merge_results(dense_results, sparse_results)
    
    # Rerank using weighted combination
    final_results = rerank_results(query, merged_results)
    
    print(f"Final hybrid search returned {len(final_results)} reranked results")
    
    return final_results