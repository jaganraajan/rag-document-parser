from pinecone import Pinecone
from typing import Iterable, List, Dict
# from .id_strategy import IDStrategy
import uuid
import os
from .corpus_store import ChunkResult, bm25_search, get_corpus_store

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY env var")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "philosophy-rag"
EMBED_MODEL = "llama-text-embed-v2"  # keep consistent everywhere
NAMESPACE = "__default__"  # or "philosophy" if multi-tenant mode is needed

def ensure_index():
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": EMBED_MODEL,
                "field_map": {"text": "chunk_text"}   # tells pipeline which field to embed
            }
        )
    return pc.Index(INDEX_NAME)

def _flatten_metadata(meta: Dict[str, any]) -> Dict[str, any]:
    flat = {}
    for k, v in (meta or {}).items():
        if v in (None, "", [], {}):
            continue
        key = f"meta_{k}"
        if isinstance(v, (str, int, float, bool)):
            flat[key] = v
        elif isinstance(v, list):
            flat[key] = [str(item) for item in v if item not in (None, "")]
        else:
            flat[key] = str(v)

    print(f"Flattened metadata: {flat}")
    return flat

def to_records(chunks: Iterable[Dict]) -> List[Dict]:
    """
    chunks each: {
      "id": str,
      "chunk_text": str,
      "metadata": {...}
    }
    Returns records ready for managed embedding (no 'values').
    """
    records = []
    for c in chunks:
        metadata = _flatten_metadata(c.get("metadata", {}))
        # Use existing ID if provided, otherwise generate new one
        chunk_id = c.get("id", str(uuid.uuid4()))
        records.append({
            "id": chunk_id,
            "chunk_text": c.get("chunk"),      # field mapped to 'text'
            **metadata
        })
    return records

def store_vectors(chunks: Iterable[Dict]):
    index = ensure_index()
    # print('index', index)
    # print(chunks[:2])
    records = to_records(chunks)
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        index.upsert_records(NAMESPACE, batch)


def semantic_query(query: str):
    index = ensure_index()
    print('search for query:', query)
    # Search the dense index
    results = index.search(
        namespace=NAMESPACE,
        query={
            "inputs": {
                "text": query
            },
            "top_k": 5
        }
    )

    return results


def vector_search(query: str, top_k: int = 5) -> List[ChunkResult]:
    """
    Wrapper around semantic_query that returns normalized ChunkResult objects.
    """
    try:
        results = semantic_query(query)
        chunk_results = []
        
        # Extract matches from Pinecone results
        matches = results.get('matches', [])
        if not matches:
            return []
        
        # Collect scores for normalization
        scores = [match.get('score', 0.0) for match in matches[:top_k]]
        if len(scores) <= 1:
            norm_scores = scores
        else:
            # Min-max normalization
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                norm_scores = [1.0] * len(scores)
        
        # Convert to ChunkResult format
        for i, match in enumerate(matches[:top_k]):
            chunk_results.append(ChunkResult(
                id=match.get('id', ''),
                text=match.get('metadata', {}).get('chunk_text', ''),
                score=norm_scores[i],
                source="vector",
                metadata=match.get('metadata', {})
            ))
        
        return chunk_results
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []


def normalize_scores(scores: List[float]) -> List[float]:
    """Apply min-max normalization to a list of scores."""
    if len(scores) <= 1:
        return scores
    
    min_score, max_score = min(scores), max(scores)
    if max_score > min_score:
        return [(s - min_score) / (max_score - min_score) for s in scores]
    else:
        return [1.0] * len(scores)


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> List[ChunkResult]:
    """
    Perform hybrid search combining BM25 and vector similarity.
    
    Args:
        query: Search query
        top_k: Number of results to return
        alpha: Blending weight (0.0 = pure BM25, 1.0 = pure vector)
    
    Returns:
        List of ChunkResult objects with hybrid scores
    """
    try:
        # Get results from both methods (fetch more to have better candidate pool)
        vector_results = vector_search(query, top_k * 2)
        bm25_results = bm25_search(query, top_k * 2)
        
        # Create a unified candidate set by ID
        candidates = {}
        
        # Add vector results
        for result in vector_results:
            candidates[result.id] = {
                'id': result.id,
                'text': result.text,
                'metadata': result.metadata,
                'vector_score': result.score,
                'bm25_score': 0.0
            }
        
        # Add/update with BM25 results
        for result in bm25_results:
            if result.id in candidates:
                candidates[result.id]['bm25_score'] = result.score
            else:
                candidates[result.id] = {
                    'id': result.id,
                    'text': result.text,
                    'metadata': result.metadata,
                    'vector_score': 0.0,
                    'bm25_score': result.score
                }
        
        # Normalize scores within the candidate set
        if candidates:
            vector_scores = [c['vector_score'] for c in candidates.values()]
            bm25_scores = [c['bm25_score'] for c in candidates.values()]
            
            norm_vector_scores = normalize_scores(vector_scores)
            norm_bm25_scores = normalize_scores(bm25_scores)
            
            # Calculate hybrid scores and create results
            hybrid_results = []
            for i, (cand_id, cand) in enumerate(candidates.items()):
                hybrid_score = alpha * norm_vector_scores[i] + (1 - alpha) * norm_bm25_scores[i]
                hybrid_results.append(ChunkResult(
                    id=cand['id'],
                    text=cand['text'],
                    score=hybrid_score,
                    source="hybrid",
                    metadata=cand['metadata']
                ))
            
            # Sort by hybrid score and return top-k
            hybrid_results.sort(key=lambda x: x.score, reverse=True)
            return hybrid_results[:top_k]
        
        return []
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []
