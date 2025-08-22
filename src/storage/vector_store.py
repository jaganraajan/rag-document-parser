from pinecone import Pinecone
from typing import Iterable, List, Dict
# from .id_strategy import IDStrategy
import uuid
import os
import time
from src.observability.instruments import (
    queries_total, query_errors_total,
    query_end_to_end_seconds, vector_search_seconds,
    retrieval_result_count, retrieval_top_score,
    retrieval_any_result_total, no_result_queries_total,
    upsert_records_total, upsert_errors_total, upsert_batch_seconds,
    safe_attrs,
)

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
        records.append({
            "id": str(uuid.uuid4()),
            "chunk_text": c.get("chunk"),      # field mapped to 'text'
            **metadata
        })
    return records

def store_vectors(chunks: Iterable[Dict]):
    index = ensure_index()
    records = to_records(chunks)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        t0 = time.perf_counter()
        try:
            index.upsert_records(NAMESPACE, batch)
            elapsed = time.perf_counter() - t0
            upsert_batch_seconds.record(elapsed, safe_attrs({"index": INDEX_NAME}))
            upsert_records_total.add(len(batch), safe_attrs({"index": INDEX_NAME}))
        except Exception as e:
            upsert_errors_total.add(1, safe_attrs({"error.type": e.__class__.__name__}))
            raise


def semantic_query(query: str, top_k: int = 5):
    index = ensure_index()
    queries_total.add(1, safe_attrs({"top_k": str(top_k)}))
    q_start = time.perf_counter()
    try:
        vs_start = time.perf_counter()
        results = index.search(
            namespace=NAMESPACE,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            }
        )
        vector_search_seconds.record(time.perf_counter() - vs_start, safe_attrs({"top_k": str(top_k)}))

        matches = []
        if isinstance(results, dict):
            matches = results.get("matches", [])

        # Result counts
        retrieval_result_count.add(len(matches), safe_attrs({"top_k": str(top_k)}))
        if len(matches) > 0:
            retrieval_any_result_total.add(1, safe_attrs({}))
            # try to compute a top score if present
            try:
                scores = [m.get("score") for m in matches if isinstance(m, dict) and isinstance(m.get("score"), (int, float))]
                if scores:
                    retrieval_top_score.record(max(scores), safe_attrs({}))
            except Exception:
                pass
        else:
            no_result_queries_total.add(1, safe_attrs({}))

        query_end_to_end_seconds.record(time.perf_counter() - q_start, safe_attrs({"status": "success", "top_k": str(top_k)}))
        return results
    except Exception as e:
        query_errors_total.add(1, safe_attrs({"error.type": e.__class__.__name__}))
        query_end_to_end_seconds.record(time.perf_counter() - q_start, safe_attrs({"status": "error", "top_k": str(top_k)}))
        raise
