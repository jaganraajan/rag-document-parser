import os
import uuid
from typing import Iterable, Dict, List, Any, Optional

from pinecone import Pinecone, ServerlessSpec
from .vector_store import NAMESPACE, INDEX_NAME as DENSE_INDEX_NAME, ensure_index as ensure_dense_index

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY for sparse store.")

pc = Pinecone(api_key=PINECONE_API_KEY)

SPARSE_INDEX_NAME = "philosophy-rag-sparse"
SPARSE_MODEL = "pinecone-sparse-english-v0"  # managed sparse encoder
DEFAULT_SPARSE_MODEL = "pinecone-sparse-english-v0"
SPARSE_MODEL_OPTIONS = [
    "pinecone-sparse-english-v0",
    "pinecone-sparse-multilingual-v0",
    "bm25-sparse"
]

def ensure_sparse_index():
    """
    Create (once) a serverless sparse index using the managed sparse model.
    Records must contain the field 'chunk_text' (per field_map).
    """
    if not pc.has_index(SPARSE_INDEX_NAME):
        pc.create_index_for_model(
            name=SPARSE_INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":SPARSE_MODEL,
                "field_map":{"text": "chunk_text"}
            }
        )
    return pc.Index(SPARSE_INDEX_NAME)

def _flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for k, v in (meta or {}).items():
        if v in (None, "", [], {}):
            continue
        key = f"meta_{k}"
        if isinstance(v, (str, int, float, bool)):
            flat[key] = v
        elif isinstance(v, list):
            flat[key] = [str(i) for i in v if i not in (None, "")]
        else:
            flat[key] = str(v)
    return flat

def to_sparse_records(chunks: Iterable[Dict]) -> List[Dict]:
    """
    Input chunks: each dict may have:
      {
        "id": optional str,
        "chunk_text" | "chunk" | "text": str,
        "metadata": dict
      }
    Output records (no 'values'; managed sparse embedding will infer):
      {"id": str, "chunk_text": "...", <flattened meta_*> }
    """
    records: List[Dict] = []
    for idx, c in enumerate(chunks):
        raw_text = c.get("chunk_text") or c.get("chunk") or c.get("text")
        if not raw_text:
            continue
        text = raw_text.strip()
        if not text:
            continue
        meta = _flatten_metadata(c.get("metadata", {}))
        rec_id = c.get("id") or str(uuid.uuid4())
        rec = {"id": rec_id, "chunk_text": text, **meta}
        records.append(rec)
    return records

def store_sparse_vectors(chunks: Iterable[Dict], batch_size: int = 100):
    """
    Upsert raw text records into sparse index; Pinecone applies sparse encoder.
    """
    index = ensure_sparse_index()
    records = to_sparse_records(chunks)
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        index.upsert_records(NAMESPACE, batch)
    stats = index.describe_index_stats()
    print("Sparse index stats:", stats)

def sparse_query(query_text: str, top_k: int = 10, filter: Optional[Dict] = None):
    """
    Query sparse index letting Pinecone embed query with sparse model.
    """
    index = ensure_sparse_index()

    return index.search(
        namespace=NAMESPACE, 
        query={
            "inputs": {"text": query_text}, 
            "top_k": top_k
        }
    )