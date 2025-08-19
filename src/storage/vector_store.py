from pinecone import Pinecone
from typing import Iterable, List, Dict
# from .id_strategy import IDStrategy
import uuid
import os

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
    # print('index', index)
    # print(chunks[:2])
    records = to_records(chunks)
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        index.upsert_records(NAMESPACE, batch)


def semantic_query(query: str, top_k: int = 5) -> List[Dict[str, any]]:
    index = ensure_index()
    print('search for query:', query)
    # Search the dense index
    results = index.search(
        namespace=NAMESPACE,
        query={
            "inputs": {
                "text": query
            },
            "top_k": top_k
        }
    )

    return results
