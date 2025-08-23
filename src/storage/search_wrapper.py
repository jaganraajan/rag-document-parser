from typing import List, Dict, Any

from .sparse_store import sparse_query
from .vector_store import semantic_query

def search_with_metadata(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simplified parser for Pinecone response shape:
      {
        "result": {
          "hits": [
            {
              "_id": "...",
              "_score": float,
              "fields": {
                  "chunk_text": "...",
                  "meta_title": "...",
                  "meta_page_number": 3.0,
                  ...
              }
            }, ...
          ]
        }
      }
    """
    dense_results_raw = semantic_query(query, top_k=top_k)
    sparse_results_raw = sparse_query(query, top_k=top_k)
    # print('in search_with_metadata, raw:', raw)
    # print('result is', raw.get("result", {}))
    # print('hits is', raw.get("result", {}).get("hits", []))

    
    # print('hits2 is', hits)
    
    def normalize(raw):
        hits = raw.get("result", {}).get("hits", [])
        normalized: List[Dict[str, Any]] = []
        for h in hits:
            fields = h.get("fields", {}) or {}
            text = fields.get("chunk_text") or fields.get("text") or ""
            # print('text is', text)
            meta: Dict[str, Any] = {}
            for k, v in fields.items():
                if k.startswith("meta_"):
                    meta[k[5:]] = v  # strip 'meta_'
            normalized.append({
                "id": h.get("_id"),
                "score": h.get("_score"),
                "text": text,
                "metadata": meta,
                "page_number": meta.get("page_number"),
                "paragraph_index": meta.get("paragraph_index"),
                "source_file": meta.get("source_file"),
                "title": meta.get("title"),
                "raw": h
            })
        return normalized
    
    return {
        "dense_results": normalize(dense_results_raw),
        "sparse_results": normalize(sparse_results_raw)
    }