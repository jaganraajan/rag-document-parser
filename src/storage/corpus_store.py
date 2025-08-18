#!/usr/bin/env python3
"""
Corpus Store for BM25 Retrieval

This module provides functionality to persist and load document chunks for BM25-based
sparse retrieval. It handles the corpus JSONL format and builds BM25 indexes.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi


@dataclass
class ChunkResult:
    """Standardized result format for all retrieval methods."""
    id: str
    text: str
    score: float
    source: str  # "bm25" | "vector" | "hybrid"
    metadata: Dict[str, Any]


class CorpusStore:
    """Manages document corpus persistence and BM25 index for sparse retrieval."""
    
    def __init__(self, corpus_path: str = "data/chunks_corpus.jsonl"):
        self.corpus_path = corpus_path
        self.chunks: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._index_loaded = False
        
    def save_chunk(self, chunk_id: str, chunk_text: str, metadata: Dict[str, Any]):
        """Save a single chunk to the corpus file."""
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        
        chunk_record = {
            "id": chunk_id,
            "chunk_text": chunk_text,
            "metadata": metadata or {}
        }
        
        with open(self.corpus_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(chunk_record) + "\n")
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        """Load all chunks from the corpus file."""
        if not os.path.exists(self.corpus_path):
            print(f"Warning: Corpus file {self.corpus_path} not found")
            return []
        
        chunks = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line in corpus: {e}")
        
        self.chunks = chunks
        return chunks
    
    def build_bm25_index(self, tokenizer=None):
        """Build BM25 index from loaded corpus."""
        if not self.chunks:
            self.load_corpus()
        
        if not self.chunks:
            print("Warning: No chunks available to build BM25 index")
            return None
        
        # Default tokenizer: simple split
        if tokenizer is None:
            tokenizer = lambda text: text.lower().split()
        
        # Tokenize all chunk texts
        tokenized_docs = []
        for chunk in self.chunks:
            text = chunk.get("chunk_text", "")
            tokenized_docs.append(tokenizer(text))
        
        self.bm25_index = BM25Okapi(tokenized_docs)
        self._index_loaded = True
        print(f"Built BM25 index over {len(tokenized_docs)} documents")
        return self.bm25_index
    
    def search(self, query: str, top_k: int = 5, tokenizer=None) -> List[ChunkResult]:
        """Perform BM25 search over the corpus."""
        if not self._index_loaded:
            self.build_bm25_index(tokenizer)
        
        if not self.bm25_index or not self.chunks:
            return []
        
        # Default tokenizer: simple split
        if tokenizer is None:
            tokenizer = lambda text: text.lower().split()
        
        # Tokenize query
        tokenized_query = tokenizer(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results with scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(ChunkResult(
                id=chunk["id"],
                text=chunk["chunk_text"],
                score=float(scores[idx]),
                source="bm25",
                metadata=chunk.get("metadata", {})
            ))
        
        return results
    
    def clear_corpus(self):
        """Clear the corpus file (useful for reingestion)."""
        if os.path.exists(self.corpus_path):
            os.remove(self.corpus_path)
        self.chunks = []
        self.bm25_index = None
        self._index_loaded = False


# Global instance for easy access
_default_store = None

def get_corpus_store(corpus_path: str = "data/chunks_corpus.jsonl") -> CorpusStore:
    """Get the default corpus store instance."""
    global _default_store
    if _default_store is None or _default_store.corpus_path != corpus_path:
        _default_store = CorpusStore(corpus_path)
    return _default_store


def bm25_search(query: str, top_k: int = 5) -> List[ChunkResult]:
    """Convenience function for BM25 search using default corpus store."""
    store = get_corpus_store()
    return store.search(query, top_k)