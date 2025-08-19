"""
Sparse vector storage and management for hybrid retrieval.

This module handles:
- Vocabulary management (token to integer ID mapping)
- Document frequency tracking for TF-IDF calculations
- Sparse vector construction from text
- Pinecone sparse index operations
"""

import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set

from pinecone import Pinecone

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY env var")

pc = Pinecone(api_key=PINECONE_API_KEY)

SPARSE_INDEX_NAME = "philosophy-rag-sparse"
NAMESPACE = "__default__"
VOCAB_FILE = "/home/runner/work/rag-document-parser/rag-document-parser/data/vocab.json"
DF_FILE = "/home/runner/work/rag-document-parser/rag-document-parser/data/df.json"

# Basic stopwords list
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'you', 'we', 'they', 'this', 'but', 'not', 'or', 'have', 'had', 'been',
    'their', 'if', 'would', 'could', 'should', 'can', 'may', 'might', 'must'
}


class SparseVectorStore:
    def __init__(self):
        self.vocab = self._load_vocab()
        self.df = self._load_df()
        self.next_token_id = max(self.vocab.values()) + 1 if self.vocab else 1
    
    def _load_vocab(self) -> Dict[str, int]:
        """Load vocabulary from file or return empty dict."""
        if os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load vocab from {VOCAB_FILE}, starting fresh")
        return {}
    
    def _save_vocab(self):
        """Save vocabulary to file."""
        os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
        with open(VOCAB_FILE, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def _load_df(self) -> Dict[str, int]:
        """Load document frequencies from file or return empty dict."""
        if os.path.exists(DF_FILE):
            try:
                with open(DF_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load DF from {DF_FILE}, starting fresh")
        return {}
    
    def _save_df(self):
        """Save document frequencies to file."""
        os.makedirs(os.path.dirname(DF_FILE), exist_ok=True)
        with open(DF_FILE, 'w') as f:
            json.dump(self.df, f, indent=2)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text according to specs:
        - Lowercase
        - Split on non-alphanumeric
        - Filter stopwords
        - Length >= 2
        """
        if not text:
            return []
        
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        
        # Filter stopwords and length
        filtered_tokens = [
            token for token in tokens 
            if len(token) >= 2 and token not in STOPWORDS
        ]
        
        return filtered_tokens
    
    def get_or_create_token_id(self, token: str) -> int:
        """Get existing token ID or create new one."""
        if token not in self.vocab:
            self.vocab[token] = self.next_token_id
            self.next_token_id += 1
        return self.vocab[token]
    
    def build_sparse_vector(self, text: str, update_df: bool = True) -> Tuple[List[int], List[float]]:
        """
        Build sparse vector from text using TF-IDF.
        
        Args:
            text: Input text to vectorize
            update_df: Whether to update document frequencies (True during ingestion)
        
        Returns:
            Tuple of (indices, values) for sparse vector
        """
        tokens = self.tokenize(text)
        if not tokens:
            return [], []
        
        # Calculate term frequencies
        tf_counts = Counter(tokens)
        unique_tokens = set(tokens)
        
        # Update document frequencies if needed
        if update_df:
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        # Get total number of documents (approximate)
        N = max(self.df.values()) if self.df else 1
        
        indices = []
        values = []
        
        for token, tf in tf_counts.items():
            token_id = self.get_or_create_token_id(token)
            df = self.df.get(token, 1)  # Default to 1 if not found
            
            # TF-IDF formula: (1 + log(tf)) * log((N + 1) / (df + 1)) + 1
            tf_component = 1 + math.log(tf)
            idf_component = math.log((N + 1) / (df + 1))
            tfidf_score = tf_component * idf_component + 1
            
            indices.append(token_id)
            values.append(tfidf_score)
        
        return indices, values
    
    def ensure_sparse_index(self):
        """Ensure sparse index exists in Pinecone."""
        if not pc.has_index(SPARSE_INDEX_NAME):
            # Create serverless index for sparse vectors
            pc.create_index(
                name=SPARSE_INDEX_NAME,
                dimension=1,  # Minimal dimension since we're using sparse vectors
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        return pc.Index(SPARSE_INDEX_NAME)
    
    def upsert_sparse_vectors(self, chunks: List[Dict[str, Any]]):
        """
        Upsert sparse vectors to Pinecone index.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk', 'id', 'metadata'
        """
        index = self.ensure_sparse_index()
        vectors = []
        
        for chunk in chunks:
            chunk_text = chunk.get('chunk', '')
            chunk_id = chunk.get('id', str(uuid.uuid4()))
            metadata = chunk.get('metadata', {})
            
            indices, values = self.build_sparse_vector(chunk_text, update_df=True)
            
            if indices and values:  # Only add if we have valid sparse vector
                vector = {
                    "id": chunk_id,
                    "sparse_values": {
                        "indices": indices,
                        "values": values
                    },
                    "metadata": {
                        "chunk_text": chunk_text[:1000],  # Limit text length
                        **{f"meta_{k}": v for k, v in metadata.items() 
                           if isinstance(v, (str, int, float, bool))}
                    }
                }
                vectors.append(vector)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch, namespace=NAMESPACE)
        
        # Save updated vocab and DF
        self._save_vocab()
        self._save_df()
        
        print(f"Upserted {len(vectors)} sparse vectors to {SPARSE_INDEX_NAME}")
    
    def sparse_query(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Execute sparse keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            index = self.ensure_sparse_index()
            
            # Build sparse vector for query (don't update DF)
            indices, values = self.build_sparse_vector(query, update_df=False)
            
            if not indices or not values:
                print(f"Warning: No valid tokens in query '{query}'")
                return []
            
            # Execute sparse search
            results = index.query(
                sparse_vector={
                    "indices": indices,
                    "values": values
                },
                top_k=top_k,
                namespace=NAMESPACE,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'text': match.get('metadata', {}).get('chunk_text', ''),
                    'metadata': {
                        k[5:] if k.startswith('meta_') else k: v 
                        for k, v in match.get('metadata', {}).items()
                        if k != 'chunk_text'
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Warning: Sparse search failed: {e}")
            return []
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
    
    def get_document_count(self) -> int:
        """Get approximate document count from DF stats."""
        return max(self.df.values()) if self.df else 0


# Global instance
sparse_store = SparseVectorStore()