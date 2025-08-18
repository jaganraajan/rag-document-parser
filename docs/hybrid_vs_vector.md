# Hybrid vs Vector Retrieval

This document explains the hybrid retrieval approach implemented in this RAG system, combining sparse (BM25) and dense (vector) retrieval methods.

## Rationale for Hybrid Retrieval

Traditional vector-only retrieval has limitations:
- **Lexical gaps**: May miss exact keyword matches that are semantically distant
- **Out-of-vocabulary terms**: Struggles with proper nouns, technical terms, or rare words
- **Query-document mismatch**: Vector similarity doesn't always align with relevance

BM25 (sparse retrieval) excels at:
- **Exact keyword matching**: Strong signal for term importance
- **Query term coverage**: Rewards documents containing query terms
- **Fast execution**: Lightweight compared to vector similarity

Hybrid retrieval combines both approaches to leverage their complementary strengths.

## Scoring Combination Formula

The hybrid score is computed as a weighted combination of normalized BM25 and vector scores:

```
hybrid_score = α × norm_vector_score + (1-α) × norm_bm25_score
```

Where:
- **α (alpha)**: Blending weight between 0.0 and 1.0
  - α = 0.0: Pure BM25 (sparse only)
  - α = 1.0: Pure vector (dense only)  
  - α = 0.5: Equal weighting (default)
- **norm_vector_score**: Min-max normalized vector similarity score
- **norm_bm25_score**: Min-max normalized BM25 score

## Normalization

Score normalization is applied per query across the candidate set:

### Min-Max Normalization
For a set of scores [s₁, s₂, ..., sₙ]:
```
norm_score_i = (s_i - min(scores)) / (max(scores) - min(scores))
```

This ensures both BM25 and vector scores are on the same scale [0, 1] before blending.

### Candidate Set Formation
1. Retrieve top-2k results from both vector and BM25 methods
2. Union the candidate sets by document ID
3. Normalize scores within this combined candidate set
4. Compute hybrid scores and return top-k

## Tuning Guidance

### Alpha Parameter (α)
- **Start with α = 0.5** for balanced retrieval
- **Increase α (toward 1.0)** if:
  - Vector search performs well on your domain
  - Semantic similarity is more important than exact matches
  - You have high-quality embeddings
- **Decrease α (toward 0.0)** if:
  - Exact keyword matching is crucial
  - Your domain has many technical terms or proper nouns
  - BM25 outperforms vector search alone

### Top-k Parameter
- **k = 5-10**: Good for most applications
- **Higher k**: Better recall but may include less relevant results
- **Lower k**: Higher precision but may miss relevant documents

### Evaluation-Driven Tuning
Use the evaluation script to systematically test different parameters:

```bash
# Test different alpha values
python -m src.scripts.evaluate_retrieval --alpha 0.3 --top-k 5
python -m src.scripts.evaluate_retrieval --alpha 0.7 --top-k 5

# Test different k values  
python -m src.scripts.evaluate_retrieval --alpha 0.5 --top-k 3
python -m src.scripts.evaluate_retrieval --alpha 0.5 --top-k 10
```

## Performance Considerations

### Latency Trade-offs
- **BM25**: Very fast (~1-50ms for small corpora)
- **Vector Search**: Moderate latency (~50-200ms depending on index size)
- **Hybrid**: Combines both, typically adds 20-50% overhead

### When to Accept Latency Trade-offs
Accept higher hybrid latency when:
- Coverage improvements outweigh latency costs
- Downstream answer quality correlates with retrieval coverage
- User experience tolerates the additional latency
- Business value justifies the compute cost

### Optimization Strategies
1. **Parallel Execution**: Run BM25 and vector search concurrently
2. **Caching**: Cache BM25 index in memory across requests
3. **Early Termination**: Stop if one method finds sufficient high-quality results
4. **Async Processing**: Use async/await for non-blocking execution

## Expected Performance Gains

Based on typical evaluation results:

| Metric | Vector Only | BM25 Only | Hybrid (α=0.5) |
|--------|-------------|-----------|-----------------|
| Coverage@5 | 45-60% | 55-70% | 65-80% |
| Precision@5 | 0.25-0.40 | 0.20-0.35 | 0.30-0.45 |
| Latency P95 | 150-200ms | 20-50ms | 200-300ms |

Actual results depend on:
- Domain and document characteristics
- Query types and complexity  
- Embedding model quality
- Corpus size and indexing setup

## Implementation Notes

### ID Consistency
Both BM25 and vector stores use the same document IDs generated during ingestion, enabling proper result merging and deduplication.

### Tokenization
BM25 uses simple whitespace tokenization by default. This can be customized for domain-specific needs:

```python
# Custom tokenizer example
def custom_tokenizer(text):
    import re
    # Remove punctuation, lowercase, split
    return re.findall(r'\b\w+\b', text.lower())

corpus_store.build_bm25_index(tokenizer=custom_tokenizer)
```

### Error Handling
The hybrid search gracefully degrades:
- If vector search fails: Falls back to BM25 only
- If BM25 search fails: Falls back to vector only  
- If both fail: Returns empty results with error logging