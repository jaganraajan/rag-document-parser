# Evaluation Guide

This guide explains how to build evaluation datasets, run retrieval evaluations, and interpret the results for the hybrid RAG system.

## Overview

The evaluation system measures retrieval effectiveness across three methods:
- **Vector Search**: Dense retrieval using embeddings
- **BM25 Search**: Sparse retrieval using keyword matching  
- **Hybrid Search**: Weighted combination of both methods

## Building an Evaluation Set

### Dataset Format

Evaluation data uses JSON format with this structure:

```json
[
  {
    "query": "existential meaning",
    "relevant_substrings": ["existential", "meaning of life", "purpose"],
    "notes": "Optional description of the query intent",
    "answer_quality": 0.8
  }
]
```

### Required Fields
- **query**: The search query string
- **relevant_substrings**: List of substrings that indicate relevance

### Optional Fields  
- **notes**: Human-readable description for reference
- **answer_quality**: Float (0-1) representing downstream answer quality for correlation analysis

### Defining Relevance

A retrieved chunk is considered relevant if it contains **any** of the `relevant_substrings` (case-insensitive matching).

#### Guidelines for Relevant Substrings:
1. **Be specific but not too narrow**: Include key terms and synonyms
2. **Cover variations**: Include different forms (singular/plural, verb forms)
3. **Domain-appropriate**: Use terminology natural to your corpus
4. **Balanced coverage**: Not too broad (everything matches) or too narrow (nothing matches)

#### Example Strategy:
```json
{
  "query": "artificial intelligence ethics",
  "relevant_substrings": [
    "artificial intelligence",
    "AI ethics", 
    "machine learning ethics",
    "algorithmic bias",
    "responsible AI"
  ]
}
```

### Evaluation Set Size

For meaningful results:
- **Minimum**: 10-15 queries for basic evaluation
- **Recommended**: 20-50 queries for reliable metrics
- **Comprehensive**: 100+ queries for production evaluation

### Creating Quality Evaluation Sets

#### 1. Representative Query Distribution
- Cover different query types (factual, conceptual, complex)
- Include both common and edge-case queries
- Vary query length and complexity

#### 2. Difficulty Levels
- **Easy**: Direct keyword matches expected
- **Medium**: Require some semantic understanding  
- **Hard**: Need deep domain knowledge or inference

#### 3. Domain Coverage
- Ensure queries span your document corpus
- Include both broad and specific topics
- Test boundary cases and specialized terminology

#### 4. Answer Quality Correlation (Optional)
If you want to measure correlation between retrieval coverage and downstream answer quality:

1. Generate answers for each query using your RAG system
2. Have humans rate answer quality (0-1 scale)
3. Include ratings in evaluation data as `answer_quality` field
4. The system will compute Pearson correlation with coverage

## Running Evaluations

### Basic Usage

```bash
# Use sample evaluation set
python -m src.scripts.evaluate_retrieval

# Use custom evaluation file
python -m src.scripts.evaluate_retrieval --eval-file my_eval.json

# Test different parameters
python -m src.scripts.evaluate_retrieval --top-k 10 --alpha 0.3
```

### Parameter Options

- `--eval-file`: Path to evaluation dataset (default: `eval/eval_set.sample.json`)
- `--top-k`: Number of results to evaluate (default: 5)
- `--alpha`: Hybrid search weighting (default: 0.5)
- `--verbose`: Show detailed per-query results
- `--show-table`: Display per-query metrics table
- `--test-mode`: Run with mock data (no search backend required)
- `--output-dir`: Directory for result files (default: `eval/results`)

### Output Files

Results are saved to `eval/results/latest_results.json` with detailed per-query breakdown:

```json
{
  "query": "existential meaning",
  "relevant_substrings": ["existential", "meaning"],
  "methods": {
    "vector": {
      "metrics": {
        "coverage_at_k": 1.0,
        "precision_at_k": 0.4,
        "mrr_at_k": 1.0
      },
      "latency_ms": 156.2,
      "results": [...]
    }
  }
}
```

## Understanding Metrics

### Coverage@k (Hit Rate)
- **Definition**: Proportion of queries with at least one relevant result in top-k
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - 0.8 = 80% of queries found at least one relevant document
  - Most important metric for RAG systems

### Precision@k  
- **Definition**: Average proportion of relevant documents in top-k results
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 0.3 = On average, 30% of returned documents are relevant
  - Measures result quality

### MRR@k (Mean Reciprocal Rank)
- **Definition**: Average of 1/rank_of_first_relevant_result across queries
- **Range**: 0.0 to 1.0 (higher is better)  
- **Interpretation**:
  - 0.5 = First relevant result appears at rank 2 on average
  - Measures how quickly users find relevant information

### Latency
- **P95 Latency**: 95th percentile response time (most queries complete within this time)
- **Average Latency**: Mean response time across all queries
- **Use P95 for SLA planning**, average for capacity planning

## Interpreting Results

### Summary Table Example
```
Method   Coverage@5  Precision@5  MRR@5  P95 Latency (ms)
Vector   0.65        0.32         0.51   180
BM25     0.71        0.28         0.58   45  
Hybrid   0.78        0.35         0.62   220
```

### Key Insights:
1. **Hybrid achieves highest coverage** (78% vs 65% for vector)
2. **BM25 is fastest** but lowest precision
3. **Hybrid trades latency for coverage** (+40ms for +13 percentage points)

### Decision Framework

**Choose Vector** when:
- Semantic similarity is most important
- Latency is critical
- Domain has good embedding coverage

**Choose BM25** when:  
- Exact keyword matching is crucial
- Very low latency required
- Lexical matching suffices

**Choose Hybrid** when:
- Maximum coverage is needed
- Willing to accept latency trade-off
- Want robust retrieval across query types

## Performance Optimization

### Evaluation Set Optimization
1. **Iterative refinement**: Start small, expand based on insights
2. **Error analysis**: Identify failure patterns and add targeted queries
3. **Balanced difficulty**: Mix easy and hard queries

### System Optimization
1. **Parameter tuning**: Systematically test α values and k
2. **Correlation analysis**: Measure retrieval-answer quality relationship
3. **A/B testing**: Compare different configurations in production

### Continuous Evaluation
1. **Regular evaluation**: Re-run evaluation as corpus changes
2. **Query drift monitoring**: Track whether new queries match evaluation set
3. **Performance tracking**: Monitor metrics over time

## Troubleshooting Common Issues

### Low Coverage
- **Symptoms**: < 50% coverage across all methods
- **Causes**: Poor evaluation set, corpus mismatch, bad embeddings
- **Solutions**: Review relevant_substrings, check corpus content, validate search functions

### High Latency
- **Symptoms**: > 500ms P95 latency
- **Causes**: Large corpus, inefficient indexing, network issues
- **Solutions**: Optimize indexes, use async search, implement caching

### Inconsistent Results
- **Symptoms**: High variance between evaluation runs
- **Causes**: Small evaluation set, random sampling, unstable search
- **Solutions**: Increase evaluation size, fix random seeds, debug search stability

## Best Practices

1. **Version control evaluation sets**: Track changes to queries and relevance judgments
2. **Document evaluation methodology**: Record how relevant_substrings were chosen
3. **Regular updates**: Refresh evaluation sets as domain evolves
4. **Human validation**: Periodically verify automatic relevance matching
5. **Cross-validation**: Test evaluation approach on known-good/bad results