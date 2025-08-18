# RAG Retrieval Evaluation

This directory contains the `evaluate_retrieval.py` script for evaluating the retrieval accuracy of the RAG system.

## Overview

The evaluation script measures how well the RAG system retrieves relevant documents for given queries by computing standard retrieval metrics:

- **Precision@k**: The proportion of retrieved documents that are relevant
- **Recall@k**: The proportion of relevant documents that are retrieved  
- **Hit Rate**: Whether at least one relevant document is retrieved

## Usage

### Basic Usage

```bash
# Run with built-in sample dataset
python src/scripts/evaluate_retrieval.py

# Run in test mode (when Pinecone is not available)
python src/scripts/evaluate_retrieval.py --test-mode --verbose
```

### Custom Evaluation Dataset

```bash
# Use your own evaluation file
python src/scripts/evaluate_retrieval.py --eval-file my_evaluation.json --k 10

# Use CSV format
python src/scripts/evaluate_retrieval.py --eval-file my_evaluation.csv --k 5 --verbose
```

### Generate Sample Files

```bash
# Generate sample JSON file
python src/scripts/evaluate_retrieval.py --save-sample sample.json

# Generate sample CSV file  
python src/scripts/evaluate_retrieval.py --save-sample sample.csv
```

## Evaluation Dataset Format

### JSON Format

```json
[
  {
    "query": "existential meaning",
    "expected_chunks": [
      "existential philosophy and the search for meaning",
      "meaning of life in existential thought"
    ],
    "expected_ids": ["doc_123", "doc_456"]
  }
]
```

### CSV Format

```csv
query,expected_chunks,expected_ids
"existential meaning","existential philosophy and the search for meaning;meaning of life in existential thought","doc_123;doc_456"
```

## Parameters

- `--eval-file`: Path to evaluation dataset (JSON or CSV)
- `--k`: Number of top results to evaluate (default: 5)
- `--similarity-threshold`: Text similarity threshold for relevance (default: 0.3)
- `--verbose`: Enable detailed output
- `--test-mode`: Run with mock results for testing
- `--save-sample`: Generate sample evaluation file

## Output

The script produces a summary table showing:

```
================================================================================
RAG RETRIEVAL EVALUATION SUMMARY (k=5)
================================================================================
Query                          Precision@k  Recall@k   Hit Rate  Rel/Tot 
------------------------------ ------------ ---------- --------- --------
existential meaning            0.333        0.333      1.0       1/3     
consciousness and awareness    0.000        0.000      0.0       0/3     
AVERAGE                        0.167        0.167      0.500    

Detailed Statistics:
  Total queries evaluated: 2
  Average Precision@5: 0.167
  Average Recall@5: 0.167
  Average Hit Rate: 0.500
```

## How It Works

1. **Load Evaluation Data**: Reads queries and expected results from JSON or CSV
2. **Retrieve Results**: Uses `semantic_query()` to get top-k results for each query
3. **Match Relevance**: Compares retrieved results with expected chunks using:
   - Exact ID matching (if expected_ids provided)
   - Text similarity using Jaccard coefficient
4. **Calculate Metrics**: Computes precision, recall, and hit rate
5. **Display Results**: Shows formatted summary table

## Text Similarity

The script uses Jaccard similarity to match retrieved text with expected chunks:
- Converts text to lowercase word sets
- Calculates intersection over union
- Default threshold: 0.3 (configurable)

## Test Mode

When Pinecone is unavailable, use `--test-mode` to run with mock results:
- Generates realistic mock retrieval results
- Useful for development and testing
- Some mock results are designed to partially match expected chunks