# RAG Document Parser

## Project Description
RAG Document Parser is a tool designed to parse and process documents efficiently using Retrieval-Augmented Generation (RAG) techniques. It supports various document formats and provides structured outputs for downstream applications. The tool is modular and can be extended to include advanced features for enhanced functionality.

## Installation
To install the project dependencies, run:
```bash
pip install -r requirements.txt
```

## Environment Variables
The following environment variables are required:

- **PINECONE_API_KEY**: Your Pinecone API key for vector storage and semantic search
- **GEMINI_API_KEY**: Your Google Gemini API key for AI-powered answer generation

Create a `.env` file in the project root with these variables:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Web Interface
To start the Flask web interface:
```bash
FLASK_APP=src/web/app.py flask run
```

Then navigate to `http://localhost:5000` in your browser to access the RAG search interface.

### Web Interface Features
- **Semantic Search**: Enter queries to search through ingested documents using vector similarity
- **Search Results**: View retrieved document chunks with relevance scores and metadata
- **AI Answer Generation**: Click "Generate Answer" to get AI-powered responses based on retrieved context
- **Real-time Interaction**: Generate answers without page reloads using AJAX

## Usage
Here is an example of how to use the RAG Document Parser:
```python
from rag_document_parser import DocumentParser

parser = DocumentParser()
result = parser.parse("path/to/document.pdf")
print(result)
```

## Extending the Project
The RAG Document Parser can be extended with the following features:

## Hybrid Retrieval & Reranking

The RAG Document Parser now supports hybrid retrieval that combines both dense semantic search and sparse keyword search with intelligent reranking. This approach significantly improves retrieval accuracy by leveraging the strengths of both methods.

### How Hybrid Retrieval Works

1. **Dense Semantic Search**: Uses vector embeddings to find semantically similar content
2. **Sparse Keyword Search**: Uses TF-IDF based keyword matching for exact term matches
3. **Result Merging**: Combines results from both approaches, deduplicating by document ID
4. **Intelligent Reranking**: Scores results using multiple relevance signals:
   - Normalized dense similarity score (weight: 0.5)
   - Normalized sparse keyword score (weight: 0.3)
   - Lexical overlap ratio (weight: 0.2)

### Using Hybrid Search

#### Basic Usage

```python
from src.retrieval.hybrid_search import hybrid_search

# Execute hybrid search
results = hybrid_search(
    query="existential meaning of life",
    top_k_dense=5,      # Number of dense results
    top_k_sparse=20     # Number of sparse results
)

# Results include detailed scoring breakdown
for result in results:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Source: {result['source']}")  # 'dense', 'sparse', 'both'
    
    # Detailed score breakdown
    breakdown = result['score_breakdown']
    print(f"Dense: {breakdown['normalized_dense']:.3f}")
    print(f"Sparse: {breakdown['normalized_sparse']:.3f}")
    print(f"Overlap: {breakdown['overlap_score']:.3f}")
```

#### Testing Hybrid Search

```bash
# Test both semantic and hybrid search
python src/scripts/test_search.py
```

### Document Ingestion for Hybrid Search

The ingestion process now creates both dense and sparse indexes:

```bash
# Ingest documents with hybrid indexing
python src/scripts/ingest_documents.py
```

During ingestion:
- **Dense vectors** are stored in the `philosophy-rag` index using managed embeddings
- **Sparse vectors** are stored in the `philosophy-rag-sparse` index using TF-IDF weights
- **Vocabulary** and **document frequencies** are persisted in `data/vocab.json` and `data/df.json`

### Technical Details

#### Sparse Vector Construction
- **Tokenization**: Lowercase, alphanumeric splitting, stopword filtering, minimum length 2
- **TF-IDF Formula**: `(1 + log(tf)) * log((N + 1) / (df + 1)) + 1`
- **Vocabulary Management**: Token-to-ID mapping persisted locally
- **Document Frequencies**: Global DF statistics for IDF calculation

#### Reranking Algorithm
1. Normalize dense and sparse scores separately using min-max normalization
2. Calculate lexical overlap as intersection/union of query and document tokens
3. Compute weighted combination: `0.5 * dense + 0.3 * sparse + 0.2 * overlap`
4. Sort results by final relevance score

#### Graceful Degradation
- If sparse index is unavailable, falls back to dense-only search
- If dense index is unavailable, falls back to sparse-only search
- Maintains consistent result format regardless of available indexes

### Data Storage

The system creates and maintains:
- `data/vocab.json`: Token to integer ID mapping for sparse vectors
- `data/df.json`: Document frequencies for TF-IDF calculations
- Two Pinecone indexes:
  - `philosophy-rag`: Dense vectors with managed embeddings
  - `philosophy-rag-sparse`: Sparse vectors with TF-IDF weights

## Extending the Project
The RAG Document Parser can be extended with the following features:

### 1. Reranking with Cross-Encoder
Enhance the retrieval process by adding a reranking step using a cross-encoder. After retrieving the initial `top_k` documents, use a cross-encoder model to rerank the results based on relevance.

### 2. Generation Layer with Prompt Templates
Incorporate a generation layer using a Large Language Model (LLM). Use prompt templates that reference retrieved chunks to generate coherent and contextually relevant outputs.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
