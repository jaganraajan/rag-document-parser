# RAG Document Parser

## Project Description
RAG Document Parser is a modular system for parsing, chunking, and searching documents using Retrieval-Augmented Generation (RAG) techniques. It supports PDF ingestion, metadata extraction, chunking, and semantic search via Pinecone vector database, and provides a Flask web interface for user queries and feedback.

## Installation
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Environment Variables
Required variables:
- **PINECONE_API_KEY**: Your Pinecone API key for vector storage and semantic search
- **GEMINI_API_KEY**: Your Google Gemini API key for AI-powered answer generation

Create a `.env` file in the project root:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Web Interface
Start Flask:
```bash
FLASK_APP=src/web/app.py flask run
```
Navigate to [http://localhost:5000](http://localhost:5000).

### Web Interface Features
- **Semantic Search**: Search documents using dense and sparse vector embeddings (Pinecone).
- **Model Selection**: Choose dense/sparse embedding models and reranker model via dropdowns.
- **Search Results**: View retrieved document chunks with scores and metadata.
- **Reranking**: Rerank results with different models, including Pinecone's managed rerankers.
- **User Feedback**: Provide relevance feedback per result; feedback is logged for evaluation.
- **AI Answer Generation**: Use Google Gemini for context-based Q&A.

## Usage Example
```python
from rag_document_parser import DocumentParser

parser = DocumentParser()
result = parser.parse("path/to/document.pdf")
print(result)
```
*(Note: Actual usage follows ingestion and search pipeline in `src/scripts/ingest_documents.py` and the Flask app in `src/web/app.py`.)*

## Extending the Project
- Add new embedding or reranker models via config/options.
- Integrate with additional vector stores or classical indices (e.g., Elasticsearch).
- Add more advanced feedback evaluation or continuous learning.

## Architecture Decisions
See [docs/adr/README.md](docs/adr/README.md) for Architecture Decision Records.

## Evaluation and Experiment Tracking
- User feedback is logged and aggregated for precision/recall metrics.
- MLflow is used for experiment tracking and metric visualization.
- Model selection and evaluation are reflected in retrieval quality.

## License
MIT License (see LICENSE).