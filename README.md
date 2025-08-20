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

### 1. Reranking with Cross-Encoder
Enhance the retrieval process by adding a reranking step using a cross-encoder. After retrieving the initial `top_k` documents, use a cross-encoder model to rerank the results based on relevance.

### 2. Hybrid Retrieval
Combine classical and dense retrieval methods by storing text in a classical index (e.g., Elasticsearch) while also leveraging dense embeddings. This hybrid approach improves retrieval accuracy and robustness.

### 3. Generation Layer with Prompt Templates
Incorporate a generation layer using a Large Language Model (LLM). Use prompt templates that reference retrieved chunks to generate coherent and contextually relevant outputs.

## Architecture Decisions
See [docs/adr/README.md](docs/adr/README.md) for Architecture Decision Records documenting key technical choices.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
