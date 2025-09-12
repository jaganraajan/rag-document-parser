# RAG Document Parser

## Project Description
RAG Document Parser is a modular system for parsing, chunking, and searching documents using Retrieval-Augmented Generation (RAG) techniques. It supports PDF ingestion, metadata extraction, chunking, and semantic search via Pinecone vector database, and provides a Flask web interface for user queries and feedback.

**NEW**: The project now includes a **Hybrid Document Normalization Pipeline** that combines rule-based extraction with selective AI fallback for cost-effective document processing.

## Demo
<img width="1439" height="806" alt="Screenshot 2025-08-26 at 9 31 01 AM" src="https://github.com/user-attachments/assets/5c781001-f1c7-4f9e-9043-495dd5145b6c" />
<img width="1395" height="811" alt="Screenshot 2025-08-26 at 9 31 54 AM" src="https://github.com/user-attachments/assets/c3d0cfe7-7b55-4357-bad1-fc48f60a5929" />

### Grafana Dashboard
<img width="1439" height="806" alt="Screenshot 2025-08-26 at 9 25 28 AM" src="https://github.com/user-attachments/assets/fec2e171-e66c-407d-ae83-bb2090fd3ae5" />

### MLFlow Dashboard
<img width="1205" height="767" alt="Screenshot 2025-08-26 at 10 16 52 AM" src="https://github.com/user-attachments/assets/04aba876-db2b-48e8-ae09-0edb842f171d" />
<img width="1395" height="767" alt="Screenshot 2025-08-26 at 10 12 56 AM" src="https://github.com/user-attachments/assets/c67fcbbf-a763-408a-915c-e70398e28a1e" />
<img width="1395" height="811" alt="Screenshot 2025-08-26 at 9 51 56 AM" src="https://github.com/user-attachments/assets/7673c5f9-c402-4ed2-a2e6-1ae699f4be7f" />
<img width="1395" height="811" alt="Screenshot 2025-08-26 at 9 35 35 AM" src="https://github.com/user-attachments/assets/8d8d0c43-280e-495b-b0fb-555540e2a8cc" />


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

## Hybrid Document Normalization Pipeline

### Overview
The Hybrid Document Normalization Pipeline is a cost-effective system that processes heterogeneous document formats (PDF digital, PDF scan, image, email, HTML/text) and outputs normalized JSON with interpretability logs and selective AI usage.

### Key Features
- **🔄 Multi-format Support**: PDF, images, emails, text, HTML
- **🏷️ Layout Signature Learning**: Automatic document layout recognition and reuse
- **📋 Rule Engine**: Global and signature-specific extraction rules
- **🤖 Selective AI Fallback**: Only uses LLM when rules fail
- **📊 Cost Tracking**: Complete cost monitoring and limits
- **📝 Interpretability Logs**: Full audit trail of processing decisions
- **🌐 Streamlit UI**: Web interface for document upload and management

### Architecture Components
1. **Normalized JSON Schema**: Stable, extensible schema for all document types
2. **Layout Signature System**: Quantized structural tokens with Jaccard similarity matching
3. **Rule Engine**: YAML-based regex/heuristic extraction rules
4. **Selective AI Fallback**: Pluggable LLM providers with cost controls
5. **Interpretation Logging**: JSONL event logs for every pipeline decision
6. **Storage Repository**: Persistent document storage with metadata indexing

### Quick Start with Hybrid Pipeline

#### 1. Test the Pipeline
```bash
python test_pipeline.py
```

#### 2. Run the Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

#### 3. Upload Documents
Navigate to http://localhost:8501 and use the Document Upload page to:
- Upload multiple documents (PDF, images, emails, text)
- Set cost limits for LLM usage
- View processing results and extracted fields
- Download normalized JSON outputs

### Directory Structure
```
rag-document-parser/
├── src/normalization/           # Hybrid normalization pipeline
│   ├── schema.py               # Normalized JSON schema
│   ├── signatures.py           # Layout signature system
│   ├── rules_engine.py         # Rule-based extraction
│   ├── extractor/              # Document extractors
│   │   ├── pdf_extractor.py    # PDF layout extraction
│   │   ├── ocr.py              # OCR for images
│   │   └── email_extractor.py  # Email parsing
│   ├── model_fallback/         # LLM integration
│   ├── logs/                   # Interpretation logging
│   ├── storage/                # Document repository
│   └── pipeline/               # Main orchestrator
├── app/streamlit_app.py        # Streamlit web interface
├── rules/                      # Extraction rules
│   ├── global_rules.yml        # Global rule definitions
│   └── signature_overrides/    # Signature-specific rules
├── signatures/                 # Layout signature storage
├── outputs/                    # Normalized JSON outputs
└── logs/                       # Interpretation logs
```

### Pipeline Usage Examples

#### Programmatic Usage
```python
from src.normalization.pipeline.document_pipeline import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline()

# Process a document
document, log_file = pipeline.process_document(
    file_path="path/to/document.pdf",
    max_cost_usd=1.0
)

# Access extracted data
print(f"Extracted {len(document.key_values)} fields")
for kv in document.key_values:
    print(f"{kv.key}: {kv.value} ({kv.extraction_method})")
```

#### Rule Configuration
Edit `rules/global_rules.yml` to add new extraction patterns:

```yaml
rules:
  - field_name: "custom_field"
    pattern: "(?:field)[\\s:]*([A-Z0-9-]+)"
    confidence: 0.8
    description: "Extract custom field"
    required: true
```

### Cost Management
- Set per-document cost limits
- Track LLM API usage in real-time
- Automatic cost-based field prioritization
- Comprehensive cost reporting

### Interpretability
- Every pipeline decision logged
- Stage-by-stage processing audit trail
- Cost attribution per operation
- Downloadable logs in CSV/JSON format

## Original RAG System

### Running the Web Interface
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

### Usage Example
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
- **NEW**: Add custom extraction rules and LLM providers to the hybrid pipeline.

## Architecture Decisions
See [docs/adr/README.md](docs/adr/README.md) for Architecture Decision Records.

## Evaluation and Experiment Tracking
- User feedback is logged and aggregated for precision/recall metrics.
- MLflow is used for experiment tracking and metric visualization.
- Model selection and evaluation are reflected in retrieval quality.
- **NEW**: Hybrid pipeline provides detailed cost and coverage tracking.

## License
MIT License (see LICENSE).
