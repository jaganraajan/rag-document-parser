# Architecture Decision Records (ADR)

This directory contains ADRs documenting key technical choices for the RAG Document Parser project.

## Key Decisions

- **Pinecone Managed Index**:  
  Adopted Pinecone's managed vector index for both dense and sparse retrieval. Model selection for embeddings is exposed in the UI, enabling flexible experimentation and regression testing.

- **Chunking Strategy**:  
  Uses fixed-length character chunking (default: 500 chars, 80 overlap) for document segmentation, balancing recall and semantic density.

- **Metadata Extraction and Normalization**:  
  Metadata is extracted from PDFs and normalized for consistent indexing and retrieval.

- **Search Architecture**:  
  Hybrid search combines dense and sparse vector results. Users can select models for both dense and sparse embeddings.

- **Reranking**:  
  Reranking is integrated post-retrieval. Multiple models (including cross-encoder and Pinecone-managed rerankers) are available via dropdown. Experimentation is supported and evaluation metrics (Precision@k, etc.) are tracked.

- **User Feedback Evaluation**:  
  Explicit user feedback is logged per search result, and used for computing retrieval metrics. Feedback logs are evaluated and tracked via MLflow.

- **Observability**:  
  OpenTelemetry is integrated for ingestion and query tracing, with metrics for latency, failures, and throughput.

## How ADRs Are Organized

Each ADR addresses a specific decision:
- Technology or library choices (e.g., Pinecone, Gemini, OpenTelemetry).
- Algorithmic or architectural tradeoffs (e.g., chunking, reranking).
- Evaluation and feedback mechanisms.

See individual ADR files for details.

## Updating or Adding ADRs

When making significant changes to the codebase or architecture, add a new ADR documenting the rationale, alternatives, and consequences.

## References
- [Pinecone Managed Index](./0001-pinecone-managed-index.md)
- [Chunking Strategy](./0002-chunking-strategy.md)
- [Retrieval Evaluation Methodology](./0003-retrieval-evaluation-methodology.md)
- [Observability Strategy](./0004-observability-strategy.md)
- [Cross-Encoder Model Choice](./0005-cross-encoder-model-choice.md)
- [Feedback Evaluation Harness](./0006-feedback-evaluation-harness.md)