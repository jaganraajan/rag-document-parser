# ADR 0001: Use Pinecone Managed Vector Index with Built-in Embedding

Status: Accepted  
Date: 2025-08-20

## Context
The system requires fast semantic retrieval for PDF-sourced chunks with minimal operational overhead. Options considered: (1) Pinecone managed index with server-side embedding pipeline, (2) local embedding + FAISS/pgvector, (3) Elasticsearch dense vector, (4) hybrid from day one. Early phase priorities: speed of iteration, low ops, predictable latency, and avoiding premature scaling complexity.

## Decision
Adopt a Pinecone managed index named `philosophy-rag` using model `llama-text-embed-v2` via `create_index_for_model` and a simple field map `{"text": "chunk_text"}`. Use a single namespace `__default__` initially.

## Rationale
- Time-to-value: eliminates separate embedding job and custom batching code.
- Operational simplicity: managed scaling + availability.
- Consistency: embedding & index upgrades are coordinated by provider.
- Adequate for current scale (few thousands to low millions of vectors expected initially).

## Consequences
+ Rapid prototype-to-usable system path.
+ Reduced risk of embedding/index drift.
+ Simplified ingestion code (no explicit embedding calls).
- Vendor lock-in (API & model coupling).
- Less visibility into embedding internals / upgrade scheduling.
- Reindex cost when changing model.

## Alternatives
- Local embedding + FAISS (lower variable cost; higher infra + build-your-own replication & monitoring).
- pgvector in Postgres (simpler operations for moderate scale; weaker high-scale latency profile).
- Elasticsearch/OpenSearch dense vectors (built-in hybrid potential; higher configuration overhead now).

## Follow-ups
- Add ADR when introducing multi-namespace tenancy or switching model version.
- Reevaluate at vector count threshold (TBD after metrics) or cost triggers.