# ADR 0002: Chunking Strategy (size=500, overlap=80)

Status: Accepted  
Date: 2025-08-20

## Context
Retrieval quality & embedding cost hinge on chunk segmentation. Current ingestion uses fixed-size character chunking: size=500, overlap=80. Alternatives include semantic splitting, sentence boundary segmentation, or adaptive sizing. Early project stage needs a deterministic approach with minimal dependencies.

## Decision
Use fixed-length character chunking with `chunk_size=500` and `overlap=80` for initial ingestion (`chunk_document` function). Store each chunk + normalized metadata.

## Rationale
- Simplicity: no external NLP libs / sentence tokenizers.
- Balanced tradeoff: 500 chars approximates a paragraph for philosophical prose (improves semantic density without excess length).
- Overlap preserves context continuity across boundary edges (reduces truncation-induced recall loss).
- Deterministic => easier evaluation & baseline metric comparison.

## Consequences
+ Predictable cost per document (roughly len(text)/(chunk_size - overlap)).
+ Facilitates establishing retrieval baselines before optimizing segmentation.
- May split semantic units mid-sentence (can slightly hurt reranking later).
- Overlap inflates total vector count (~ +19% given (overlap)/(chunk_size - overlap)).
- Not adaptive to heterogeneous documents (tables, very short sections).

## Alternatives
- Sentence/paragraph boundary splitting (needs NLP parsing, variable lengths). 
- Semantic splitting (e.g., embedding-based coherence scoring; more CPU/time upfront). 
- Adaptive chunking based on token counts (requires tokenizer dependency tied to future LLM model). 

## Follow-ups
- Revisit after retrieval evaluation + metrics show distribution of chunk hit frequency.
- Consider semantic or heading-aware splitting before adding reranking.
- Add ADR if/when switching to token-based sizing.