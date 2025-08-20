# ADR 0003: Retrieval Evaluation Methodology

Status: Accepted  
Date: 2025-08-20

## Context
Need a reproducible baseline to track retrieval quality before adding reranking, hybrid search, or generation. Current script (`src/scripts/evaluate_retrieval.py`) computes Precision@k, Recall@k, Hit Rate, with relevance determined by either exact ID match or Jaccard similarity over token sets using a threshold (default 0.3) when IDs are absent.

## Decision
Adopt evaluation script using: 
- Input dataset (JSON/CSV) of queries + expected chunks and/or expected IDs.
- Relevance matching: (1) ID match if provided, else (2) Jaccard similarity >= 0.3 (lowercase word set) between expected chunk text and retrieved chunk text.
- Metrics: Precision@k, Recall@k, Hit Rate per query + averages.

## Rationale
- Lightweight: no external embedding-based evaluator required yet.
- Transparent & deterministic threshold fosters quick iteration.
- Supports partial test mode for offline development.

## Consequences
+ Enables regression detection for retrieval changes.
+ Serves as baseline to measure impact of chunking or model shifts.
- Jaccard similarity is crude—misses semantic paraphrases.
- Threshold tuning may be dataset-sensitive.
- Does not measure ranking quality beyond binary relevance.

## Alternatives
- Embedding cosine similarity (needs consistent embedding pipeline + thresholds per model).
- Cross-encoder relevance scoring (higher latency, requires model hosting or API).
- Manual labeling / human evaluation (costly & slower feedback loop).

## Follow-ups
- Introduce nDCG or MRR when reranking added.
- Replace / augment Jaccard with embedding similarity once local embedding strategy exists.
- Add ADR when evaluation introduces semantic or LLM-based judgments.