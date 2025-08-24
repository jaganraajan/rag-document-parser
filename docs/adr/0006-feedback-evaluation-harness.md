# ADR-00XX: Evaluation Harness Uses User Feedback as Relevance Signal

## Status

Proposed

## Context

Measuring the real-world quality of retrieval in a Retrieval-Augmented Generation (RAG) system is challenging. While traditional evaluation relies on static "ground truth" datasets (queries with pre-labeled relevant chunks), these are often expensive to build, may not represent real user needs, and can drift over time as documents or user behavior change.

Our RAG system is deployed with a web interface where users submit queries and are shown ranked results (document chunks). This provides a unique opportunity to gather explicit, ongoing feedback from real users on the relevance of retrieved results.

By capturing this feedback, we can evaluate and track retrieval performance using metrics that reflect true user satisfaction and real-world scenarios, rather than relying solely on offline, synthetic datasets.

## Decision

We will implement an evaluation harness that uses explicit user feedback as the primary relevance signal for measuring retrieval quality. The mechanism is as follows:

- **Feedback Logging:**  
  For each query and search result presented to the user, we log whether the user marked each result as "relevant" (positive feedback) or "not relevant" (negative feedback) in a structured JSONL file.

- **Evaluation Metrics:**  
  We will compute key retrieval metrics (such as Precision@k, Hit Rate@k, etc.) by aggregating user feedback across many sessions. For each query, a result is considered "relevant" if the user gave positive feedback.

- **Evaluation Harness:**  
  A Python script will process the feedback logs, compute metrics, and log the results to MLflow for experiment tracking and visualization.

- **Artifact Logging:**  
  The feedback logs and metric reports will be stored as MLflow artifacts to enable reproducibility and auditability.

## Consequences

**Pros:**
- Evaluation reflects real user needs and behavior.
- No need for expensive, static ground-truth annotations.
- Enables continuous, real-time quality monitoring as the system evolves.
- Can track improvements and regressions after model or index changes.

**Cons:**
- User feedback can be noisy or sparse (not all users provide feedback).
- Feedback may be biased by UI or user population.
- Interpretation requires aggregating enough sessions for stability.

**Mitigations:**
- Regularly review feedback data quality and patterns.
- Optionally supplement with curated test sets for regression evaluation.
- Use aggregation and smoothing to reduce the impact of outliers.

## Implementation

- **Frontend:**  
  UI presents "Relevant" / "Not Relevant" buttons for each search result. On click, feedback is sent to the backend.

- **Backend:**  
  Feedback is stored as JSONL (one record per user action) with fields: query, chunk_id, feedback, chunk_text, timestamp.

- **Evaluation:**  
  Script computes Precision@k, Hit Rate@k, etc., using feedback as the relevance label. Metrics are logged to MLflow.

- **Documentation:**  
  This ADR and related code documentation describe the feedback-driven evaluation mechanism.

## Alternatives Considered

- Relying solely on static, hand-labeled gold sets (less representative, hard to maintain).
- Only using operational metrics (latency, throughput) without real relevance evaluation.

## References

- [MLflow: Open Source Experiment Tracking](https://mlflow.org/)
- [Precision@k, Hit Rate, and Retrieval Metrics in IR](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
