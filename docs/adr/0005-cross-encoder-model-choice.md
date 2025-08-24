# ADR 0004: Choice of Cross-Encoder Model for Reranking

## Status

Accepted

## Context

To improve the relevance of retrieved document chunks in the RAG Document Parser, we decided to add a reranking step using a cross-encoder. This model is used after the initial dense retrieval (from Pinecone) to rescore and reorder top candidate chunks based on their actual semantic match to the user query.

Several cross-encoder models are available, with different trade-offs between accuracy, latency, and resource requirements.

## Decision

I chose to use the model `cross-encoder/ms-marco-MiniLM-L-12-v2` for reranking.

### Rationale:

- **Proven Effectiveness:** This model is widely adopted for passage reranking tasks and consistently demonstrates strong performance on the MS MARCO benchmark and in real-world information retrieval scenarios.
- **Balance of Quality and Speed:** While LLM-based cross-encoders can offer marginally higher accuracy, MiniLM-L-12-v2 provides an excellent balance of low latency and high relevance for top-N reranking, especially when used on pools of 20-100 candidates.
- **Model Size:** At 33M parameters, it is substantially lighter than BERT or RoBERTa-based cross-encoders, enabling feasible deployment on CPUs or small GPUs without significant hardware investment.
- **Open Availability:** The model is available via HuggingFace Transformers with a permissive license and no additional cost.
- **Community Adoption:** It is a de facto standard in open-source RAG and search pipelines and has been tested extensively.

## Consequences

- The reranking step will be both fast and accurate for document retrieval in production.
- The model can be swapped for a different cross-encoder (or a larger LLM) in the future if requirements or hardware change.
- By using a standard model, we benefit from compatibility, community support, and reproducibility.

---

*See also: [MiniLM-L-12-v2 on HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)*