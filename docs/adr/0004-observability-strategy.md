# ADR 0004: Observability Strategy (Tracing-first)

Status: Accepted  
Date: 2025-08-20

## Context
The project recently integrated OpenTelemetry tracing (ingestion → vector store → query). Decisions needed: order of observability layers (traces, metrics, logs), privacy posture, and sampling approach. System is still evolving; root latency & structural visibility are higher priority than aggregate dashboards initially.

## Decision
Adopt tracing-first observability with OpenTelemetry:
- Spans: ingestion.run, document.ingest, pdf.load, metadata.extract, metadata.normalize, text.chunk, vector.ensure_index, vector.records.prepare, vector.upsert, query.request, vector.search.
- Default sampler: ParentBased + TraceIdRatioBased(1.0) in dev; allow env override later.
- Exclude raw document text and full query strings from span attributes (only lengths / counts).
- Metrics layer to be added in a subsequent phase (separate ADR) once baseline spans validated.

## Rationale
- Traces expose structural bottlenecks early (e.g., PDF parsing vs upsert time) without upfront metrics schema design.
- Minimizes premature metric cardinality decisions.
- Aligns with planned future addition of metrics + exemplars referencing trace IDs.

## Consequences
+ Rapid feedback on ingestion & query performance.
+ Foundation for adding metrics and logs with trace correlation.
- Lacks aggregate SLO visibility until metrics added.
- Potential sampling tweak needed as volume scales.

## Alternatives
- Metrics-first (faster SLO definition; less structural context initially).
- Logs-first (simple but harder to correlate & aggregate systematically).
- Distributed tracing via another vendor (increases lock-in early).

## Follow-ups
- Add metrics (latency histograms, error counters) in next phase.
- Consider exemplar integration for latency histograms.
- Add ADR when introducing log enrichment policy (trace_id injection, redaction rules).