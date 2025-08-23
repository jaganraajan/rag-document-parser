from src.observability.metrics import init_metrics, get_meter

init_metrics()

# Prevent resource attribute duplication as metric labels
FORBIDDEN_LABELS = {"service.name", "deployment.environment"}

def safe_attrs(attrs: dict | None) -> dict:
    return {k: v for k, v in (attrs or {}).items() if k not in FORBIDDEN_LABELS}

meter = get_meter()

# Existing metrics (kept/centralized here for single source of truth)
documents_ingested_total = meter.create_counter("documents_ingested_total", description="Count of successfully ingested documents")
ingestion_errors_total = meter.create_counter("ingestion_errors_total", description="Count of ingestion errors")
document_ingest_seconds = meter.create_histogram("document_ingest_seconds", description="End-to-end document ingestion duration")

# New: ingestion volume and phase metrics
chunk_count_total = meter.create_counter("chunk_count_total", description="Number of chunks produced by ingestion")
chunk_chars_sum = meter.create_counter("chunk_chars_sum", description="Total characters across all chunks")
pdf_pages_total = meter.create_counter("pdf_pages_total", description="Number of PDF pages processed")
text_extraction_failures_total = meter.create_counter("text_extraction_failures_total", description="Count of page-level text extraction failures")

pdf_load_seconds = meter.create_histogram("pdf_load_seconds", description="Time to load and extract text from PDF")
metadata_extract_seconds = meter.create_histogram("metadata_extract_seconds", description="Time to extract and normalize metadata")
chunking_seconds = meter.create_histogram("chunking_seconds", description="Time to create text chunks")

# Query-side existing metrics
queries_total = meter.create_counter("queries_total", description="Count of vector queries")
query_errors_total = meter.create_counter("query_errors_total", description="Count of query errors")
query_end_to_end_seconds = meter.create_histogram("query_end_to_end_seconds", description="End-to-end query duration")
vector_search_seconds = meter.create_histogram("vector_search_seconds", description="Vector search duration")
retrieval_result_count = meter.create_counter("retrieval_result_count", description="Number of results returned by retrieval")

# New: vector store writes
upsert_records_total = meter.create_counter("upsert_records_total", description="Total records upserted to the vector store")
upsert_errors_total = meter.create_counter("upsert_errors_total", description="Upsert errors in vector store")
upsert_batch_seconds = meter.create_histogram("upsert_batch_seconds", description="Latency of vector store batch upserts")

# New: query effectiveness
retrieval_top_score = meter.create_histogram("retrieval_top_score", description="Top match score per query")
retrieval_any_result_total = meter.create_counter("retrieval_any_result_total", description="Queries that returned at least one result")
no_result_queries_total = meter.create_counter("no_result_queries_total", description="Queries that returned zero results")

__all__ = [
    # helper
    "safe_attrs",
    # ingestion
    "documents_ingested_total", "ingestion_errors_total", "document_ingest_seconds",
    "chunk_count_total", "chunk_chars_sum", "pdf_pages_total", "text_extraction_failures_total",
    "pdf_load_seconds", "metadata_extract_seconds", "chunking_seconds",
    # query existing
    "queries_total", "query_errors_total", "query_end_to_end_seconds", "vector_search_seconds", "retrieval_result_count",
    # vector store writes
    "upsert_records_total", "upsert_errors_total", "upsert_batch_seconds",
    # query effectiveness
    "retrieval_top_score", "retrieval_any_result_total", "no_result_queries_total",
]

# (Optional) simple helper context managers
from contextlib import contextmanager
import time as _time

@contextmanager
def time_histogram(hist_instrument, attributes: dict | None = None):
    start = _time.perf_counter()
    try:
        yield
        duration = _time.perf_counter() - start
        hist_instrument.record(duration, attributes or {})
    except Exception:
        duration = _time.perf_counter() - start
        hist_instrument.record(duration, {**(attributes or {}), "status": "error"})
        raise