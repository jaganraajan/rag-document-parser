from .metrics import init_metrics, get_meter
import time

init_metrics()

m = get_meter()

# Counters
documents_ingested_total = m.create_counter(
    "documents_ingested_total", unit="1",
    description="Count of successfully ingested documents"
)
ingestion_errors_total = m.create_counter(
    "ingestion_errors_total", unit="1",
    description="Count of ingestion errors"
)
queries_total = m.create_counter(
    "queries_total", unit="1",
    description="Total queries issued"
)
query_errors_total = m.create_counter(
    "query_errors_total", unit="1",
    description="Total failed queries"
)

# Histograms
document_ingest_seconds = m.create_histogram(
    "document_ingest_seconds", unit="s",
    description="End-to-end document ingestion duration"
)
query_end_to_end_seconds = m.create_histogram(
    "query_end_to_end_seconds", unit="s",
    description="End-to-end query latency"
)
vector_search_seconds = m.create_histogram(
    "vector_search_seconds", unit="s",
    description="Latency of Pinecone vector search step"
)
retrieval_result_count = m.create_histogram(
    "retrieval_result_count", unit="1",
    description="Distribution of number of matches returned"
)

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