import time
from dotenv import load_dotenv
load_dotenv()
from src.storage.sparse_store import sparse_query  # if exists
from src.storage.vector_store import semantic_query
from src.observability.tracing import init_tracing, get_tracer
from src.observability.metrics import init_metrics, get_meter, force_flush
init_metrics(export_interval_sec=2)  # shorter for CLI
meter = get_meter()
init_tracing(console=True)
tracer = get_tracer()

# Create instruments AFTER init
query_counter = meter.create_counter("cli_queries_total")
query_latency = meter.create_histogram("cli_query_latency_ms")

if __name__ == '__main__':
    query = "existential meaning life"
    start = time.perf_counter()
    with tracer.start_as_current_span("cli.search"):
        results = semantic_query(query)
    dur_ms = (time.perf_counter() - start) * 1000
    query_counter.add(1, {"status": "ok"})
    query_latency.record(dur_ms, {"status": "ok"})
    print(f"Query='{query}' duration_ms={dur_ms:.1f}")
    force_flush()  # ensure export before exit