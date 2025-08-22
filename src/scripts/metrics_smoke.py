import time, random
from src.observability.metrics import init_metrics
from src.observability.instruments import (
    query_end_to_end_seconds, queries_total
)

init_metrics(export_interval_sec=2)
for _ in range(5):
    dur = random.uniform(0.05, 0.3)
    query_end_to_end_seconds.record(dur, {"status": "success"})
    queries_total.add(1, {"source": "smoke"})
    time.sleep(0.2)

print("Sleeping to allow export...")
time.sleep(3)