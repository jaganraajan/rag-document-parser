import os
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation

_INITIALIZED = False
_METER = None
_PROVIDER: MeterProvider | None = None

def init_metrics(export_interval_sec: int = 10):
    global _INITIALIZED, _METER, _PROVIDER
    if _INITIALIZED:
        return

    resource = Resource.create({
        "service.name": "rag-document-parser",
        "deployment.environment": os.getenv("ENV", "dev")
    })

    # endpoint = (
    #     os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    #     or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    #     or "http://localhost:4318/v1/metrics"
    #     # "http://localhost:4318/v1/metrics"
    # )
    endpoint = "http://localhost:4318/v1/metrics"
    print("[otel] resolved metrics exporter endpoint:", endpoint)
    exporter = OTLPMetricExporter(endpoint=endpoint)
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=export_interval_sec * 1000
    )

    # Example custom view: better latency buckets for query latency
    views = [
        View(
            instrument_name="query_end_to_end_seconds",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
            ),
        )
    ]

    provider = MeterProvider(resource=resource, metric_readers=[reader], views=views)
    metrics.set_meter_provider(provider)
    _PROVIDER = provider
    _METER = metrics.get_meter("rag-document-parser")
    _INITIALIZED = True

def force_flush(timeout_millis: int = 3000):
    if not _INITIALIZED or _PROVIDER is None:
        return
    readers = getattr(_PROVIDER, "_metric_readers", [])
    for r in readers:
        flush = getattr(r, "force_flush", None)
        if callable(flush):
            try:
                flush(timeout_millis=timeout_millis)
            except Exception:
                pass

def get_meter():
    return metrics.get_meter("rag-document-parser")