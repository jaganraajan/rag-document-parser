import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

_INITIALIZED = False

def init_tracing(
    service_name: str = "rag-document-parser",
    console: bool = False,
    sample_ratio: float = 1.0
):
    global _INITIALIZED
    if _INITIALIZED:
        return

    resource = Resource.create({
        "service.name": service_name,
        "deployment.environment": os.getenv("ENV", "dev")
    })

    sampler = ParentBased(TraceIdRatioBased(sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)
    trace.set_tracer_provider(provider)

    # OTLP endpoint (defaults to localhost:4318 if not provided)
    # otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
    print("[otel] resolved exporter endpoint:", otlp_exporter._endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Instrument outbound HTTP (Pinecone client uses requests)
    RequestsInstrumentor().instrument()

    _INITIALIZED = True

def get_tracer():
    return trace.get_tracer("rag-document-parser")