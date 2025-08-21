import os, time, random
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry import metrics

def init_demo_metrics(interval_sec=2):
    res = Resource.create({"service.name":"demo-service"})
    exporter = OTLPMetricExporter()  # uses OTEL_EXPORTER_OTLP_ENDPOINT
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=interval_sec*1000)
    provider = MeterProvider(resource=res, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("demo.meter")
    counter = meter.create_counter("demo_requests_total")
    latency = meter.create_histogram("demo_request_latency_ms")
    state = {"val":0}
    def cpu_obs(_):
        # fake gauge
        return [metrics.Observation(value=random.uniform(5,50), attributes={"instrument": "demo_cpu_utilization"})]
    try:
        # SDK 1.36 uses create_observable_gauge
        meter.create_observable_gauge("demo_cpu_utilization", callbacks=[cpu_obs])
    except Exception:
        pass
    return counter, latency

def main():
    c, h = init_demo_metrics()
    for i in range(15):
        ms = random.uniform(20, 400)
        status = "ok" if random.random() > 0.1 else "error"
        c.add(1, {"status":status})
        h.record(ms, {"status":status})
        print(f"Emitted sample #{i+1} latency={ms:.1f}ms status={status}")
        time.sleep(1.0)
    # force flush
    try:
        metrics.get_meter_provider().force_flush()
    except Exception:
        pass
    print("Done. Curl the Prometheus endpoint now.")

if __name__ == "__main__":
    # Ensure env defaults if not set
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL","http/protobuf")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT","http://localhost:4318")
    main()