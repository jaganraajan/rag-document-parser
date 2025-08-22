import requests

r = requests.get("http://localhost:9464/metrics")
lines = r.text.splitlines()
summary = {"success": 0, "error": 0, "errors": {}}
for line in lines:
    if line.startswith("documents_ingested_total") and "success" in line:
        summary["success"] += int(line.split()[-1])
    if line.startswith("ingestion_errors_total"):
        parts = line.split("{")
        if len(parts) > 1:
            label_part = parts[1].split("}")[0]
            if 'error_type="' in label_part:
                err_type = label_part.split('error_type="')[1].split('"')[0]
                summary["errors"].setdefault(err_type, 0)
                summary["errors"][err_type] += int(line.split()[-1])
                summary["error"] += int(line.split()[-1])
print(f"Documents ingested: {summary['success']}")
print(f"Ingestion errors: {summary['error']}")
for k, v in summary["errors"].items():
    print(f"  - {k}: {v}")