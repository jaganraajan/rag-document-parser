import argparse
import mlflow
from collections import defaultdict

def load_feedback_log(feedback_log_path):
    import json
    feedbacks = []
    with open(feedback_log_path, "r") as f:
        for line in f:
            feedbacks.append(json.loads(line))
    return feedbacks

def feedback_metrics(feedbacks, k):
    if not feedbacks:
        return 0.0, 0.0
    feedback_values = [f["feedback"] for f in feedbacks]
    precision = sum(feedback_values) / len(feedback_values)
    hit_rate = 1.0 if any(feedback_values) else 0.0
    return precision, hit_rate

def main(feedback_log_path, k=5):
    feedbacks = load_feedback_log(feedback_log_path)
    grouped = defaultdict(list)
    for fb in feedbacks:
        dense = fb.get("dense_model", "unknown")
        rerank = fb.get("rerank_model", None)
        grouped[(dense, rerank)].append(fb)

    dense_metrics = defaultdict(list)
    rerank_metrics = defaultdict(list)
    all_metrics = {}

    mlflow.set_experiment("rag-feedback-eval")
    with mlflow.start_run():
        mlflow.log_param("top_k", k)
        for (dense_model, rerank_model), group in grouped.items():
            avg_precision, avg_hit_rate = feedback_metrics(group, k)
            tag = f"{dense_model}_{rerank_model or 'none'}"
            all_metrics[(dense_model, rerank_model)] = (avg_precision, avg_hit_rate)
            mlflow.log_metric(f"precision_at_{k}_{tag}", avg_precision)
            mlflow.log_metric(f"hit_rate_at_{k}_{tag}", avg_hit_rate)
            print(f"[{tag}] Precision@{k}: {avg_precision:.3f} | Hit Rate@{k}: {avg_hit_rate:.3f}")

            # Collect for comparison
            if not rerank_model or rerank_model == "none":
                dense_metrics[dense_model].append(avg_precision)
            else:
                rerank_metrics[rerank_model].append(avg_precision)

        mlflow.log_artifact(feedback_log_path)

        # Compare dense models
        print("\n=== Dense Model Comparison (no reranker) ===")
        best_dense = max(dense_metrics.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
        for model, scores in dense_metrics.items():
            avg = sum(scores)/len(scores) if scores else 0
            print(f"Dense Model '{model}': Avg Precision@{k} = {avg:.3f}")
        print(f"Best Dense Model: {best_dense[0]} (Avg Precision@{k} = {sum(best_dense[1])/len(best_dense[1]):.3f})")

        # Compare reranker models
        print("\n=== Reranker Model Comparison ===")
        best_rerank = max(rerank_metrics.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
        for model, scores in rerank_metrics.items():
            avg = sum(scores)/len(scores) if scores else 0
            print(f"Reranker Model '{model}': Avg Precision@{k} = {avg:.3f}")
        print(f"Best Reranker Model: {best_rerank[0]} (Avg Precision@{k} = {sum(best_rerank[1])/len(best_rerank[1]):.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_log", type=str, default="feedback_log.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.feedback_log, args.k)