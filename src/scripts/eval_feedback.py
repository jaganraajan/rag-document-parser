import json
import mlflow
import argparse
from collections import defaultdict
from operator import itemgetter

def load_feedback_log(path):
    feedbacks_by_query = defaultdict(list)
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            feedbacks_by_query[entry["query"]].append(entry)
    return feedbacks_by_query

def precision_at_k(feedbacks, k):
    # feedbacks: list of feedback dicts, sorted by rank (timestamp ascending)
    return sum(f['feedback'] == 1 for f in feedbacks[:k]) / k if k > 0 else 0.0

def hit_rate(feedbacks, k):
    return int(any(f['feedback'] == 1 for f in feedbacks[:k])) if k > 0 else 0

def feedback_metrics(feedbacks_by_query, k):
    precisions = []
    hits = []
    for query, feedbacks in feedbacks_by_query.items():
        # Sort feedbacks for each query by timestamp (earliest first, or keep as-is if order reflects ranking)
        feedbacks_sorted = sorted(feedbacks, key=itemgetter("timestamp"))
        precisions.append(precision_at_k(feedbacks_sorted, k))
        hits.append(hit_rate(feedbacks_sorted, k))
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_hit_rate = sum(hits) / len(hits) if hits else 0
    return avg_precision, avg_hit_rate

def main(feedback_log_path, k=5):
    feedbacks_by_query = load_feedback_log(feedback_log_path)
    avg_precision, avg_hit_rate = feedback_metrics(feedbacks_by_query, k)
    mlflow.set_experiment("rag-feedback-eval")
    with mlflow.start_run():
        mlflow.log_param("top_k", k)
        mlflow.log_metric(f"feedback_precision_at_{k}", avg_precision)
        mlflow.log_metric(f"feedback_hit_rate_at_{k}", avg_hit_rate)
        mlflow.log_artifact(feedback_log_path)
        print(f"Precision@{k} (user feedback): {avg_precision:.3f}")
        print(f"Hit Rate@{k}: {avg_hit_rate:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_log", type=str, default="feedback_log.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.feedback_log, args.k)