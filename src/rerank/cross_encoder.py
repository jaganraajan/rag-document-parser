from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def rerank(self, query, results, text_key="text", top_n=5):
        """
        Args:
            query (str): The user query
            results (list): List of dicts, each with a 'text' key (the retrieved chunk)
            text_key (str): The key in each result dict for the chunk text
            top_n (int): How many reranked results to return

        Returns:
            List of top_n results, sorted by cross-encoder score (descending)
        """

        pairs = [(query, r[text_key]) for r in results]
        # Tokenize all pairs
        encodings = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = self.model(**encodings).logits
            print("Model logits:", logits)
            scores = logits.squeeze(-1).cpu().numpy()
            # scores = np.nan_to_num(scores, nan=0.0)
        print('Scores:', scores[:10])
        # Attach scores and rerank
        for res, score in zip(results, scores):
            res["rerank_score"] = float(score)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        print('reranked results:', reranked[:top_n])
        return reranked[:top_n]