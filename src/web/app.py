import os
from flask import Flask, request, render_template, jsonify
from ..storage.search_wrapper import search_with_metadata
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

from src.rerank.cross_encoder import CrossEncoderReranker

reranker = CrossEncoderReranker()


# Optional config usage
try:
    from src.web.config import select_config 
except ImportError:
    select_config = None

app = Flask(__name__)
if select_config:
    app.config.from_object(select_config())

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-2.5-flash"
else:
    print("Warning: GEMINI_API_KEY not set. Answer generation will not work.")

def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def build_context_from_results(dense_results, sparse_results, max_total_chars: int = 8000) -> str:
    """Build context string from search results with truncation."""
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(dense_results + sparse_results):
        chunk_text = result.get("text", "")
        truncated_chunk = truncate_text(chunk_text, 500)
        
        chunk_section = f"\n--- Chunk {i+1} ---\n{truncated_chunk}"
        
        if total_chars + len(chunk_section) > max_total_chars:
            break
            
        context_parts.append(chunk_section)
        total_chars += len(chunk_section)
    
    return "\n".join(context_parts)

def highlight(text: str, query: str):
    """
    Very simple highlight: wrap case-insensitive occurrences
    of each query token (length > 2) in <mark>.
    """
    import re
    tokens = [t for t in query.split() if len(t) > 2]
    highlighted = text
    for t in tokens:
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
    return highlighted

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("index.html", error="Enter a query.")
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5

    results = search_with_metadata(q, top_k=k)
    # Add highlighted text
    for r in results["dense_results"]:
        r["highlighted"] = highlight(r["text"], q)
    for r in results["sparse_results"]:
        r["highlighted"] = highlight(r["text"], q)

    return render_template(
        "results.html",
        query=q,
        dense_results=results["dense_results"],
        sparse_results=results["sparse_results"],
        results=results["dense_results"] + results["sparse_results"],  # For backward compatibility
        k=k
    )

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    k = int(request.args.get("k", 5))
    print(f"API search for query: {q} with top_k={k}")
    # results = search_with_metadata(q, top_k=k)
    # return jsonify({
    #     "query": q,
    #     "top_k": k,
    #     "results": results
    # })

@app.route("/rerank", methods=["GET"])
def rerank():
    """Rerank search results and render a results page."""
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return render_template("index.html", error="Enter a query.")

        try:
            k = int(request.args.get("k", 5))
        except ValueError:
            k = 5

        # Get the original search results
        original_results = search_with_metadata(query, top_k=k)
        if not original_results:
            return render_template("results.html", query=query, dense_results=[], sparse_results=[], reranked_results=[], k=k, error="No search results found.")

        dense_results = original_results.get("dense_results", [])
        sparse_results = original_results.get("sparse_results", [])

        print(f"Reranking {len(dense_results)} dense results and {len(sparse_results)} sparse results for query: {query}")
        for r in dense_results:
            r["highlighted"] = highlight(r["text"], query)
        for r in sparse_results:
            r["highlighted"] = highlight(r["text"], query)

        reranked_dense_results = reranker.rerank(query, dense_results, text_key="text", top_n=5)

        context = build_context_from_results(reranked_dense_results, sparse_results)
        print('context is', context)
        # Optionally, remove 'raw' key for template safety
        # for r in reranked_results.get("dense_results", []):
        #     r.pop("raw", None)
        # for r in reranked_results.get("sparse_results", []):
        #     r.pop("raw", None)

        return render_template(
            "results.html",
            query=query,
            dense_results=reranked_dense_results,
            sparse_results=sparse_results,
            k=k,
            reranked=True,
            context=context
        )
    except Exception as e:
        print(f"Error reranking results: {str(e)}")
        return render_template("results.html", query="", dense_results=[], sparse_results=[], reranked_results=[], k=5, error="An error occurred while reranking results.")

@app.route("/generate_answer", methods=["POST"])
def generate_answer():
    print('in generate_answer')
    """Generate an answer using Gemini AI from retrieved context."""
    try:
        # Check if Gemini API key is configured
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key not configured"}), 500
        
        # Get query from request JSON
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing query in request body"}), 400
        
        query = data["query"].strip()
        context = data["context"].strip()

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Construct prompt for Gemini
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the user query. If the answer is not in the context, say you do not have enough information.

            Query: {query}

            Context:
            {context}

            Answer:"""
        
        # Call Gemini API
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return jsonify({"error": "Failed to generate answer"}), 500
        
        return jsonify({"answer": response.text.strip()})
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return jsonify({"error": "An error occurred while generating the answer"}), 500

if __name__ == "__main__":
    # Dev only (Gunicorn will NOT execute this block)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)