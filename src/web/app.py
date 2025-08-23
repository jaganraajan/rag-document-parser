import os
from flask import Flask, request, render_template, jsonify
from ..storage.search_wrapper import search_with_metadata
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()


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

def build_context_from_results(results, max_total_chars: int = 8000) -> str:
    """Build context string from search results with truncation."""
    context_parts = []
    total_chars = 0
    
    # Handle both old and new result formats
    if isinstance(results, dict) and 'dense_results' in results:
        # New format: iterate through both dense and sparse results
        all_results = results.get('dense_results', []) + results.get('sparse_results', [])
    else:
        # Old format: assume it's a list
        all_results = results if isinstance(results, list) else []
    
    for i, result in enumerate(all_results):
        chunk_text = result.get("text", "")
        truncated_chunk = truncate_text(chunk_text, 500)
        
        chunk_section = f"\n--- Chunk {i+1} ---\n{truncated_chunk}"
        
        if total_chars + len(chunk_section) > max_total_chars:
            break
            
        context_parts.append(chunk_section)
        total_chars += len(chunk_section)
    
    return "\n".join(context_parts)

def rerank_results(query: str, dense_results: list, sparse_results: list) -> dict:
    """
    Placeholder reranking function. 
    In a real implementation, this would use a cross-encoder model.
    For now, we'll implement a simple strategy:
    1. Combine all results
    2. Re-sort by score in descending order
    3. Apply some filtering (e.g., minimum score threshold)
    4. Return top results split back into dense/sparse
    """
    import random
    
    # Add a source type to track which results came from where
    for result in dense_results:
        result['source_type'] = 'dense'
    for result in sparse_results:
        result['source_type'] = 'sparse'
    
    # Combine all results
    all_results = dense_results + sparse_results
    
    # Simple reranking: sort by score descending
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Apply a minimum score threshold (placeholder)
    min_score = 0.1
    filtered_results = [r for r in all_results if r.get('score', 0) >= min_score]
    
    # Limit to top results (e.g., top 10)
    top_results = filtered_results[:10]
    
    # Add a small random boost to simulate cross-encoder reranking
    # In practice, this would be replaced with actual cross-encoder scores
    for result in top_results:
        # Simulate cross-encoder giving a boost/penalty between -0.1 and +0.1
        cross_encoder_boost = (random.random() - 0.5) * 0.2
        result['reranked_score'] = result.get('score', 0) + cross_encoder_boost
    
    # Re-sort by the new reranked score
    top_results.sort(key=lambda x: x.get('reranked_score', 0), reverse=True)
    
    # Update the score field to reflect reranked scores
    for result in top_results:
        result['score'] = result.get('reranked_score', result.get('score', 0))
        # Clean up temporary fields
        if 'reranked_score' in result:
            del result['reranked_score']
    
    # Split results back into dense and sparse (maintaining order)
    reranked_dense = [r for r in top_results if r.get('source_type') == 'dense']
    reranked_sparse = [r for r in top_results if r.get('source_type') == 'sparse']
    
    # Clean up source_type field
    for result in reranked_dense + reranked_sparse:
        if 'source_type' in result:
            del result['source_type']
    
    return {
        'dense_results': reranked_dense,
        'sparse_results': reranked_sparse
    }

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

@app.route("/rerank", methods=["POST"])
def rerank():
    """Rerank search results using a placeholder cross-encoder strategy."""
    try:
        # Get query from request JSON
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing query in request body"}), 400
        
        query = data["query"].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Get the original search results
        # We need to re-run the search to get the original results
        # In a more sophisticated implementation, we might cache the original results
        k = data.get("k", 5)  # Default to 5 if not provided
        
        original_results = search_with_metadata(query, top_k=k)
        
        if not original_results:
            return jsonify({"error": "No search results found"}), 400
        
        # Extract results
        dense_results = original_results.get("dense_results", [])
        sparse_results = original_results.get("sparse_results", [])
        
        # Apply highlighting to the original results before reranking
        for r in dense_results:
            r["highlighted"] = highlight(r["text"], query)
        for r in sparse_results:
            r["highlighted"] = highlight(r["text"], query)
        
        # Perform reranking
        reranked_results = rerank_results(query, dense_results, sparse_results)
        
        return jsonify(reranked_results)
        
    except Exception as e:
        print(f"Error reranking results: {str(e)}")
        return jsonify({"error": "An error occurred while reranking results"}), 500

@app.route("/generate_answer", methods=["POST"])
def generate_answer():
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
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Retrieve context using semantic search
        top_k = 5  # configurable
        results = search_with_metadata(query, top_k=top_k)
        
        if not results:
            return jsonify({"error": "No relevant context found"}), 400
        
        # Build context from retrieved chunks
        context = build_context_from_results(results)
        
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