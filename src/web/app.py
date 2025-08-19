import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
load_dotenv()

# from src.storage.search_wrapper import search_with_metadata

app = Flask(__name__)

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

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("index.html", error="Enter a query.")
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    # results = search_with_metadata(q, top_k=k)
    # # Add highlighted text
    # for r in results:
    #     r["highlighted"] = highlight(r["text"], q)
    return render_template("results.html", query=q, results=[], k=k)

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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)