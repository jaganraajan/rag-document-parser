# Define the query
from dotenv import load_dotenv  # dev dependency
load_dotenv()
from src.storage.vector_store import semantic_query
from src.retrieval.hybrid_search import hybrid_search

def test_semantic_search():
    """Test traditional semantic search."""
    print("=== SEMANTIC SEARCH TEST ===")
    query = "existential meaning life"
    
    # Search the dense index
    results = semantic_query(query)
    print('Semantic search results:')
    print(results)
    print()

def test_hybrid_search():
    """Test hybrid search with reranking."""
    print("=== HYBRID SEARCH TEST ===")
    query = "existential meaning life"
    
    # Execute hybrid search
    results = hybrid_search(query, top_k_dense=5, top_k_sparse=20)
    
    print(f"\nHybrid search results for '{query}':")
    print(f"Total results: {len(results)}")
    print()
    
    # Display top results with detailed scoring
    for i, result in enumerate(results[:5]):  # Show top 5
        print(f"Result {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  Relevance Score: {result['relevance_score']:.4f}")
        print(f"  Source: {result['source']}")
        print(f"  Text: {result['text'][:100]}...")
        
        breakdown = result['score_breakdown']
        print(f"  Score Breakdown:")
        print(f"    Dense: {breakdown['dense_score']:.4f} -> normalized: {breakdown['normalized_dense']:.4f}")
        print(f"    Sparse: {breakdown['sparse_score']:.4f} -> normalized: {breakdown['normalized_sparse']:.4f}")
        print(f"    Overlap: {breakdown['overlap_score']:.4f}")
        print(f"    Final: {breakdown['final_score']:.4f}")
        print()

if __name__ == '__main__':
    # Test both search methods
    test_semantic_search()
    test_hybrid_search()