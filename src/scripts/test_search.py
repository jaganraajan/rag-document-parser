# Define the query
from dotenv import load_dotenv  # dev dependency
load_dotenv()
from src.storage.vector_store import semantic_query

if __name__ == '__main__':
    # A sentence (or fragment) you expect exists in a chunk
    query = "existential meaning"

    # Search the dense index
    results = semantic_query(query)
    print('results are')
    print(results)

    # # Print the results
    # for hit in results['result']['hits']:
    #         print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")