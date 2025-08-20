from dotenv import load_dotenv
load_dotenv()
from src.storage.sparse_store import sparse_query  # if exists
from src.storage.vector_store import semantic_query
from src.observability.tracing import init_tracing, get_tracer

init_tracing(console=True)
tracer = get_tracer()

if __name__ == '__main__':
    query = "existential meaning life"
    with tracer.start_as_current_span("cli.search"):
        # results = sparse_query(query) if False else semantic_query(query)
        results = semantic_query(query)
        # print('results are')
        # print(results)