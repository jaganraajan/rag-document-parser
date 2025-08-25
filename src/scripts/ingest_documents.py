import os, time
# import json
from dotenv import load_dotenv

load_dotenv()
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.metadata_schema import extract_metadata
from src.ingestion.normalizer import normalize_metadata
from src.ingestion.chunk_document import chunk_document
from src.storage.vector_store import store_vectors
from src.storage.sparse_store import store_sparse_vectors 
# from src.logging_utils.audit_logger import log_event

from src.observability.instruments import (
    documents_ingested_total,
    ingestion_errors_total,
    document_ingest_seconds,
    # new instruments
    chunk_count_total,
    chunk_chars_sum,
    pdf_pages_total,
    text_extraction_failures_total,
    pdf_load_seconds,
    metadata_extract_seconds,
    chunking_seconds,
    # helper
    safe_attrs,
)

def ingest_documents(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            print(f'Processing {file_path}...')
            
            start = time.perf_counter()
            status = "success"
            try:
                # Phase: load PDF + extract page texts
                t0 = time.perf_counter()
                pages, pdf_metadata = load_pdf_pages(file_path)
                pdf_load_seconds.record(time.perf_counter() - t0, safe_attrs({}))

                # Count pages and page-level extraction failures
                page_texts = []
                failures = 0
                for _, page_text in pages:
                    if page_text:
                        page_texts.append(page_text)
                    else:
                        page_texts.append("")
                        failures += 1
                pdf_pages_total.add(len(pages), safe_attrs({}))
                if failures:
                    text_extraction_failures_total.add(failures, safe_attrs({"reason": "empty_text"}))
                
                pdf_content = "\n".join(page_texts)

                # Phase: extract + normalize metadata
                t1 = time.perf_counter()
                metadata = extract_metadata(pdf_metadata)
                normalized_metadata = normalize_metadata(metadata)
                metadata_extract_seconds.record(time.perf_counter() - t1, safe_attrs({}))
                
                # Phase: chunking
                t2 = time.perf_counter()
                chunks = chunk_document(pdf_content, normalized_metadata)
                chunking_seconds.record(time.perf_counter() - t2, safe_attrs({}))
                chunk_count_total.add(len(chunks), safe_attrs({}))
                total_chars = sum(len(c.get('chunk', '')) for c in chunks)
                if total_chars:
                    chunk_chars_sum.add(total_chars, safe_attrs({}))

                print(f'Chunks created: {len(chunks)}')
                
                # Store vectors
                store_vectors(chunks, dense_model=os.getenv("DENSE_MODEL"))
                store_sparse_vectors(chunks)
                
                # # Log the ingestion event
                # log_event(f'Document ingested: {filename}', metadata)
            except Exception as e:
                status = "error"
                ingestion_errors_total.add(1, safe_attrs({"error.type": e.__class__.__name__}))
                raise
            finally:
                duration = time.perf_counter() - start
                document_ingest_seconds.record(duration, safe_attrs({"status": status}))
                if status == "success":
                    documents_ingested_total.add(1, safe_attrs({"status": "success"}))

if __name__ == '__main__':
    directory = '/Users/jaganraajan/projects/rag-document-parser/docs/test'  # Update this path
    ingest_documents(directory)