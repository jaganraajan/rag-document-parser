import os
# import json
from src.ingestion.pdf_loader import load_pdf
# from src.ingestion.metadata_schema import extract_metadata
# from src.ingestion.normalizer import normalize_metadata
# from src.ingestion.chunker import chunk_text
# from src.storage.vector_store import store_vectors
# from src.logging_utils.audit_logger import log_event

def ingest_documents(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            print(f'Processing {file_path}...')
            
            # Load PDF
            pdf_content = load_pdf(file_path)
            print(pdf_content)
            
            # Extract and normalize metadata
            # metadata = extract_metadata(pdf_content)
            # normalized_metadata = normalize_metadata(metadata)
            
            # # Chunk text
            # chunks = chunk_text(pdf_content)
            
            # # Store vectors
            # vector_ids = store_vectors(chunks, normalized_metadata)
            
            # # Log the ingestion event
            # log_event(f'Document ingested: {filename}', metadata)

if __name__ == '__main__':
    directory = '/Users/jaganraajan/projects/rag-document-parser/docs'  # Update this path
    ingest_documents(directory)