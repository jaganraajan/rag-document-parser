import os
# import json
from dotenv import load_dotenv  # dev dependency
load_dotenv()
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.metadata_schema import extract_metadata
from src.ingestion.normalizer import normalize_metadata
from src.ingestion.chunk_document import chunk_document
from src.storage.vector_store import store_vectors
from src.storage.sparse_store import sparse_store
# from src.logging_utils.audit_logger import log_event

def ingest_documents(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            print(f'Processing {file_path}...')
            
            # Load PDF
            pdf_content, pdf_metadata = load_pdf(file_path)
            # print(pdf_content)
            print(pdf_metadata)
            
            # Extract and normalize metadata
            metadata = extract_metadata(pdf_metadata)
            print(metadata)
            normalized_metadata = normalize_metadata(metadata)
            print(normalized_metadata)
            
            # Chunk text
            chunks = chunk_document(pdf_content, normalized_metadata)
            print(f'Chunks created: {len(chunks)}')
            
            # Assign consistent IDs to chunks for both dense and sparse storage
            import uuid
            for chunk in chunks:
                chunk['id'] = str(uuid.uuid4())
            
            # Store dense vectors
            vector_ids = store_vectors(chunks)
            print(f'Stored {len(chunks)} dense vectors for {filename}.')
            
            # Store sparse vectors
            try:
                sparse_store.upsert_sparse_vectors(chunks)
                print(f'Stored {len(chunks)} sparse vectors for {filename}.')
            except Exception as e:
                print(f'Warning: Failed to store sparse vectors for {filename}: {e}')
            
            # # Log the ingestion event
            # log_event(f'Document ingested: {filename}', metadata)

if __name__ == '__main__':
    directory = '/Users/jaganraajan/projects/rag-document-parser/docs/test'  # Update this path
    ingest_documents(directory)