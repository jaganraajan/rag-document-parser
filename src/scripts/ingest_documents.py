import os
# import json
from dotenv import load_dotenv  # dev dependency
load_dotenv()
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.metadata_schema import extract_metadata
from src.ingestion.normalizer import normalize_metadata
from src.ingestion.chunk_document import chunk_document
from src.storage.vector_store import store_vectors
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
            # print(chunks[:2])
            
            # Store vectors
            # vector_ids = store_vectors(chunks, normalized_metadata)
            vector_ids = store_vectors(chunks)
            # print(f'Stored {len(vector_ids)} vectors for {filename}.')
            
            # # Log the ingestion event
            # log_event(f'Document ingested: {filename}', metadata)

if __name__ == '__main__':
    directory = '/Users/jaganraajan/projects/rag-document-parser/docs/test'  # Update this path
    ingest_documents(directory)