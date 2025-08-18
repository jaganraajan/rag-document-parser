import os
import uuid
# import json
from dotenv import load_dotenv  # dev dependency
load_dotenv()
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.metadata_schema import extract_metadata
from src.ingestion.normalizer import normalize_metadata
from src.ingestion.chunk_document import chunk_document
from src.storage.vector_store import store_vectors
from src.storage.corpus_store import get_corpus_store
# from src.logging_utils.audit_logger import log_event

def ingest_documents(directory):
    # Get corpus store instance
    corpus_store = get_corpus_store()
    
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
            
            # Add stable UUIDs to chunks and save to corpus
            chunks_with_ids = []
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_with_id = {
                    "id": chunk_id,
                    "chunk": chunk["chunk"],
                    "metadata": chunk["metadata"]
                }
                chunks_with_ids.append(chunk_with_id)
                
                # Save to corpus store for BM25
                corpus_store.save_chunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk["chunk"],
                    metadata=chunk["metadata"]
                )
            
            # Store vectors (will use the IDs we provided)
            vector_ids = store_vectors(chunks_with_ids)
            print(f'Stored {len(chunks_with_ids)} chunks to both vector store and corpus.')
            
            # # Log the ingestion event
            # log_event(f'Document ingested: {filename}', metadata)

if __name__ == '__main__':
    directory = '/Users/jaganraajan/projects/rag-document-parser/docs/test'  # Update this path
    ingest_documents(directory)