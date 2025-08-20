import os
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.extract_paragraphs import extract_paragraphs
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.metadata_schema import extract_metadata
from src.ingestion.normalizer import normalize_metadata
from src.ingestion.paragraph_utils import paragraphize
from src.storage.vector_store import store_vectors
from src.storage.sparse_store import store_sparse_vectors

def ingest_documents_with_pages(directory):
    """
    Ingest PDFs where each paragraph (with page + paragraph index) becomes a record.
    """
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.pdf'):
            continue
        file_path = os.path.join(directory, filename)
        print(f'Processing {file_path}...')

        pages, raw_metadata = load_pdf_pages(file_path)
        extracted = extract_metadata(raw_metadata)
        normalized_metadata = normalize_metadata(extracted)

        paragraph_records = []
        for page_number, page_text in pages:
            paragraphs = extract_paragraphs(page_text)
            print(f"Page {page_number} has {len(paragraphs)} paragraphs.")
            for p_idx, paragraph_text in enumerate(paragraphs):
                meta = {
                    **normalized_metadata,
                    "source_file": filename,
                    "page_number": page_number,
                    "paragraph_index": p_idx
                }
                paragraph_records.append({
                    "chunk": paragraph_text,
                    "metadata": meta
                })

        print(f"Total paragraphs to store: {len(paragraph_records)}")
        # print(paragraph_records)
        if paragraph_records:
            store_vectors(paragraph_records)
            store_sparse_vectors(paragraph_records)


if __name__ == '__main__':
    directory = './docs/test'  # Adjust path
    ingest_documents_with_pages(directory)