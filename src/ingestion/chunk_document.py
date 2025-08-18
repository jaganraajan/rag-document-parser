def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Splits the input text into chunks of specified size with overlap.
    
    Parameters:
    - text (str): The text to be chunked.
    - chunk_size (int): The maximum size of each chunk.
    - overlap (int): The number of overlapping tokens between chunks.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_document(document_text, metadata, chunk_size=500, overlap=80):
    """
    Processes a document and chunks its text content.
    
    Parameters:
    - document_text
    - metadata: dictionary containing document metadata and text.
    - chunk_size (int): The maximum size of each chunk.
    - overlap (int): The number of overlapping tokens between chunks.
    
    Returns:
    - List[dict]: A list of dictionaries containing chunked text and metadata.
    """
    chunks = chunk_text(document_text, chunk_size, overlap)
    return [{'chunk': chunk, 'metadata': metadata} for chunk in chunks]