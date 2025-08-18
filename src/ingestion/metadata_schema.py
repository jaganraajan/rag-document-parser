def extract_metadata(metadata):
    """
    Extracts and processes metadata from a given dictionary.

    Args:
        metadata (dict): A dictionary containing metadata fields.

    Returns:
        dict: A dictionary with cleaned and structured metadata.
    """
    extracted_metadata = {
        "title": metadata.get('/Title', '').strip(),
        "author": metadata.get('/Author', '').strip(),
        "producer": metadata.get('/Producer', '').strip(),
        "creator": metadata.get('/Creator', '').strip(),
        "creation_date": metadata.get('/CreationDate', '').strip(),
        "modification_date": metadata.get('/ModDate', '').strip(),
        "keywords": metadata.get('/Keywords', '').strip(),
        "apple_keywords": metadata.get('/AAPL:Keywords', []),
        "rgid": metadata.get('/rgid', '').strip()
    }

    # Additional processing for dates if needed
    # Example: Convert PDF date format to a standard datetime object
    # extracted_metadata['creation_date'] = parse_pdf_date(extracted_metadata['creation_date'])
    # extracted_metadata['modification_date'] = parse_pdf_date(extracted_metadata['modification_date'])

    return extracted_metadata
