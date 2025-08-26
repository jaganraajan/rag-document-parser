def extract_metadata(metadata):
    """
    Extracts and processes metadata from a given dictionary.

    Args:
        metadata (dict): A dictionary containing metadata fields.

    Returns:
        dict: A dictionary with cleaned and structured metadata.
    """
    def safe_str(val):
        # Convert IndirectObject or None to string
        try:
            return str(val).strip() if val is not None else ""
        except Exception:
            return ""
        
    extracted_metadata = {
        "title": safe_str(metadata.get('/Title', '')),
        "author": safe_str(metadata.get('/Author', '')),
        "producer": safe_str(metadata.get('/Producer', '')),
        "creator": safe_str(metadata.get('/Creator', '')),
        "creation_date": safe_str(metadata.get('/CreationDate', '')),
        "modification_date": safe_str(metadata.get('/ModDate', '')),
        "keywords": safe_str(metadata.get('/Keywords', '')),
        "apple_keywords": metadata.get('/AAPL:Keywords', []),
        "rgid": safe_str(metadata.get('/rgid', ''))
    }

    # Additional processing for dates if needed
    # Example: Convert PDF date format to a standard datetime object
    # extracted_metadata['creation_date'] = parse_pdf_date(extracted_metadata['creation_date'])
    # extracted_metadata['modification_date'] = parse_pdf_date(extracted_metadata['modification_date'])

    return extracted_metadata
