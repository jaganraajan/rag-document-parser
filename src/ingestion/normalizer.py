from datetime import datetime

def normalize_metadata(metadata):
    normalized_metadata = {}
    
    for key, value in metadata.items():
        # Normalize key to lowercase
        normalized_key = key.lower()
        
        # Normalize value (example: stripping whitespace)
        normalized_value = value.strip() if isinstance(value, str) else value
        
        normalized_metadata[normalized_key] = normalized_value
    
    return normalized_metadata

def normalize_date(date_str):
    try:
        # Example normalization: convert to ISO format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.isoformat()
    except ValueError:
        return date_str  # Return original if parsing fails

def normalize_document(document):
    normalized_document = {}
    data_fields = {'creation_date', 'modification_date'}
    
    # Normalize each field in the document
    for field, value in document.items():
        if field.lower() in data_fields:
            normalized_document[field] = normalize_date(value)
        else:
            normalized_document[field] = normalize_metadata(value)
    
    return normalized_document