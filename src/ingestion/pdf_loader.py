from PyPDF2 import PdfReader

def load_pdf(file_path):
    """Load a PDF file and extract text and metadata."""
    reader = PdfReader(file_path)
    text = ""
    metadata = reader.metadata

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text, metadata

def parse_pdf(file_path):
    """Parse a PDF file to extract text and metadata."""
    text, metadata = load_pdf(file_path)
    # Further processing can be done here if needed
    return text, metadata