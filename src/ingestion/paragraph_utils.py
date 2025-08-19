import re

def normalize_whitespace(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).strip()

def paragraphize(page_text: str, min_len=20):
    """
    Split page text into paragraphs. Strategy:
    1. Split on blank lines OR large line breaks.
    2. Merge very short fragments into the previous paragraph.
    """
    # Replace Windows newlines
    cleaned = page_text.replace('\r', '')
    raw_paras = re.split(r'\n\s*\n+', cleaned)  # blank-line separated
    paragraphs = []
    buffer = []
    for para in raw_paras:
        lines = [normalize_whitespace(l) for l in para.split('\n')]
        candidate = normalize_whitespace(" ".join([l for l in lines if l]))
        if not candidate:
            continue
        if len(candidate) < min_len and paragraphs:
            # append to previous if too short
            paragraphs[-1] = paragraphs[-1] + " " + candidate
        else:
            paragraphs.append(candidate)
    return paragraphs