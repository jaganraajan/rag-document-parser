import re
from typing import List

PARA_MIN_CHARS = 80          # ignore ultra-short noise paragraphs
FALLBACK_TARGET_CHARS = 600  # grouping size when no blank lines exist

_sentence_end_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _clean_page_text(txt: str) -> str:
    # Normalize newlines
    txt = txt.replace('\r', '\n')
    # Fix hyphenation across line breaks: word-\ncontinuation -> wordcontinuation
    txt = re.sub(r'(\w)-\n(\w)', r'\1\2', txt)
    # Remove isolated line breaks inside paragraphs: lines ending mid‑sentence
    # First mark double newlines to preserve paragraph boundaries
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    # Replace single newlines that are not followed by another newline with a space
    txt = re.sub(r'(?<!\n)\n(?!\n)', ' ', txt)
    # Collapse spaces
    txt = re.sub(r'[ \t]+', ' ', txt)
    return txt.strip()

def _split_on_blank_lines(txt: str) -> List[str]:
    parts = [p.strip() for p in re.split(r'\n\s*\n', txt) if p.strip()]
    return parts

def _fallback_sentence_grouping(txt: str) -> List[str]:
    sents = _sentence_end_re.split(txt)
    grouped = []
    buf = []
    char_count = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        buf.append(s)
        char_count += len(s) + 1
        if char_count >= FALLBACK_TARGET_CHARS:
            paragraph = ' '.join(buf).strip()
            if len(paragraph) >= PARA_MIN_CHARS:
                grouped.append(paragraph)
            buf = []
            char_count = 0
    if buf:
        paragraph = ' '.join(buf).strip()
        if len(paragraph) >= PARA_MIN_CHARS:
            grouped.append(paragraph)
    return grouped

def extract_paragraphs(page_text: str) -> List[str]:
    if not page_text or not page_text.strip():
        return []
    cleaned = _clean_page_text(page_text)
    # First attempt: split on blank lines (if present)
    if '\n\n' in page_text:
        paras = _split_on_blank_lines(cleaned)
        # print(paras)
    else:
        # No blank lines detected originally, fallback to sentence grouping
        paras = _fallback_sentence_grouping(cleaned)
        # print(paras)
    # Filter too-short fragments (e.g. headers, page numbers)
    paras = [p for p in paras if len(p) >= PARA_MIN_CHARS]
    return paras