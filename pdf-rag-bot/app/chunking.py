from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
import re

def guess_section_title(text: str) -> str | None:
    # naive heuristic: first ALL-CAPS line or a numbered heading
    for line in text.splitlines():
        if re.match(r"^\s*(\d+\.)+\s+\S", line) or (len(line) < 80 and line.isupper()):
            return line.strip()
    return None

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        # derive a cheap "section" label once per chunk
        c.metadata["section"] = guess_section_title(c.page_content) or "Unknown"
    return chunks
