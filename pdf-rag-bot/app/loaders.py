from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List
from loguru import logger

def load_pdfs(paths: List[str]):
    """Load PDFs into LangChain Documents with basic metadata."""
    docs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"File not found: {p}")
            continue
        loader = PyPDFLoader(str(path))
        file_docs = loader.load()  # each page is a Document with metadata
        # Ensure minimal metadata
        for d in file_docs:
            d.metadata["source"] = path.name
            d.metadata["file_path"] = str(path.resolve())
            # page number is typically present as 'page'
            d.metadata["page"] = d.metadata.get("page", None)
        docs.extend(file_docs)
    return docs
