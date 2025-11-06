# app/retriever.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore import similarity_search

RawWhere = Union[str, Dict[str, Any], None]


# ---------------------------
# Small helpers
# ---------------------------

def _sql_quote(val: Any) -> str:
    """Return a single-quoted literal for DataFusion/Lance filters."""
    s = str(val)
    return "'" + s.replace("'", "''") + "'"


def _or_equals(expr: str, values: Sequence[Any]) -> str:
    """expr = v1 OR expr = v2 ... (avoid IN() for widest compatibility)."""
    parts = [f"{expr} = {_sql_quote(v)}" for v in values]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _and_all(clauses: List[str]) -> Optional[str]:
    """Join non-empty clauses with AND, or return None if empty."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return "(" + ") AND (".join(clauses) + ")"


# ---------------------------
# Filter building
# ---------------------------

def _normalize_where(where: RawWhere) -> Optional[str]:
    """
    Accept:
      - None
      - raw string (returned as-is)
      - dict with {"source": "file.pdf"} or {"source": ["a.pdf","b.pdf"]}
    Return a Lance/DataFusion filter string or None.
    """
    if not where:
        return None

    if isinstance(where, str):
        s = where.strip()
        return s or None

    if isinstance(where, dict):
        clauses: List[str] = []

        if "source" in where and "metadata" not in where:
            src_val = where["source"]
            expr = "metadata['source']"
            if isinstance(src_val, (list, tuple, set)):
                src_list = list(src_val)
                if src_list:
                    clauses.append(_or_equals(expr, src_list))
            else:
                clauses.append(f"{expr} = {_sql_quote(src_val)}")

        # Extend here if you add more keys later (e.g., page ranges, section, etc.)

        return _and_all(clauses)

    # Unknown shape
    return None


def _with_corpus(filter_str: Optional[str], corpus_id: Optional[str]) -> Optional[str]:
    """
    Add corpus filter if provided. If the table/schema doesn't have this field,
    Lance/DataFusion would normally error; we'll catch it at query time.
    """
    if not corpus_id:
        return filter_str
    clause = f"metadata['corpus_id'] = {_sql_quote(corpus_id)}"
    return _and_all([c for c in [filter_str, clause] if c])


# ---------------------------
# Public API
# ---------------------------

def retrieve(
    query: str,
    where: RawWhere = None,
    k: Optional[int] = None,
    corpus_id: Optional[str] = None,
) -> List[Document]:
    """
    Similarity search with optional filename and corpus scoping.

    - where: None | raw string | {"source": "file.pdf"} | {"source": ["a.pdf","b.pdf"]}
    - corpus_id: if provided, restricts hits to the current indexing session
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = int(k or getattr(settings, "TOP_K", 6))

    base = _normalize_where(where)
    filt = _with_corpus(base, corpus_id)

    # Run the query; if the table schema lacks metadata['corpus_id'], avoid crashing.
    try:
        return similarity_search(query, k=top_k, where=filt)
    except Exception as e:
        msg = str(e)
        # If schema doesn't have corpus_id, return no docs (honours "limit to current corpus")
        if "corpus_id" in msg and ("not found" in msg or "No field" in msg or "Field" in msg):
            return []
        # Otherwise, bubble up the error for visibility
        raise

def format_context(docs: List[Document]) -> str:
    """Readable context block for the LLM, including source markers."""
    lines: List[str] = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page", "n/a")
        sec = md.get("section", "Unknown")
        content = d.page_content or ""
        lines.append(f"[{src} | p.{page} | {sec}]\n{content}")
    return "\n\n---\n\n".join(lines)
