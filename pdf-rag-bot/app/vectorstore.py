from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, Any, Optional, List, Iterable

import lancedb
from langchain_community.vectorstores import LanceDB as LC_LanceDB
from app.embeddings import get_embeddings
from app.config import settings


def _db_path() -> str:
    # Use LANCE_DIR if present, otherwise reuse PERSIST_DIR
    base = getattr(settings, "LANCE_DIR", None) or settings.PERSIST_DIR
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base).resolve())


def _table_name() -> str:
    return getattr(settings, "LANCE_TABLE", "pdf_rag")


def _conn():
    return lancedb.connect(_db_path())


def _table_exists(conn, table_name: str) -> bool:
    try:
        return table_name in conn.table_names()
    except Exception:
        return False


def get_store() -> LC_LanceDB:
    """Open the LanceDB vector store (assumes table already exists)."""
    conn = _conn()
    table = _table_name()
    return LC_LanceDB(
        connection=conn,
        table_name=table,
        embedding=get_embeddings(),
    )


def index_documents(docs: Sequence):
    """
    If table doesn't exist, create it with from_documents (infers schema).
    Otherwise, append with add_documents.
    """
    if not docs:
        return  # nothing to index; avoid accidental empty table creation

    # Ensure required metadata keys exist so list_sources() / filters work
    for d in docs:
        d.metadata = d.metadata or {}
        # Normalize filename
        src = d.metadata.get("source") or d.metadata.get("path")
        if src:
            d.metadata["source"] = Path(src).name
        else:
            d.metadata["source"] = "unknown"
        d.metadata.setdefault("page", d.metadata.get("page", 0))
        d.metadata.setdefault("section", d.metadata.get("section", ""))

    conn = _conn()
    table = _table_name()
    emb = get_embeddings()

    if not _table_exists(conn, table):
        # First time: create table and insert docs in one go
        LC_LanceDB.from_documents(
            docs,
            emb,
            connection=conn,
            table_name=table,
        )
    else:
        vs = LC_LanceDB(connection=conn, table_name=table, embedding=emb)
        vs.add_documents(docs)


def similarity_search(query: str, k: int, where: Optional[Dict[str, Any]] = None):
    """
    Run similarity search with optional metadata filter.
    """
    vs = get_store()
    return vs.similarity_search(query, k=k, filter=where)


def list_sources(corpus_id: Optional[str] = None) -> list[str]:
    """
    Returns a sorted list of distinct metadata['source'] values in the LanceDB table.
    If corpus_id is provided, only returns sources belonging to that corpus.
    Safe if the table doesn't exist (returns []).
    Handles both flattened ('metadata.source') and struct ('metadata': {...}) shapes.
    """
    try:
        conn = _conn()
        table_name = _table_name()
        if not _table_exists(conn, table_name):
            return []
        tbl = conn.open_table(table_name)

        sources = set()

        # Prefer flattened columns (newer Lance/Arrow)
        try:
            for batch in tbl.to_batches(columns=["metadata.source", "metadata.corpus_id"]):
                rows = batch.to_pylist()
                for row in rows:
                    src = row.get("metadata.source")
                    corp = row.get("metadata.corpus_id")
                    if corpus_id and corp != corpus_id:
                        continue
                    if src:
                        sources.add(src)
        except Exception:
            # Fallback to older struct shape
            for batch in tbl.to_batches(columns=["metadata"]):
                rows = batch.to_pylist()
                for row in rows:
                    if isinstance(row, dict) and "metadata" in row and isinstance(row["metadata"], dict):
                        md = row["metadata"]
                    elif isinstance(row, dict):
                        md = row
                    else:
                        md = None
                    if not isinstance(md, dict):
                        continue
                    if corpus_id and md.get("corpus_id") != corpus_id:
                        continue
                    src = md.get("source")
                    if src:
                        sources.add(src)

        return sorted(sources)
    except Exception:
        return []


def reset_store() -> bool:
    """
    Drops the current LanceDB table. Returns True if dropped or didn't exist.
    """
    try:
        conn = _conn()
        table_name = _table_name()
        if _table_exists(conn, table_name):
            conn.drop_table(table_name)
        return True
    except Exception:
        return False
