# ui/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import uuid
import streamlit as st

from app.config import settings
from app.loaders import load_pdfs
from app.chunking import chunk_documents
from app.vectorstore import index_documents, list_sources, reset_store
from app.chains import build_rag_chain
from app.retriever import retrieve

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")

# ---------------- Session state ---------------- #
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "corpus_id" not in st.session_state:
    st.session_state["corpus_id"] = None
if "available_sources" not in st.session_state:
    st.session_state["available_sources"] = []
if "selected_sources" not in st.session_state:
    st.session_state["selected_sources"] = []   # user-visible selection (before Apply)
if "active_sources" not in st.session_state:
    st.session_state["active_sources"] = []     # applied filter for retrieval
if "last_docs" not in st.session_state:
    st.session_state["last_docs"] = []

# ---------------- Sidebar: Upload + Index ---------------- #
with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)

    col_idx, col_clear = st.columns([1, 1])
    with col_idx:
        if st.button("Index", type="primary", use_container_width=True):
            # Save uploads
            paths = []
            for f in files or []:
                path = Path(settings.UPLOAD_DIR) / f.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(str(path))

            if not paths:
                st.warning("Please upload at least one PDF to index.")
            else:
                # Parse → chunk → tag with a NEW corpus_id → index
                docs = load_pdfs(paths)
                chunks = chunk_documents(docs)
                if not chunks:
                    st.warning("No text extracted from the uploaded PDFs. Please check the files.")
                else:
                    corpus_id = uuid.uuid4().hex
                    for c in chunks:
                        c.metadata = c.metadata or {}
                        c.metadata["corpus_id"] = corpus_id
                    index_documents(chunks)

                    # 🟢 Immediately set sources from filenames we just indexed
                    just_indexed_sources = [Path(p).name for p in paths]

                    # Refresh session scope for this run
                    st.session_state["corpus_id"] = corpus_id
                    st.session_state["available_sources"] = just_indexed_sources[:]  # prefer this
                    st.session_state["selected_sources"] = just_indexed_sources[:]
                    st.session_state["active_sources"] = just_indexed_sources[:]

                    st.success(f"Indexed {len(chunks)} chunks from {len(paths)} PDF(s).")
                    # Re-render so the multiselect becomes clickable with new options
                    st.rerun()

    with col_clear:
        if st.button("🧹 Clear ALL indexed data", use_container_width=True):
            if reset_store():
                st.session_state["corpus_id"] = None
                st.session_state["available_sources"] = []
                st.session_state["selected_sources"] = []
                st.session_state["active_sources"] = []
                st.session_state["messages"].clear()
                st.session_state["last_docs"] = []
                st.success("Vector index cleared from disk.")
            else:
                st.error("Failed to clear the index. See logs.")

    st.divider()
    st.header("Restrict to document(s)")

    # Determine dropdown options:
    # 1) Prefer the ones we already know from *this session's indexing*
    # 2) If empty (e.g., after app restart), try discovering from LanceDB by corpus_id
    if st.session_state["corpus_id"] is None:
        st.info("No documents indexed yet. Upload PDFs and click **Index**.")
        disabled = True
        current_options = []
    else:
        current_options = st.session_state["available_sources"][:]
        if not current_options:
            # Fallback discovery (may vary by Lance/Arrow versions)
            current_options = list_sources(corpus_id=st.session_state["corpus_id"])
            st.session_state["available_sources"] = current_options[:]
        # If options changed, reset selection to all by default
        if sorted(st.session_state["selected_sources"]) != sorted(current_options):
            st.session_state["selected_sources"] = current_options[:]
        disabled = len(current_options) == 0

    sel = st.multiselect(
        "Choose one or more documents (default: all)",
        options=current_options,
        default=st.session_state["selected_sources"],
        disabled=disabled,
        help="Select which PDFs the chatbot is allowed to use. By default, all are selected."
    )
    st.session_state["selected_sources"] = list(sel)

    if st.button("Apply filter"):
        if not st.session_state["selected_sources"]:
            st.warning("Select at least one document, then click Apply filter.")
        else:
            st.session_state["active_sources"] = list(st.session_state["selected_sources"])
            st.success(f"Filter applied to {len(st.session_state['active_sources'])} document(s).")

    st.divider()
    session_id = st.text_input(
        "Session ID",
        value="ui-session-1",
        help="Conversation memory key."
    )

# ---------------- Chat history (above input) ---------------- #
for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

# ---------------- Chat input (Enter submits) ---------------- #
query = st.chat_input("Ask about your PDFs (press Enter)")

if query:
    # If nothing indexed yet, do not search
    if st.session_state["corpus_id"] is None or not st.session_state["available_sources"]:
        st.warning("No indexed documents in this run. Upload PDFs and click **Index** first.")
    elif not st.session_state["active_sources"]:
        st.warning("No documents are selected. Use the sidebar to select documents and click **Apply filter**.")
    else:
        st.chat_message("user").write(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        # Build filename filter: {"source": ["doc1.pdf","doc2.pdf"]}
        where = {"source": st.session_state["active_sources"]}

        with st.spinner("Thinking…"):
            # Restrict by corpus + filenames
            docs = retrieve(
                query,
                where=where,
                k=int(getattr(settings, "TOP_K", 6)),
                corpus_id=st.session_state["corpus_id"],
            )
            st.session_state["last_docs"] = docs

            chain = build_rag_chain()
            payload = {"question": query}
            payload["where"] = where
            payload["corpus_id"] = st.session_state["corpus_id"]

            result = chain.invoke(payload, config={"configurable": {"session_id": session_id}})
            answer = result.content if hasattr(result, "content") else str(result)

        st.chat_message("assistant").write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

# --------------- Retrieved Chunks Preview (last turn) --------------- #
if st.session_state.get("last_docs"):
    with st.expander("🔎 Retrieved Chunks (last answer)"):
        for d in st.session_state["last_docs"]:
            md = d.metadata or {}
            src = md.get("source", "unknown")
            page = md.get("page", "n/a")
            sec = md.get("section", "Unknown")
            st.caption(f"**{src}** | p.{page} | {sec}")
            text = d.page_content or ""
            st.write(text[:800] + ("…" if len(text) > 800 else ""))
            st.divider()
