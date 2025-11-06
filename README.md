# PDF RAG Chatbot (Production-Ready, 5 Levels)

A modular, production-minded RAG chatbot that ingests one or more PDFs and answers questions with citations. 
Supports conversational memory, metadata filtering, and an agent that routes between tools.

## Features
- Multi-PDF ingestion, chunking, embeddings, vector DB (Chroma)
- Provider-switchable: OpenAI / Azure OpenAI / Gemini (via .env)
- Conversational memory (RunnableWithMessageHistory)
- Metadata tagging: filename, page, section (heuristic)
- Metadata filtering (e.g., restrict to a file)
- Agent tools: Answer, Summarise, Scoped answer
- Streamlit UI + FastAPI endpoints for Postman

## Setup (Windows)
1. `py -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `copy .env.example .env` and set your keys.
4. Run UI: `streamlit run ui/streamlit_app.py`
5. Run API: `uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload`

## Usage
- Upload PDFs in the Streamlit sidebar, click **Index**.
- Ask questions; optionally set **Session ID** for persistent memory.
- Use **Filters** to restrict to a document by filename.
- View retrieved chunks (transparency & trust).

## Decisions
- Chroma for metadata filters & persistence.
- LangChain LCEL for clean composition & memory.
- pypdf for Windows-friendly parsing.
- Prompts enforce grounded, cited answers.

## Extensibility
- Plug in OCR (pytesseract) for scanned docs.
- Add reranking for accuracy.
- Persist memory in Redis or SQLite.
