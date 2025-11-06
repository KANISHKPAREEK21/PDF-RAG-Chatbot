from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from app.loaders import load_pdfs
from app.chunking import chunk_documents
from app.vectorstore import index_documents
from app.chains import build_rag_chain
from app.agents import build_agent
from app.config import settings
from pathlib import Path
from loguru import logger

app = FastAPI(title="PDF RAG Chatbot", version="1.0")

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    paths = []
    for f in files:
        path = Path(settings.UPLOAD_DIR) / f.filename
        with open(path, "wb") as out:
            out.write(await f.read())
        paths.append(str(path))
    docs = load_pdfs(paths)
    chunks = chunk_documents(docs)
    if not chunks:
        return {"indexed": 0, "warning": "No text extracted from PDFs"}
    index_documents(chunks)
    return {"indexed": len(chunks)}

@app.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("default"), doc_name: Optional[str] = Form(None)):
    try:
        chain = build_rag_chain()
        payload = {"question": question}
        if doc_name:
            payload["where"] = {"source": doc_name}
        result = chain.invoke(payload, config={"configurable": {"session_id": session_id}})
        return {"answer": result.content}
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/agent")
async def agent(input: str = Form(...)):
    try:
        agent = build_agent()
        result = agent.invoke({"input": input})
        return {"answer": str(result)}
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})
