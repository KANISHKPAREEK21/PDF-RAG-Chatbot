from typing import Dict, Any
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.prompts import ANSWER_PROMPT
from app.retriever import retrieve, format_context
from app.config import settings

# In-memory session store; plug Redis or DB for prod
_SESSION_STORE: dict[str, ChatMessageHistory] = {}

def _get_llm():
    prov = settings.PROVIDER.lower()
    if prov == "openai":
        return ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0.1,
            api_key=settings.OPENAI_API_KEY
        )
    if prov == "azure":
        return AzureChatOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.1,
        )
    if prov == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1
        )
    raise ValueError("Unsupported PROVIDER")

def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

# Retrieval step as a Runnable
def _retrieve_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    question = inputs["question"]
    where = inputs.get("where")
    docs = retrieve(question, where=where)
    return {"question": question, "docs": docs, "context": format_context(docs)}

def build_rag_chain():
    llm = _get_llm()
    chain = (
        RunnableMap({"question": lambda x: x["question"], "where": lambda x: x.get("where")})
        | _retrieve_fn
        | RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: x["context"]
        })
        | ANSWER_PROMPT
        | llm
    )
    # Attach message history for multi-turn sessions
    chain_with_mem = RunnableWithMessageHistory(
        chain,
        lambda session_id: _get_history(session_id),
        input_messages_key="question",   # what counts as the user's message
        history_messages_key="history",  # stored keys (auto)
    )
    return chain_with_mem
