from typing import Dict, Any
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from app.chains import build_rag_chain
from app.retriever import retrieve, format_context
from app.prompts import SUMMARY_PROMPT
from app.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

def _llm_small():
    prov = settings.PROVIDER.lower()
    if prov == "openai":
        return ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, temperature=0.1, api_key=settings.OPENAI_API_KEY)
    if prov == "azure":
        return AzureChatOpenAI(api_key=settings.AZURE_OPENAI_API_KEY,
                               azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                               azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                               temperature=0.1)
    if prov == "gemini":
        return ChatGoogleGenerativeAI(model=settings.GEMINI_CHAT_MODEL, google_api_key=settings.GOOGLE_API_KEY, temperature=0.1)
    raise ValueError("Unsupported PROVIDER")

def make_tools():
    rag = build_rag_chain()

    def answer_tool_run(q: str):
        return rag.invoke({"question": q}, config={"configurable": {"session_id": "agent"}}).content

    def summarise_tool_run(arg: str):
        # arg can be a free text summary request or query → retrieve first
        docs = retrieve(arg)
        ctx = format_context(docs)
        return _llm_small().invoke(SUMMARY_PROMPT.format(context=ctx)).content

    def doc_scope_tool_run(arg: str):
        # Input format: "document:MyFile.pdf | question: your question"
        try:
            doc, q = [s.strip() for s in arg.split("|")]
            doc_name = doc.split(":", 1)[1].strip()
            question = q.split(":", 1)[1].strip()
        except Exception:
            return "Usage: document:<filename.pdf> | question:<your question>"
        out = build_rag_chain().invoke(
            {"question": question, "where": {"source": doc_name}},
            config={"configurable": {"session_id": "agent"}}
        )
        return out.content

    tools = [
        Tool(name="AnswerFromDocs", func=answer_tool_run,
             description="Answer a question using the indexed PDFs with full RAG pipeline."),
        Tool(name="SummariseRelevantContext", func=summarise_tool_run,
             description="Retrieve relevant chunks and summarise them for a quick overview."),
        Tool(name="AnswerScopedToDocument", func=doc_scope_tool_run,
             description="Answer a question restricted to a specific document by filename."),
    ]
    return tools

def build_agent():
    tools = make_tools()
    llm = _llm_small()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful RAG agent. Use tools to answer accurately; if unsure, ask for a more precise query."),
        ("human", "{input}"),
        ("assistant", "Think step-by-step and decide which tool to use."),
        ("assistant", "Use the best tool or decline if the info is not in the PDFs."),
    ])
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
