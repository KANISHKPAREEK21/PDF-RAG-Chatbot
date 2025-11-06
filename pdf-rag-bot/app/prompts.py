from langchain.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful, precise AI assistant. Answer strictly from the provided context. "
     "If the answer is not in context, say you don't know. Cite sources with "
     "filename and page numbers. Keep answers concise and well-structured."),
    ("human",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Return:\n- Direct answer\n- 2-5 bullet points\n- Sources list (file:page)")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class technical summarizer."),
    ("human", "Summarise the following context for a non-expert in 5-7 bullet points:\n{context}")
])
