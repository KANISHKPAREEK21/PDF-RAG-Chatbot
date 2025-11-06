# 📄 PDF-RAG-Chatbot

**A Production-Ready Retrieval-Augmented Generation (RAG) Chatbot for Multi-PDF Querying**

---

## 🚀 Overview

This project implements a **fully functional, production-worthy RAG Chatbot** capable of answering questions from one or more PDF documents.
It was developed as part of a job interview assignment — but evolved into a complete, scalable AI solution that supports multiple providers (**Gemini**, **OpenAI**, **Azure OpenAI**) and modern vector storage with **LanceDB**.

The chatbot can:

* Parse and chunk PDFs
* Generate embeddings
* Store and retrieve vector data efficiently
* Maintain conversational memory
* Filter by document metadata
* Orchestrate responses agentically using LangChain chains

All wrapped in a slick **Streamlit UI** and deployed for free on **Streamlit Cloud**.

---

## 🧠 Tech Stack

| Category            | Technology                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **Frontend / UI**   | Streamlit                                                                                     |
| **Backend / Logic** | Python (LangChain, FastAPI optional)                                                          |
| **Vector Store**    | LanceDB                                                                                       |
| **LLMs**            | Gemini / OpenAI / Azure OpenAI                                                                |
| **Embeddings**      | `text-embedding-004` (Gemini) / `text-embedding-3-large` (OpenAI)                             |
| **Libraries**       | langchain, langchain-community, langchain-google-genai, lancedb, pydantic-settings, streamlit |
| **Deployment**      | Streamlit Cloud (Free Tier)                                                                   |

---

## 🧩 Features by Level

### **Level 1 – Semantic Search RAG**

* PDF parsing, chunking, embedding generation, and top-k retrieval.
* Query answering through a pre-trained LLM.

### **Level 2 – Modular Production Design**

* Pipeline refactored using LangChain and LanceDB.
* Clean architecture for easy model swapping.
* Multi-PDF upload and error-resilient handling.

### **Level 3 – Conversational Memory**

* Persistent chat history per session.
* Smooth context carryover for multi-turn conversation.

### **Level 4 – Metadata Tagging and Filtering**

* Metadata for document name, page, and section.
* Dynamic filter UI: choose one or more PDFs to search.

### **Level 5 – Agent-Based Behavior**

* Adaptive response orchestration via LangChain.
* Intelligent routing for answering, summarizing, or switching docs.
* Graceful fallback and logging for robustness.

---

## 🌐 Live Demo

**Deployed App:**
👉 [https://pdf-rag-chatbot-9r4nvcdfph9uk2qnyrwnjr.streamlit.app/](https://pdf-rag-chatbot-9r4nvcdfph9uk2qnyrwnjr.streamlit.app/)

**GitHub Repository:**
👉 [https://github.com/KANISHKPAREEK21/PDF-RAG-Chatbot](https://github.com/KANISHKPAREEK21/PDF-RAG-Chatbot)

**Project walkthrough screen recording:**
👉 [walkthrough-PDF-RAG-Chatbot](https://drive.google.com/file/d/1bIFmdc2HItkHwzdMUEaHS9mSkrgZTFO5/view?usp=sharing)

---

## ⚙️ Setup (Local or Cloud)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/KANISHKPAREEK21/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Credentials

#### 🧪 Option A — Local `.env`

Create a `.env` file in the root folder:

```env
PROVIDER=gemini
GOOGLE_API_KEY=YOUR_REAL_GOOGLE_API_KEY
GEMINI_CHAT_MODEL=gemini-2.5-pro
GEMINI_EMBED_MODEL=text-embedding-004
```

#### ☁️ Option B — Streamlit Secrets

On Streamlit Cloud, go to
**Settings → Secrets** → paste:

```toml
PROVIDER = "gemini"
GOOGLE_API_KEY = "YOUR_REAL_GOOGLE_API_KEY"
GEMINI_CHAT_MODEL = "gemini-2.5-pro"
GEMINI_EMBED_MODEL = "text-embedding-004"
```

*(or OpenAI equivalent keys if preferred)*

---

## ▶️ Run the App

**Streamlit UI:**

```bash
streamlit run ui/streamlit_app.py
```

**Optional FastAPI backend:**

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧭 Usage

1. **Upload PDFs** from the sidebar.
2. Click **Index** (mandatory for new uploads).
3. Ask questions in the chat box.
4. Use the **filter dropdown** to restrict which documents are searched.
5. Review retrieved chunks in the expander below the chat.

> Each new indexing session generates a unique `corpus_id`, so the chatbot isolates results per session while allowing selective multi-document queries.

---

## 💾 Data Paths

| Directory         | Purpose                       |
| ----------------- | ----------------------------- |
| `./.data/lancedb` | Vector database storage       |
| `./data/uploads`  | Uploaded PDFs                 |
| `./data/store`    | Persistent metadata / configs |

All folders auto-create at runtime.

---

## 🔐 Security

* API keys are never hardcoded.
* Configurable via `.env` or Streamlit Secrets.
* No third-party uploads — data stays local to the Streamlit session.

---

## 🧰 Troubleshooting

* ❌ *“Keeps thinking”* → Check your API key validity.
* ⚠️ *“No indexed documents”* → Click **Index** after upload.
* 🧠 *“Wrong document answering”* → Use the document filter to isolate specific PDFs.
* 💀 *Crash during embed* → Reduce file size or chunk size in `config.py`.

---

## 💬 Author

**Kanishk Pareek**
📧 [kanishkpareek26@gmail.com](mailto:kanishkpareek26@gmail.com)
LinkedIN : [kanishk-pareek/](https://www.linkedin.com/in/kanishk-pareek/)
Open for collaboration, discussion, and improvements.

> “This RAG Chatbot was built not just as an assignment —
> but as a live demonstration of how AI can transform document understanding into seamless, conversational intelligence.”

---

## ⭐ If you like it

Give it a star ⭐ on GitHub, test it live, and share your feedback!

