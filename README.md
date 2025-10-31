# Intelligent RAG Document Q&A Bot 

This project is a simple but functional document question–answering bot built with **LangChain** and **Retrieval-Augmented Generation (RAG)**.  
It lets you upload your own text or PDF files, index them, and then ask natural language questions that get answered using only those documents.

---

## What it does

- Takes a set of PDFs or text files and breaks them into smaller chunks.
- Turns those chunks into vector embeddings using **SentenceTransformers** (`all-MiniLM-L6-v2`).
- Stores them in a **Chroma** vector database for fast semantic search.
- When you ask a question, it finds the most relevant chunks and feeds them into an **LLM** (via the **OpenRouter API**) to generate an answer that's based on your data, not the model's training.
- Answers include references to the original document pages.
- Comes with a small **Streamlit web app** for a clean, simple interface.

---

## Tech used

- **Python 3.12+**
- **LangChain** for orchestration  
- **Chroma** for vector storage  
- **SentenceTransformers** for embeddings  
- **OpenRouter** for LLM access  
- **Streamlit** for the front-end web app  
- Runs cleanly in **WSL/Linux**  

---

## How to run it

```bash
# Set up a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter credentials
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-..."

# Build the document index
python scripts/build_index.py

# Start the web app
streamlit run ui/app.py
```

---

## Notes

- The `index/` and `data/` folders are local - they aren't uploaded to GitHub.
- You can replace the model in `rag.py` with another one from OpenRouter if you prefer.
- The Streamlit app keeps short-term chat history for context-aware follow-ups.

---

## About

Built to demonstrate end-to-end RAG pipeline skills - from document indexing and retrieval to context-grounded LLM responses, all wrapped in a minimal, working web interface.
