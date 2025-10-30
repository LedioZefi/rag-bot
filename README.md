# Intelligent Document Q&A Bot (RAG)

**Tech:** Python, LangChain (1.x), Chroma (langchain-chroma), HuggingFaceEmbeddings (langchain-huggingface), Streamlit, OpenRouter API.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-..."  # your key
python scripts/build_index.py
streamlit run ui/app.py


