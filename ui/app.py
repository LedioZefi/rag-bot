import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.rag import RAGAssistant


st.set_page_config(page_title="RAG Q&A Bot", layout="wide")

# --- Initialize ---
if "assistant" not in st.session_state:
    st.session_state.assistant = RAGAssistant(index_dir="index/chroma")
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📘 Intelligent Document Q&A RAG Bot")

# --- Input box ---
user_q = st.text_input("Ask a question about your document:", "")

if st.button("Ask") and user_q.strip():
    with st.spinner("Thinking..."):
        res = st.session_state.assistant.ask(user_q, st.session_state.history)
        st.session_state.history.append(("human", user_q))
        st.session_state.history.append(("assistant", res["answer"]))
    st.markdown("### 💬 Answer")
    st.write(res["answer"])
    if res["sources"]:
        st.markdown("### 📄 Sources")
        for s in res["sources"]:
            st.markdown(f"- {s}")

# --- Conversation history ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕘 Chat History")
    for role, msg in st.session_state.history[-6:]:
        if role == "human":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")