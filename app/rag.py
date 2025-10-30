from __future__ import annotations
from typing import Optional, List, Dict, Any
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _fmt_sources(docs) -> List[str]:
    seen, out = set(), []
    for d in docs:
        md = d.metadata or {}
        page = md.get("page", md.get("page_number", "NA"))
        src = md.get("source") or md.get("file_path") or md.get("path") or "unknown"
        key = f"{os.path.basename(src)}:p.{page}"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

def _fmt_context(docs) -> str:
    parts = []
    for d in docs:
        md = d.metadata or {}
        page = md.get("page", md.get("page_number", "NA"))
        src = md.get("source") or md.get("file_path") or md.get("path") or "unknown"
        parts.append(f"[source: {os.path.basename(src)} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

# Prompts (history-aware rewrite + grounded answer)
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's follow-up into a standalone search query using the chat history only as context."),
    ("human", "Chat history:\n{chat_history}\n\nUser question:\n{input}\n\nStandalone query:")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly from the provided context. If it is not there, say \"I don't know.\" "
               "Always cite as [source: filename p.page]."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

class RAGAssistant:
    """RAG Assistant that answers questions based on retrieved documents (LangChain 1.x)."""

    def __init__(self, index_dir: str = "index/chroma", k: int = 5):
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore = Chroma(persist_directory=index_dir, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # LLM via OpenRouter (you already exported OPENAI_BASE_URL/OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="minimax/minimax-m2:free", temperature=0)

        # Runnable pipelines (1.x style)
        self.rewriter = (REWRITE_PROMPT | self.llm | StrOutputParser())
        self.answerer = (ANSWER_PROMPT | self.llm | StrOutputParser())

    def _rewrite(self, question: str, chat_history: Optional[List[tuple]]) -> str:
        history_txt = ""
        if chat_history:
            history_txt = "\n".join(f"{role}: {content}" for role, content in chat_history)
        return self.rewriter.invoke({"chat_history": history_txt, "input": question}).strip()

    def ask(self, question: str, chat_history: Optional[List[tuple]] = None) -> Dict[str, Any]:
        standalone = self._rewrite(question, chat_history)

        docs = self.retriever.invoke(standalone)
        if not docs:
            return {"answer": "I don't know.", "sources": []}

        context = _fmt_context(docs)
        answer = self.answerer.invoke({"question": question, "context": context}).strip()
        sources = _fmt_sources(docs)

        return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    a = RAGAssistant(index_dir="index/chroma")
    res = a.ask("Give me a brief overview with citations.", chat_history=[])
    print("\nANSWER:\n", res["answer"])
    print("\nSOURCES:", res["sources"])