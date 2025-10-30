"""
Build a persistent Chroma index from PDF documents in the data/ directory.

This script:
1. Loads all PDFs from data/ using PyPDFLoader
2. Splits documents with RecursiveCharacterTextSplitter
3. Creates embeddings using HuggingFaceEmbeddings
4. Builds a persistent Chroma index at index/chroma
5. Prints document and chunk counts
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def load_pdfs(data_dir: str = "data") -> List[Document]:
    """
    Load all PDF files from the specified directory.
    
    Args:
        data_dir: Directory containing PDF files
        
    Returns:
        List of loaded documents
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory '{data_dir}' does not exist")
        return documents
    
    pdf_files = list(data_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) in {data_dir}/")
    
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
        print(f"  → Loaded {len(docs)} pages")
    
    return documents


def split_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


def build_index(
    documents: List[Document],
    index_dir: str = "index/chroma",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Chroma:
    """
    Build a persistent Chroma index from documents.
    
    Args:
        documents: List of document chunks to index
        index_dir: Directory to store the Chroma index
        model_name: HuggingFace model name for embeddings
        
    Returns:
        Chroma vector store instance
    """
    print(f"\nInitializing embeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print(f"Building Chroma index at {index_dir}/")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=index_dir
    )
    
    return vectorstore


def main():
    """
    End-to-end pipeline: load PDFs, split, and build index.
    """
    print("=" * 60)
    print("RAG Bot - Index Builder")
    print("=" * 60)
    
    # Step 1: Load PDFs
    print("\n[Step 1] Loading PDFs...")
    documents = load_pdfs("data")
    print(f"Total documents loaded: {len(documents)}")
    
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Step 2: Split documents
    print("\n[Step 2] Splitting documents into chunks...")
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=120)
    print(f"Total chunks created: {len(chunks)}")
    
    # Step 3: Build index
    print("\n[Step 3] Building Chroma index...")
    vectorstore = build_index(
        chunks,
        index_dir="index/chroma",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Index Build Complete!")
    print("=" * 60)
    print(f"Documents loaded:  {len(documents)}")
    print(f"Chunks created:    {len(chunks)}")
    print(f"Index location:    index/chroma")
    print("=" * 60)


if __name__ == "__main__":
    main()

