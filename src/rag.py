"""
Local LLM & Private RAG
Ingestion + Retrieval usando ChromaDB + nomic-embed (LM Studio) + Ollama
"""
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR     = os.getenv("DOCS_DIR", "./docs")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Embeddings (nomic-embed via Ollama) ───────────────────────
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )

# ── Vector Store ───────────────────────────────────────────────
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )

# ── Ingest documents ──────────────────────────────────────────
def ingest_documents(docs_dir: str = DOCS_DIR) -> dict:
    """Load .md and .txt files, split and store in ChromaDB"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return {"status": "error", "message": f"Directory {docs_dir} not found"}

    # Load all .md and .txt files
    all_docs = []
    for pattern in ["**/*.md", "**/*.txt"]:
        for filepath in docs_path.glob(pattern):
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    if not all_docs:
        return {"status": "error", "message": "No documents found"}

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR
    )

    return {
        "status": "ok",
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
        "chroma_dir": CHROMA_DIR
    }

# ── RAG Chain ─────────────────────────────────────────────────
def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context provided.
If the answer is not in the context, say "I don't have that information in my documents."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Direct query ──────────────────────────────────────────────
def query_rag(question: str) -> dict:
    """Query the RAG system"""
    try:
        chain = get_rag_chain()
        answer = chain.invoke(question)

        # Get source documents
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        source_docs = retriever.invoke(question)
        sources = list(set([
            doc.metadata.get("source", "unknown") for doc in source_docs
        ]))

        return {
            "answer": answer,
            "sources": sources,
            "model": OLLAMA_MODEL,
            "embed_model": EMBED_MODEL
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "model": OLLAMA_MODEL}
