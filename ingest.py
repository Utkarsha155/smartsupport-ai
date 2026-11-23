"""
Builds the FAISS vector index from the company knowledge base.

Steps:
  1. Load every .md file from data/knowledge_base
  2. Split each file into small overlapping chunks
  3. Convert each chunk into an embedding (vector) using the LLM provider
  4. Store all vectors in a FAISS index saved to ./faiss_index

Run it once, and again whenever the knowledge base changes:
    python ingest.py
"""
from app import config
from app.llm import get_embeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def main():
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .md files found in {config.KNOWLEDGE_BASE_DIR}")
    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n## ", "\n\n", "\n", ". ", " "],  # prefer splitting at section headings
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Creating embeddings and building FAISS index...")
    # normalize_L2=True makes every vector length 1, so the L2 distance
    # can be turned into cosine similarity at search time (see app/rag.py)
    vector_store = FAISS.from_documents(chunks, get_embeddings(), normalize_L2=True)
    vector_store.save_local(str(config.FAISS_INDEX_DIR))
    print(f"Saved FAISS index to {config.FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
