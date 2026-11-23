"""Retrieval: finds the knowledge base chunks most relevant to a question."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app import config
from app.llm import get_embeddings

_vector_store: FAISS | None = None


def load_vector_store() -> FAISS:
    """Loads the FAISS index from disk once and reuses it for every request."""
    global _vector_store
    if _vector_store is None:
        if not config.FAISS_INDEX_DIR.exists():
            raise RuntimeError("FAISS index not found. Run `python ingest.py` first.")
        _vector_store = FAISS.load_local(
            str(config.FAISS_INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # safe: we created this index ourselves
            normalize_L2=True,
        )
    return _vector_store


def retrieve(query: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """
    Returns the top-k chunks with a cosine similarity score, best match first.
    Score is between -1 and 1; closer to 1 means more relevant.
    """
    store = load_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    # FAISS returns squared L2 distance. For unit-length vectors:
    #   distance = 2 - 2 * cosine   =>   cosine = 1 - distance / 2
    return [(doc, 1 - float(distance) / 2) for doc, distance in results]


def format_context(docs: list[Document]) -> str:
    """Joins the retrieved chunks into one text block for the prompt."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs
    )
