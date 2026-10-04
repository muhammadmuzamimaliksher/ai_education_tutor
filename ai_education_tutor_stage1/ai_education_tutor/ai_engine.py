import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the embedding model."""

    return SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts, embedding_model):
    """Convert text chunks into embeddings."""

    if not texts:
        raise ValueError("No text chunks were provided.")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype="float32")


def create_faiss_index(embeddings):
    """Create a FAISS similarity search index."""

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings were provided.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def build_knowledge_base(chunks, embedding_model):
    """Build a FAISS knowledge base from document chunks."""

    if not chunks:
        raise ValueError("No chunks available.")

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    index = create_faiss_index(
        embeddings
    )

    return index


def search_knowledge_base(
    query,
    embedding_model,
    index,
    chunks,
    top_k=4,
):
    """Find the most relevant document chunks."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if index is None:
        raise ValueError("Knowledge base is not available.")

    if not chunks:
        raise ValueError("No document chunks are available.")

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    number_to_return = min(
        top_k,
        len(chunks),
    )

    scores, positions = index.search(
        query_embedding,
        number_to_return,
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0],
    ):
        if position < 0:
            continue

        results.append(
            {
                "text": chunks[position],
                "score": float(score),
                "position": int(position),
            }
        )

    return results
