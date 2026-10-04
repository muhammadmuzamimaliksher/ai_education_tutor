import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the sentence-transformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts, embedding_model):
    """Convert text chunks into normalized embeddings."""

    if not texts:
        raise ValueError("No text chunks were provided.")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype="float32")


def build_knowledge_base(chunks, embedding_model):
    """Create an in-memory vector knowledge base."""

    if not chunks:
        raise ValueError("No document chunks are available.")

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
    }


def search_knowledge_base(
    query,
    embedding_model,
    knowledge_base,
    top_k=4,
):
    """Search the knowledge base for the most relevant chunks."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if knowledge_base is None:
        raise ValueError("Knowledge base is not available.")

    chunks = knowledge_base["chunks"]
    embeddings = knowledge_base["embeddings"]

    if not chunks:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores = np.dot(
        embeddings,
        query_embedding[0],
    )

    top_k = min(top_k, len(chunks))

    positions = np.argsort(scores)[::-1][:top_k]

    results = []

    for position in positions:
        results.append(
            {
                "text": chunks[position],
                "score": float(scores[position]),
                "position": int(position),
            }
        )

    return results
