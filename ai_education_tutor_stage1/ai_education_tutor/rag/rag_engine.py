import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Create Embedding Model
# ---------------------------------------------------------

def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# Create Embeddings
# ---------------------------------------------------------

def create_embeddings(
    texts,
    embedding_model,
):
    """
    Convert text chunks into numerical embeddings.
    """

    if not texts:
        raise ValueError(
            "No text chunks were provided."
        )

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


# ---------------------------------------------------------
# Create FAISS Index
# ---------------------------------------------------------

def create_faiss_index(embeddings):
    """
    Create a FAISS similarity-search index.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            "No embeddings were provided."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Search Knowledge Base
# ---------------------------------------------------------

def search_knowledge_base(
    query,
    embedding_model,
    index,
    chunks,
    top_k=4,
):
    """
    Search the knowledge base for relevant chunks.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if index is None:
        raise ValueError(
            "FAISS index has not been created."
        )

    if not chunks:
        raise ValueError(
            "Knowledge base contains no chunks."
        )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

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


# ---------------------------------------------------------
# Build Knowledge Base
# ---------------------------------------------------------

def build_knowledge_base(
    chunks,
    embedding_model,
):
    """
    Create embeddings and a FAISS index
    from document chunks.
    """

    if not chunks:
        raise ValueError(
            "Cannot build knowledge base from empty chunks."
        )

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    index = create_faiss_index(
        embeddings
    )

    return index
