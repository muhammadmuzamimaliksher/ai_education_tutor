import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# RAG SETTINGS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum similarity score required for retrieved content.
#
# Higher value = stricter retrieval
# Lower value  = more permissive retrieval
#
# Start with 0.35 and test with your own PDFs.
MIN_RELEVANCE_SCORE = 0.30


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    texts,
    embedding_model,
):
    """
    Convert document chunks into normalized embeddings.
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

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base(
    chunks,
    embedding_model,
):
    """
    Build an in-memory vector knowledge base.
    """

    if not chunks:
        raise ValueError(
            "No document chunks are available."
        )

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
    }


# ============================================================
# SEARCH KNOWLEDGE BASE
# ============================================================

def search_knowledge_base(
    query,
    embedding_model,
    knowledge_base,
    top_k=4,
    min_score=MIN_RELEVANCE_SCORE,
):
    """
    Search the knowledge base and apply a relevance threshold.

    Only chunks with a similarity score >= min_score
    are returned.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if knowledge_base is None:
        raise ValueError(
            "Knowledge base is not available."
        )

    chunks = knowledge_base.get(
        "chunks",
        [],
    )

    embeddings = knowledge_base.get(
        "embeddings",
        None,
    )

    if not chunks:
        return []

    if embeddings is None:
        raise ValueError(
            "Knowledge base embeddings are missing."
        )

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # --------------------------------------------------------
    # Calculate cosine similarity
    #
    # Because both document and query embeddings are
    # normalized, dot product = cosine similarity.
    # --------------------------------------------------------

    scores = np.dot(
        embeddings,
        query_embedding[0],
    )

    # --------------------------------------------------------
    # Sort highest score first
    # --------------------------------------------------------

    positions = np.argsort(
        scores
    )[::-1]

    results = []

    for position in positions:

        score = float(
            scores[position]
        )

        # ----------------------------------------------------
        # RELEVANCE THRESHOLD
        # ----------------------------------------------------

        if score < min_score:
            continue

        results.append(
            {
                "text": chunks[position],
                "score": score,
                "position": int(position),
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# CHECK WHETHER RELEVANT INFORMATION EXISTS
# ============================================================

def has_relevant_information(
    search_results,
):
    """
    Return True when at least one sufficiently relevant
    chunk was found.
    """

    if not search_results:
        return False

    return True


# ============================================================
# GET BEST RELEVANCE SCORE
# ============================================================

def get_best_relevance_score(
    search_results,
):
    """
    Return the highest retrieval score.

    Returns 0.0 if no results exist.
    """

    if not search_results:
        return 0.0

    return max(
        result["score"]
        for result in search_results
    )
