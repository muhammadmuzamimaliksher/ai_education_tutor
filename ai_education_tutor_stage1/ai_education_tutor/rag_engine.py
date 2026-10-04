import numpy as np
from sentence_transformers import SentenceTransformer

from document_processor import get_chunk_text


# =========================================================
# EMBEDDING CONFIGURATION
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MIN_RELEVANCE_SCORE = 0.35


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

def load_embedding_model():
    """
    Load the Sentence Transformer embedding model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(
    texts,
    embedding_model,
):
    """
    Create normalized embeddings for text.
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


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(
    chunks,
    embedding_model,
):
    """
    Build the RAG knowledge base.

    Chunks retain:

    - Exact text
    - Page number
    - Chunk number
    """

    if not chunks:
        raise ValueError(
            "No document chunks are available."
        )

    texts = []

    for chunk in chunks:

        text = get_chunk_text(
            chunk
        )

        if text:
            texts.append(text)

    if not texts:
        raise ValueError(
            "No readable chunk text is available."
        )

    embeddings = create_embeddings(
        texts,
        embedding_model,
    )

    return {
        "chunks": chunks,
        "texts": texts,
        "embeddings": embeddings,
    }


# =========================================================
# SEARCH KNOWLEDGE BASE
# =========================================================

def search_knowledge_base(
    query,
    embedding_model,
    knowledge_base,
    top_k=4,
    min_score=MIN_RELEVANCE_SCORE,
):
    """
    Search the PDF knowledge base.

    Returns source records containing:

    - Exact text
    - Similarity score
    - Page number
    - Chunk number
    - Original position
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
        "embeddings"
    )

    if not chunks:
        return []

    if embeddings is None:
        raise ValueError(
            "Knowledge base embeddings are missing."
        )


    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )


    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    scores = np.dot(
        embeddings,
        query_embedding[0],
    )


    # =====================================================
    # SORT RESULTS
    # =====================================================

    positions = np.argsort(
        scores
    )[::-1]


    results = []

    for position in positions:

        score = float(
            scores[position]
        )

        if score < min_score:
            continue

        chunk = chunks[
            int(position)
        ]

        text = get_chunk_text(
            chunk
        )

        if not text:
            continue

        page_number = None
        chunk_number = None

        if isinstance(
            chunk,
            dict,
        ):

            page_number = chunk.get(
                "page_number"
            )

            chunk_number = chunk.get(
                "chunk_number"
            )

        results.append(
            {
                "text": text,
                "score": score,
                "position": int(position),
                "page_number": page_number,
                "chunk_number": chunk_number,
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# RELEVANCE CHECK
# =========================================================

def has_relevant_information(
    search_results
):
    """
    Check whether RAG returned relevant PDF evidence.
    """

    if not search_results:
        return False

    return True


# =========================================================
# BEST RELEVANCE SCORE
# =========================================================

def get_best_relevance_score(
    search_results
):
    """
    Return the highest similarity score.
    """

    if not search_results:
        return 0.0

    return max(
        result["score"]
        for result in search_results
    )


# =========================================================
# FORMAT RETRIEVED CONTEXT
# =========================================================

def format_retrieved_context(
    search_results,
):
    """
    Format retrieved PDF passages for AI agents.

    The original passage text is preserved.
    """

    if not search_results:
        return ""

    sections = []

    for result in search_results:

        page_number = result.get(
            "page_number"
        )

        score = result.get(
            "score",
            0.0,
        )

        text = result.get(
            "text",
            "",
        )

        if page_number is not None:

            header = (
                f"[PDF Page {page_number} | "
                f"Relevance {score:.3f}]"
            )

        else:

            header = (
                f"[PDF Passage | "
                f"Relevance {score:.3f}]"
            )

        sections.append(
            f"{header}\n{text}"
        )

    return "\n\n".join(
        sections
    )


# =========================================================
# EXACT SOURCE PASSAGES
# =========================================================

def get_exact_source_passages(
    search_results,
):
    """
    Return the exact retrieved PDF passages.

    No rewriting or AI generation occurs here.
    """

    if not search_results:
        return []

    passages = []

    for result in search_results:

        text = result.get(
            "text",
            "",
        )

        if not text:
            continue

        passages.append(
            {
                "text": text,
                "page_number": result.get(
                    "page_number"
                ),
                "score": result.get(
                    "score",
                    0.0,
                ),
                "chunk_number": result.get(
                    "chunk_number"
                ),
            }
        )

    return passages
