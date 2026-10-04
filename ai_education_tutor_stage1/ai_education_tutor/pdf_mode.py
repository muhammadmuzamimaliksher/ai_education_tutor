import hashlib

import streamlit as st

from document_processor import (
    extract_pages_from_pdf,
    split_text,
)

from rag_engine import (
    load_embedding_model,
    build_knowledge_base,
    search_knowledge_base,
    has_relevant_information,
    get_best_relevance_score,
)

from ai_engine import run_ai_tutor

from session_manager import (
    reset_pdf_conversation,
)


# =========================================================
# PDF FALLBACK
# =========================================================

PDF_FALLBACK = (
    "I couldn't find this information in the provided PDF."
)


# =========================================================
# FILE HASH
# =========================================================

def get_file_hash(uploaded_file):
    """
    Create a unique hash for the uploaded PDF.

    This prevents a problem where the user uploads
    a different PDF with the same filename.
    """

    file_bytes = uploaded_file.getvalue()

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# =========================================================
# VERBATIM SOURCE EXTRACTION
# =========================================================

def extract_verbatim_answer(
    question,
    search_results,
):
    """
    Return exact text from retrieved PDF passages.

    IMPORTANT:

    This function does NOT ask an AI model to rewrite
    the PDF.

    The returned wording comes directly from the
    retrieved PDF chunks.
    """

    if not search_results:
        return PDF_FALLBACK

    # -----------------------------------------------------
    # Current safe strategy:
    #
    # Return the strongest relevant PDF passage.
    #
    # This guarantees that the wording is taken directly
    # from the PDF source.
    # -----------------------------------------------------

    best_result = search_results[0]

    text = best_result.get(
        "text",
        "",
    )

    if not text:
        return PDF_FALLBACK

    page_number = best_result.get(
        "page_number"
    )

    if page_number is not None:

        return (
            f"**Source — PDF Page {page_number}**\n\n"
            f"{text}"
        )

    return text


# =========================================================
# DISPLAY SEARCH INFORMATION
# =========================================================

def display_pdf_source_info(
    search_results
):
    """
    Display source information for the retrieved
    PDF passage.
    """

    if not search_results:
        return

    best_result = search_results[0]

    score = best_result.get(
        "score",
        0.0,
    )

    page_number = best_result.get(
        "page_number"
    )

    if page_number is not None:

        st.caption(
            f"📄 Source: PDF Page {page_number} "
            f"| Relevance: {score:.3f}"
        )

    else:

        st.caption(
            f"📄 Source: PDF "
            f"| Relevance: {score:.3f}"
        )


# =========================================================
# MAIN PDF MODE
# =========================================================

def render_pdf_mode(settings):
    """
    Render strict PDF Question Answering mode.
    """

    st.header(
        "📚 PDF Question Answering"
    )

    st.info(
        "PDF mode uses only the uploaded PDF. "
        "Final answers are extracted from the PDF "
        "source text instead of being rewritten by AI."
    )


    # =====================================================
    # PDF UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        key="pdf_uploader",
    )


    # =====================================================
    # NO PDF
    # =====================================================

    if uploaded_file is None:

        if st.session_state.uploaded_file_name:

            st.caption(
                f"Current PDF: "
                f"{st.session_state.uploaded_file_name}"
            )

        else:

            st.warning(
                "Please upload a PDF to use "
                "PDF Question Answering."
            )

        # -------------------------------------------------
        # DISPLAY EXISTING CONVERSATION
        # -------------------------------------------------

        for message in st.session_state.pdf_messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        return


    # =====================================================
    # FILE HASH
    # =====================================================

    current_file_hash = get_file_hash(
        uploaded_file
    )

    previous_file_hash = st.session_state.get(
        "pdf_file_hash",
        "",
    )


    # =====================================================
    # PROCESS NEW PDF
    # =====================================================

    if (
        current_file_hash
        != previous_file_hash
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                # -----------------------------------------
                # STEP 1 — EXTRACT PAGES
                # -----------------------------------------

                pages = extract_pages_from_pdf(
                    uploaded_file
                )

                # -----------------------------------------
                # STEP 2 — CREATE PAGE-AWARE CHUNKS
                # -----------------------------------------

                chunks = split_text(
                    pages,
                    chunk_size=800,
                    chunk_overlap=100,
                )

                if not chunks:
                    raise ValueError(
                        "No readable text chunks were created."
                    )

                # -----------------------------------------
                # STEP 3 — LOAD EMBEDDING MODEL
                # -----------------------------------------

                if (
                    st.session_state.embedding_model
                    is None
                ):

                    st.session_state.embedding_model = (
                        load_embedding_model()
                    )

                embedding_model = (
                    st.session_state.embedding_model
                )

                # -----------------------------------------
                # STEP 4 — BUILD KNOWLEDGE BASE
                # -----------------------------------------

                knowledge_base = (
                    build_knowledge_base(
                        chunks,
                        embedding_model,
                    )
                )

                # -----------------------------------------
                # STEP 5 — SAVE STATE
                # -----------------------------------------

                st.session_state.rag_index = (
                    knowledge_base
                )

                st.session_state.document_chunks = (
                    chunks
                )

                st.session_state.rag_ready = True

                st.session_state.uploaded_file_name = (
                    uploaded_file.name
                )

                st.session_state.pdf_file_hash = (
                    current_file_hash
                )

                # -----------------------------------------
                # IMPORTANT:
                # Clear ONLY PDF conversation.
                # AI Tutor memory remains untouched.
                # -----------------------------------------

                reset_pdf_conversation()

                st.success(
                    f"PDF processed successfully. "
                    f"{len(pages)} pages and "
                    f"{len(chunks)} searchable chunks created."
                )

            except Exception as error:

                st.error(
                    f"PDF processing failed: {error}"
                )

                return


    # =====================================================
    # PDF STATUS
    # =====================================================

    st.caption(
        f"📄 PDF: "
        f"{st.session_state.uploaded_file_name}"
    )

    st.caption(
        f"🔎 Searchable chunks: "
        f"{len(st.session_state.document_chunks)}"
    )


    # =====================================================
    # DISPLAY CONVERSATION
    # =====================================================

    for message in st.session_state.pdf_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if not question:
        return


    # =====================================================
    # SAVE USER QUESTION
    # =====================================================

    st.session_state.pdf_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # =====================================================
    # RAG KNOWLEDGE BASE CHECK
    # =====================================================

    if not st.session_state.rag_ready:

        answer = PDF_FALLBACK

        st.session_state.pdf_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        return


    # =====================================================
    # SEARCH PDF
    # =====================================================

    with st.spinner(
        "Searching the PDF..."
    ):

        try:

            search_results = search_knowledge_base(
                query=question,
                embedding_model=(
                    st.session_state.embedding_model
                ),
                knowledge_base=(
                    st.session_state.rag_index
                ),
                top_k=4,
                min_score=0.35,
            )

        except Exception as error:

            answer = (
                f"PDF search failed: {error}"
            )

            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            with st.chat_message("assistant"):
                st.error(answer)

            return


    # =====================================================
    # RELEVANCE CHECK
    # =====================================================

    if not has_relevant_information(
        search_results
    ):

        answer = PDF_FALLBACK

        st.session_state.pdf_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.pdf_last_question = (
            question
        )

        st.session_state.pdf_last_answer = (
            answer
        )

        st.session_state.pdf_last_context = ""

        return


    # =====================================================
    # BEST SCORE
    # =====================================================

    best_score = get_best_relevance_score(
        search_results
    )


    # =====================================================
    # VERBATIM ANSWER
    # =====================================================

    answer = extract_verbatim_answer(
        question,
        search_results,
    )


    # =====================================================
    # SAVE CONTEXT
    # =====================================================

    context_parts = []

    for result in search_results:

        page_number = result.get(
            "page_number"
        )

        text = result.get(
            "text",
            "",
        )

        if page_number is not None:

            context_parts.append(
                f"[PDF Page {page_number}]\n{text}"
            )

        else:

            context_parts.append(
                text
            )

    retrieved_context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # SAVE LAST PDF STATE
    # =====================================================

    st.session_state.pdf_last_question = (
        question
    )

    st.session_state.pdf_last_answer = (
        answer
    )

    st.session_state.pdf_last_context = (
        retrieved_context
    )


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.pdf_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        display_pdf_source_info(
            search_results
        )

        st.caption(
            f"🔎 Best relevance score: "
            f"{best_score:.3f}"
        )
