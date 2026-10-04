# =========================================================
# PDF QUESTION ANSWERING MODE
# =========================================================

import streamlit as st

from document_processor import (
    extract_text_from_pdf,
    split_text,
)

from rag_engine import (
    load_embedding_model,
    build_knowledge_base,
    search_knowledge_base,
    get_best_relevance_score,
)

from ai_engine import run_ai_tutor

from session_manager import (
    reset_pdf_conversation,
)


def render_pdf_mode(settings):

    st.header("📚 PDF Question Answering")

    st.info(
        "Upload your study material and ask questions "
        "based on the PDF."
    )


    # =====================================================
    # PDF UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(
        "📄 Upload Study PDF",
        type=["pdf"],
    )


    # =====================================================
    # PROCESS PDF
    # =====================================================

    if uploaded_file is not None:

        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "📖 Processing PDF..."
            ):

                try:

                    # -----------------------------------------
                    # EXTRACT TEXT
                    # -----------------------------------------

                    document_text = (
                        extract_text_from_pdf(
                            uploaded_file
                        )
                    )


                    # -----------------------------------------
                    # CHUNK TEXT
                    # -----------------------------------------

                    chunks = split_text(
                        document_text,
                        chunk_size=800,
                        chunk_overlap=100,
                    )


                    if not chunks:

                        raise ValueError(
                            "No readable text was found."
                        )


                    # -----------------------------------------
                    # EMBEDDING MODEL
                    # -----------------------------------------

                    if (
                        st.session_state.embedding_model
                        is None
                    ):

                        st.session_state.embedding_model = (
                            load_embedding_model()
                        )


                    # -----------------------------------------
                    # BUILD KNOWLEDGE BASE
                    # -----------------------------------------

                    knowledge_base = (
                        build_knowledge_base(
                            chunks,
                            st.session_state.embedding_model,
                        )
                    )


                    # -----------------------------------------
                    # SAVE RAG DATA
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


                    # -----------------------------------------
                    # CLEAR ONLY PDF MEMORY
                    # -----------------------------------------

                    reset_pdf_conversation()


                    st.success(
                        f"✅ PDF processed successfully."
                    )

                    st.info(
                        f"📄 {len(chunks)} chunks created."
                    )


                except Exception as error:

                    st.session_state.rag_ready = False

                    st.error(
                        f"❌ PDF processing failed: {error}"
                    )


    # =====================================================
    # PDF STATUS
    # =====================================================

    if st.session_state.rag_ready:

        st.success(
            f"📄 Active PDF: "
            f"{st.session_state.uploaded_file_name}"
        )

    else:

        st.warning(
            "Please upload a PDF before asking questions."
        )


    # =====================================================
    # DISPLAY PDF CONVERSATION
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


    if question:

        process_pdf_question(
            question,
            settings,
        )


# =========================================================
# PROCESS PDF QUESTION
# =========================================================

def process_pdf_question(
    question,
    settings,
):

    if not settings["api_key"]:

        st.error(
            "GROQ_API_KEY is not configured."
        )

        return


    if not st.session_state.rag_ready:

        st.warning(
            "Please upload and process a PDF first."
        )

        return


    # =====================================================
    # RAG SEARCH
    # =====================================================

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
        )


    except Exception as error:

        st.error(
            f"❌ RAG search failed: {error}"
        )

        return


    # =====================================================
    # NO RELEVANT INFORMATION
    # =====================================================

    if not search_results:

        answer = (
            "I couldn't find enough information in the "
            "provided study material to answer this confidently. "
            "Please upload a relevant document or provide more context."
        )

        st.session_state.pdf_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        st.session_state.pdf_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.pdf_last_question = question

        st.session_state.pdf_last_answer = answer

        st.session_state.pdf_last_context = ""

        st.rerun()

        return


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context_parts = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):

        context_parts.append(
            f"[Retrieved Section {index}]\n"
            f"{result['text']}"
        )


    retrieved_context = "\n\n".join(
        context_parts
    )


    best_score = get_best_relevance_score(
        search_results
    )


    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.pdf_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # =====================================================
    # REQUEST DATA
    # =====================================================

    request_data = {

        "mode": "pdf",

        "question": question,

        "topic": settings["subject"],

        "subject": settings["subject"],

        "academic_level": settings[
            "academic_level"
        ],

        "language": settings["language"],

        "explanation_style": settings[
            "explanation_style"
        ],

        "context": retrieved_context,

        "history": (
            st.session_state.pdf_messages[:-1]
        ),
    }


    # =====================================================
    # AI PIPELINE
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Tutor → Research → Evaluator..."
        ):

            try:

                result = run_ai_tutor(
                    request_data=request_data,
                    api_key=settings["api_key"],
                    model=settings["model"],
                )


                if isinstance(result, dict):

                    answer = result.get(
                        "answer",
                        result.get(
                            "final_answer",
                            "",
                        ),
                    )

                else:

                    answer = str(result)


                if not answer:

                    answer = (
                        "The AI did not return a usable answer."
                    )


                st.markdown(answer)


            except Exception as error:

                answer = (
                    f"❌ AI processing failed: {error}"
                )

                st.error(answer)


    # =====================================================
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    st.session_state.pdf_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    st.session_state.pdf_last_question = (
        question
    )

    st.session_state.pdf_last_answer = (
        answer
    )

    st.session_state.pdf_last_context = (
        retrieved_context
    )


    st.caption(
        f"🔎 Retrieved sections: "
        f"{len(search_results)} | "
        f"Best relevance: {best_score:.2f}"
    )

    st.rerun()
