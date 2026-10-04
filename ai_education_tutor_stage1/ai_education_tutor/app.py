import streamlit as st

from config import (
    APP_TITLE,
    GROQ_API_KEY,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
)

from ai_engine import run_ai_tutor

from document_processor import (
    extract_text_from_pdf,
    split_text,
)

from rag_engine import (
    load_embedding_model,
    build_knowledge_base,
    search_knowledge_base,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🎓 AI Education / AI Tutor")

st.write(
    "AI-powered education platform with "
    "Multi-Agent and RAG-based learning support."
)


# =========================================================
# API KEY CHECK
# =========================================================

if not GROQ_API_KEY:

    st.error(
        "❌ GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to Streamlit Secrets "
        "and reboot the application."
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Learning Settings")

    academic_level = st.selectbox(
        "🎓 Academic Level",
        [
            "Grade 1-5",
            "Grade 6-8",
            "Grade 9-10",
            "Grade 11-12",
            "Bachelor",
            "Master",
            "PhD",
        ],
    )

    subject = st.selectbox(
        "📚 Subject",
        [
            "General",
            "Mathematics",
            "Science",
            "Physics",
            "Chemistry",
            "Biology",
            "English",
            "Computer Science",
            "Artificial Intelligence",
            "Engineering",
            "Business",
            "Economics",
            "History",
            "Geography",
            "Other",
        ],
    )

    language = st.selectbox(
        "🌐 Response Language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
            "Arabic",
            "Simple English",
        ],
    )

    explanation_style = st.selectbox(
        "🧠 Explanation Style",
        [
            "Simple",
            "Detailed",
            "Step-by-step",
            "Academic",
            "Exam Preparation",
        ],
    )

    model = st.selectbox(
        "🤖 AI Model",
        AVAILABLE_MODELS,
        index=(
            AVAILABLE_MODELS.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in AVAILABLE_MODELS
            else 0
        ),
    )


# =========================================================
# RAG DOCUMENT SECTION
# =========================================================

st.subheader("📚 Study Material")

uploaded_file = st.file_uploader(
    "Upload a PDF textbook, lecture note, or study material",
    type=["pdf"],
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Processing your study material..."
        ):

            try:

                # -----------------------------------------
                # Extract text
                # -----------------------------------------

                text = extract_text_from_pdf(
                    uploaded_file
                )

                # -----------------------------------------
                # Split text
                # -----------------------------------------

                chunks = split_text(
                    text,
                    chunk_size=800,
                    chunk_overlap=100,
                )

                if not chunks:
                    raise ValueError(
                        "No usable text was found in the PDF."
                    )

                # -----------------------------------------
                # Load embedding model
                # -----------------------------------------

                if (
                    st.session_state.embedding_model
                    is None
                ):

                    st.session_state.embedding_model = (
                        load_embedding_model()
                    )

                # -----------------------------------------
                # Build FAISS database
                # -----------------------------------------

                index = build_knowledge_base(
                    chunks,
                    st.session_state.embedding_model,
                )

                # -----------------------------------------
                # Save in session
                # -----------------------------------------

                st.session_state.document_chunks = chunks

                st.session_state.rag_index = index

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.rag_ready = True

                st.success(
                    "✅ Study material processed successfully."
                )

            except Exception as error:

                st.session_state.rag_ready = False

                st.error(
                    "❌ Could not process the PDF."
                )

                with st.expander(
                    "Technical details"
                ):
                    st.code(str(error))


# =========================================================
# KNOWLEDGE BASE STATUS
# =========================================================

if st.session_state.rag_ready:

    st.success(
        f"📗 Knowledge Base Ready: "
        f"{st.session_state.document_name}"
    )

    st.caption(
        f"{len(st.session_state.document_chunks)} "
        "text chunks are available for retrieval."
    )


# =========================================================
# QUESTION SECTION
# =========================================================

st.divider()

st.subheader("💬 Ask Your AI Tutor")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: Explain Newton's second law "
        "using the uploaded textbook."
    ),
    height=150,
)


# =========================================================
# ASK AI TUTOR
# =========================================================

if st.button(
    "🚀 Ask AI Tutor",
    type="primary",
    use_container_width=True,
):

    # -----------------------------------------------------
    # CHECK QUESTION
    # -----------------------------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK RAG KNOWLEDGE BASE
    # -----------------------------------------------------

    if not st.session_state.rag_ready:

        st.warning(
            "⚠️ Please upload a study PDF before "
            "asking a document-based question."
        )

        st.stop()


    # -----------------------------------------------------
    # RAG SEARCH
    # -----------------------------------------------------

    with st.spinner(
        "🔎 Searching your study material..."
    ):

        try:

            results = search_knowledge_base(
                query=question.strip(),
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
                "❌ RAG search failed."
            )

            with st.expander(
                "RAG technical details"
            ):

                st.code(str(error))

            st.stop()


    # -----------------------------------------------------
    # RELEVANCE CHECK
    # -----------------------------------------------------

    if not results:

        st.warning(
            "⚠️ I couldn't find enough relevant "
            "information in the uploaded study material "
            "to answer this question confidently."
        )

        st.info(
            "Please ask a question related to the "
            "uploaded PDF or upload a more relevant "
            "study document."
        )

        st.stop()


    # -----------------------------------------------------
    # CREATE RETRIEVED CONTEXT
    # -----------------------------------------------------

    retrieved_context = "\n\n".join(
        result["text"]
        for result in results
    )


    # -----------------------------------------------------
    # OPTIONAL: SHOW RAG INFORMATION
    # -----------------------------------------------------

    with st.expander(
        "🔎 Retrieved Study Material"
    ):

        for number, result in enumerate(
            results,
            start=1,
        ):

            st.markdown(
                f"**Source Chunk {number}**"
            )

            st.write(
                result["text"]
            )

            st.caption(
                f"Relevance Score: "
                f"{result['score']:.3f}"
            )

            st.divider()


    # -----------------------------------------------------
    # AI REQUEST
    # -----------------------------------------------------

    request_data = {

        "question": question.strip(),

        "academic_level": academic_level,

        "subject": subject,

        "language": language,

        "explanation_style": explanation_style,

        "retrieved_context": retrieved_context,
    }


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.spinner(
        "🤔 Tutor → Research → Evaluation..."
    ):

        try:

            answer = run_ai_tutor(
                request_data=request_data,
                api_key=GROQ_API_KEY,
                model=model,
            )

            # -------------------------------------------------
            # FINAL ANSWER
            # -------------------------------------------------

            if answer:

                st.subheader(
                    "📖 AI Tutor Answer"
                )

                st.markdown(
                    answer
                )

            else:

                st.error(
                    "❌ AI returned an empty answer."
                )

        except Exception as error:

            st.error(
                "❌ Unable to generate the answer."
            )

            with st.expander(
                "Technical error"
            ):

                st.code(
                    str(error)
                )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Education / AI Tutor • "
    "RAG + Multi-Agent Architecture"
)
