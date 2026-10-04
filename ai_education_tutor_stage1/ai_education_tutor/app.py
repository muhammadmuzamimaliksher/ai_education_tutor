import streamlit as st

from config import (
    APP_TITLE,
    GROQ_API_KEY,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
)

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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 AI Education / AI Tutor")

st.write(
    "Learn from your study material or ask the AI Tutor "
    "general educational questions."
)


# =========================================================
# SESSION STATE
# =========================================================

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")


mode = st.sidebar.radio(
    "Choose Mode",
    [
        "📚 PDF Question Answering",
        "🤖 AI Tutor",
    ],
)


# Remove emoji for internal comparison
if mode.startswith("📚"):
    selected_mode = "PDF Question Answering"
else:
    selected_mode = "AI Tutor"


model = st.sidebar.selectbox(
    "AI Model",
    AVAILABLE_MODELS,
    index=0,
)


academic_level = st.sidebar.selectbox(
    "Academic Level",
    [
        "School",
        "College",
        "University",
        "Professional",
        "General",
    ],
)


subject = st.sidebar.text_input(
    "Subject",
    placeholder="e.g. Physics, Computer Science",
)


language = st.sidebar.selectbox(
    "Answer Language",
    [
        "English",
        "Urdu",
        "Roman Urdu",
    ],
)


explanation_style = st.sidebar.selectbox(
    "Explanation Style",
    [
        "Simple",
        "Detailed",
        "Step-by-Step",
        "Exam Preparation",
    ],
)


# =========================================================
# PDF MODE
# =========================================================

if selected_mode == "PDF Question Answering":

    st.header("📚 PDF Question Answering")

    st.info(
        "Upload your study PDF. The AI will answer using "
        "the retrieved information from your document."
    )

    uploaded_file = st.file_uploader(
        "Upload Study PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        # Process only when a new file is uploaded
        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "📄 Reading and processing PDF..."
            ):

                try:

                    full_text = extract_text_from_pdf(
                        uploaded_file
                    )

                    chunks = split_text(
                        full_text,
                        chunk_size=800,
                        chunk_overlap=100,
                    )

                    if not chunks:
                        st.error(
                            "❌ No usable text chunks were "
                            "created from this PDF."
                        )

                        st.session_state.rag_ready = False
                        st.stop()

                    with st.spinner(
                        "🧠 Creating document embeddings..."
                    ):

                        embedding_model = (
                            load_embedding_model()
                        )

                        knowledge_base = (
                            build_knowledge_base(
                                chunks,
                                embedding_model,
                            )
                        )

                    st.session_state.embedding_model = (
                        embedding_model
                    )

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

                    st.success(
                        f"✅ PDF ready! "
                        f"{len(chunks)} text chunks created."
                    )

                except Exception as error:

                    st.session_state.rag_ready = False

                    st.error(
                        "❌ Could not process the PDF."
                    )

                    with st.expander(
                        "Technical error"
                    ):
                        st.code(str(error))

        else:

            st.success(
                f"✅ {uploaded_file.name} is ready."
            )


    # -----------------------------------------------------
    # QUESTION
    # -----------------------------------------------------

    question = st.text_area(
        "Ask a question about your PDF",
        placeholder=(
            "Example: Explain the main concept "
            "discussed in Chapter 1."
        ),
        height=130,
    )


    # -----------------------------------------------------
    # ASK BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚀 Ask Question",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

            st.stop()


        if not st.session_state.rag_ready:

            st.warning(
                "⚠️ Please upload and process a PDF first."
            )

            st.stop()


        # -------------------------------------------------
        # RAG SEARCH
        # -------------------------------------------------

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
                    "Technical error"
                ):
                    st.code(str(error))

                st.stop()


        # -------------------------------------------------
        # RELEVANCE CHECK
        # -------------------------------------------------

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


        best_score = get_best_relevance_score(
            results
        )


        # -------------------------------------------------
        # SHOW RETRIEVAL
        # -------------------------------------------------

        with st.expander(
            "🔎 Retrieved Study Material"
        ):

            st.caption(
                f"Best relevance score: "
                f"{best_score:.3f}"
            )

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


        retrieved_context = "\n\n".join(
            result["text"]
            for result in results
        )


        # -------------------------------------------------
        # REQUEST DATA
        # -------------------------------------------------

        request_data = {

            "mode": "PDF Question Answering",

            "question": question.strip(),

            "academic_level": academic_level,

            "subject": subject,

            "language": language,

            "explanation_style": explanation_style,

            "retrieved_context": retrieved_context,
        }


        # -------------------------------------------------
        # MULTI AGENT
        # -------------------------------------------------

        with st.spinner(
            "🤖 Tutor → Research → Evaluation..."
        ):

            try:

                answer = run_ai_tutor(
                    request_data=request_data,
                    api_key=GROQ_API_KEY,
                    model=model,
                )

                if answer:

                    st.subheader(
                        "📖 AI Tutor Answer"
                    )

                    st.markdown(answer)

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

                    st.code(str(error))


# =========================================================
# GENERAL AI TUTOR MODE
# =========================================================

else:

    st.header("🤖 AI Tutor")

    st.info(
        "Ask the AI Tutor any educational question. "
        "You do not need to upload a PDF in this mode."
    )


    question = st.text_area(
        "What would you like to learn?",
        placeholder=(
            "Example: Explain Newton's three laws "
            "of motion with simple examples."
        ),
        height=160,
    )


    if st.button(
        "🚀 Ask AI Tutor",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please enter your question."
            )

            st.stop()


        # -------------------------------------------------
        # GENERAL AI REQUEST
        # -------------------------------------------------

        request_data = {

            "mode": "AI Tutor",

            "question": question.strip(),

            "academic_level": academic_level,

            "subject": subject,

            "language": language,

            "explanation_style": explanation_style,

            "retrieved_context": "",
        }


        # -------------------------------------------------
        # MULTI AGENT
        # -------------------------------------------------

        with st.spinner(
            "🤖 Tutor → Research → Evaluation..."
        ):

            try:

                answer = run_ai_tutor(
                    request_data=request_data,
                    api_key=GROQ_API_KEY,
                    model=model,
                )

                if answer:

                    st.subheader(
                        "🎓 AI Tutor Answer"
                    )

                    st.markdown(answer)

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

                    st.code(str(error))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 AI Education / AI Tutor • "
    "RAG + Multi-Agent Educational System"
)
