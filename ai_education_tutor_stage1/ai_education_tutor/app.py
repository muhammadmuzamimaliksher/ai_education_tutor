import streamlit as st

from config import (
    APP_TITLE,
    GROQ_API_KEY,
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

from ai_engine import (
    run_ai_tutor,
    run_learning_tool,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_context" not in st.session_state:
    st.session_state.last_context = ""


# =========================================================
# TITLE
# =========================================================

st.title("🎓 AI Education / AI Tutor")

st.write(
    "Learn from your study material or ask the AI Tutor "
    "general educational questions."
)


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
    placeholder="e.g. Physics",
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
# CONVERSATION CONTROLS
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("💬 Conversation")

if st.sidebar.button(
    "🗑️ Clear Conversation",
    use_container_width=True,
):

    st.session_state.messages = []

    st.session_state.last_question = ""

    st.session_state.last_answer = ""

    st.session_state.last_context = ""

    st.rerun()


if st.session_state.messages:

    st.sidebar.caption(
        f"{len(st.session_state.messages)} "
        "messages in current session."
    )


# =========================================================
# PDF MODE
# =========================================================

if selected_mode == "PDF Question Answering":

    st.header("📚 PDF Question Answering")

    st.info(
        "Upload a study PDF. The AI will answer using "
        "relevant information retrieved from the document."
    )


    # -----------------------------------------------------
    # UPLOAD PDF
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Study PDF",
        type=["pdf"],
    )


    if uploaded_file is not None:

        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "📄 Reading PDF..."
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
                            "❌ No usable text was found."
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

                    # New document = new conversation
                    st.session_state.messages = []

                    st.session_state.last_question = ""

                    st.session_state.last_answer = ""

                    st.session_state.last_context = ""

                    st.success(
                        f"✅ PDF ready! "
                        f"{len(chunks)} chunks created."
                    )

                except Exception as error:

                    st.session_state.rag_ready = False

                    st.error(
                        "❌ Could not process PDF."
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.code(str(error))

        else:

            st.success(
                f"✅ {uploaded_file.name} is ready."
            )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

if selected_mode == "PDF Question Answering":

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

else:

    question = st.chat_input(
        "Ask your AI Tutor anything..."
    )


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    question = question.strip()


    if not question:

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()


    # =====================================================
    # PDF MODE
    # =====================================================

    if selected_mode == "PDF Question Answering":

        if not st.session_state.rag_ready:

            st.warning(
                "⚠️ Please upload a PDF first."
            )

            st.stop()


        # -------------------------------------------------
        # SHOW USER QUESTION
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        # -------------------------------------------------
        # SEARCH RAG
        # -------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching study material..."
            ):

                try:

                    results = search_knowledge_base(
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
                        "❌ RAG search failed."
                    )

                    st.code(str(error))

                    st.stop()


            # -------------------------------------------------
            # RELEVANCE CHECK
            # -------------------------------------------------

            if not results:

                answer = (
                    "I couldn't find enough relevant "
                    "information in the uploaded study "
                    "material to answer this question "
                    "confidently."
                )

                st.warning(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.stop()


            best_score = get_best_relevance_score(
                results
            )


            # -------------------------------------------------
            # CONTEXT
            # -------------------------------------------------

            retrieved_context = "\n\n".join(
                result["text"]
                for result in results
            )


            st.session_state.last_context = (
                retrieved_context
            )


            # -------------------------------------------------
            # SHOW SOURCES
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


            # -------------------------------------------------
            # REQUEST
            # -------------------------------------------------

            request_data = {

                "mode": selected_mode,

                "question": question,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": explanation_style,

                "retrieved_context": retrieved_context,

                "history": st.session_state.messages[
                    :-1
                ],
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

                except Exception as error:

                    st.error(
                        "❌ Unable to generate answer."
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.code(str(error))

                    st.stop()


            if answer:

                st.markdown(answer)

                st.session_state.last_question = (
                    question
                )

                st.session_state.last_answer = (
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


    # =====================================================
    # GENERAL AI TUTOR MODE
    # =====================================================

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            request_data = {

                "mode": "AI Tutor",

                "question": question,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": explanation_style,

                "retrieved_context": "",

                "history": st.session_state.messages[
                    :-1
                ],
            }


            with st.spinner(
                "🤖 Tutor → Research → Evaluation..."
            ):

                try:

                    answer = run_ai_tutor(
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                except Exception as error:

                    st.error(
                        "❌ Unable to generate answer."
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.code(str(error))

                    st.stop()


            if answer:

                st.markdown(answer)

                st.session_state.last_question = (
                    question
                )

                st.session_state.last_answer = (
                    answer
                )

                st.session_state.last_context = ""

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


# =========================================================
# LEARNING TOOLS
# =========================================================

if st.session_state.last_question:

    st.divider()

    st.subheader("🧠 Learning Tools")

    st.caption(
        "Use these tools to learn the current topic "
        "in different ways."
    )


    col1, col2, col3 = st.columns(3)


    # =====================================================
    # EXPLAIN AGAIN
    # =====================================================

    with col1:

        if st.button(
            "🔄 Explain Again",
            use_container_width=True,
        ):

            tool = "Explain Again"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "🔄 Explaining again..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Learning tool failed."
                    )

                    st.code(str(error))


    # =====================================================
    # SIMPLE EXPLANATION
    # =====================================================

    with col2:

        if st.button(
            "🧒 Explain Simply",
            use_container_width=True,
        ):

            tool = "Explain Simply"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "🧒 Simplifying..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Learning tool failed."
                    )

                    st.code(str(error))


    # =====================================================
    # EXAMPLE
    # =====================================================

    with col3:

        if st.button(
            "💡 Give Example",
            use_container_width=True,
        ):

            tool = "Give Example"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "💡 Creating example..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Learning tool failed."
                    )

                    st.code(str(error))


    # =====================================================
    # SECOND ROW
    # =====================================================

    col4, col5, col6 = st.columns(3)


    # =====================================================
    # EXAM ANSWER
    # =====================================================

    with col4:

        if st.button(
            "📝 Exam Answer",
            use_container_width=True,
        ):

            tool = "Exam Answer"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "📝 Preparing exam answer..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Learning tool failed."
                    )

                    st.code(str(error))


    # =====================================================
    # SUMMARY
    # =====================================================

    with col5:

        if st.button(
            "📚 Create Summary",
            use_container_width=True,
        ):

            tool = "Summary"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "📚 Creating summary..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Learning tool failed."
                    )

                    st.code(str(error))


    # =====================================================
    # QUIZ
    # =====================================================

    with col6:

        if st.button(
            "❓ Generate Quiz",
            use_container_width=True,
        ):

            tool = "Quiz"


            request_data = {

                "mode": selected_mode,

                "question": (
                    st.session_state.last_question
                ),

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "retrieved_context": (
                    st.session_state.last_context
                ),

                "history": st.session_state.messages,
            }


            with st.spinner(
                "❓ Generating quiz..."
            ):

                try:

                    tool_answer = run_learning_tool(
                        tool=tool,
                        request_data=request_data,
                        api_key=GROQ_API_KEY,
                        model=model,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "❌ Quiz generation failed."
                    )

                    st.code(str(error))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 AI Education / AI Tutor • "
    "RAG + Multi-Agent + Learning Tools"
)
