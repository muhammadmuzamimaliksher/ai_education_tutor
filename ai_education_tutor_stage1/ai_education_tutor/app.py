# =========================================================
# AI EDUCATION / AI TUTOR
# STREAMLIT MAIN APPLICATION
# =========================================================

import streamlit as st

from config import APP_TITLE, AVAILABLE_MODELS, DEFAULT_MODEL, get_groq_api_key
from document_processor import extract_text_from_pdf, split_text
from rag_engine import (
    load_embedding_model,
    build_knowledge_base,
    search_knowledge_base,
    get_best_relevance_score,
)
from ai_engine import run_ai_tutor, run_learning_tool


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

# ---------------------------------------------------------
# RAG / PDF STATE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# PDF QUESTION ANSWERING MEMORY
# ---------------------------------------------------------

if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []

if "pdf_last_question" not in st.session_state:
    st.session_state.pdf_last_question = ""

if "pdf_last_answer" not in st.session_state:
    st.session_state.pdf_last_answer = ""

if "pdf_last_context" not in st.session_state:
    st.session_state.pdf_last_context = ""


# ---------------------------------------------------------
# AI TUTOR MEMORY
# ---------------------------------------------------------

if "tutor_messages" not in st.session_state:
    st.session_state.tutor_messages = []

if "tutor_last_question" not in st.session_state:
    st.session_state.tutor_last_question = ""

if "tutor_last_answer" not in st.session_state:
    st.session_state.tutor_last_answer = ""


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clear_pdf_conversation():
    """Clear only PDF Question Answering conversation."""

    st.session_state.pdf_messages = []
    st.session_state.pdf_last_question = ""
    st.session_state.pdf_last_answer = ""
    st.session_state.pdf_last_context = ""


def clear_tutor_conversation():
    """Clear only AI Tutor conversation."""

    st.session_state.tutor_messages = []
    st.session_state.tutor_last_question = ""
    st.session_state.tutor_last_answer = ""


def clear_current_conversation(selected_mode):
    """Clear conversation only for the selected mode."""

    if selected_mode == "📚 PDF Question Answering":
        clear_pdf_conversation()

    else:
        clear_tutor_conversation()


def reset_pdf_after_new_upload():
    """
    When a new PDF is uploaded, clear only PDF conversation.
    AI Tutor memory remains untouched.
    """

    st.session_state.pdf_messages = []
    st.session_state.pdf_last_question = ""
    st.session_state.pdf_last_answer = ""
    st.session_state.pdf_last_context = ""


def get_current_memory(selected_mode):
    """
    Return the correct conversation memory according
    to the currently selected mode.
    """

    if selected_mode == "📚 PDF Question Answering":
        return (
            st.session_state.pdf_messages,
            st.session_state.pdf_last_question,
            st.session_state.pdf_last_answer,
            st.session_state.pdf_last_context,
        )

    return (
        st.session_state.tutor_messages,
        st.session_state.tutor_last_question,
        st.session_state.tutor_last_answer,
        "",
    )


# =========================================================
# TITLE
# =========================================================

st.title("🎓 AI Education / AI Tutor")

st.markdown(
    """
    **Learn smarter with AI-powered tutoring, PDF question answering,
    RAG-based knowledge retrieval, and multi-agent educational assistance.**
    """
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # -----------------------------------------------------
    # MODE
    # -----------------------------------------------------

    selected_mode = st.radio(
        "Choose Learning Mode",
        [
            "📚 PDF Question Answering",
            "🤖 AI Tutor",
        ],
        index=0,
    )

    st.divider()

    # -----------------------------------------------------
    # AI MODEL
    # -----------------------------------------------------

    model = st.selectbox(
        "🤖 AI Model",
        AVAILABLE_MODELS,
        index=(
            AVAILABLE_MODELS.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in AVAILABLE_MODELS
            else 0
        ),
    )

    st.divider()

    # -----------------------------------------------------
    # EDUCATION SETTINGS
    # -----------------------------------------------------

    academic_level = st.selectbox(
        "🎓 Academic Level",
        [
            "School",
            "College",
            "University",
            "Professional",
            "General",
        ],
    )

    subject = st.text_input(
        "📖 Subject",
        placeholder="e.g. Biology, Physics, SEO, Computer Science",
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
        "✍️ Explanation Style",
        [
            "Detailed",
            "Simple",
            "Step-by-Step",
            "Exam Focused",
            "Beginner Friendly",
        ],
    )

    st.divider()

    # -----------------------------------------------------
    # CLEAR CURRENT CONVERSATION
    # -----------------------------------------------------

    clear_label = (
        "🗑️ Clear PDF Conversation"
        if selected_mode == "📚 PDF Question Answering"
        else "🗑️ Clear Tutor Conversation"
    )

    if st.button(
        clear_label,
        use_container_width=True,
    ):
        clear_current_conversation(selected_mode)

        st.success("Current conversation cleared.")

        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # API STATUS
    # -----------------------------------------------------

    api_key = get_groq_api_key()

    if api_key:
        st.success("✅ Groq API configured")
    else:
        st.error(
            "❌ GROQ_API_KEY is not configured.\n\n"
            "Add it to Streamlit Secrets."
        )


# =========================================================
# CURRENT MEMORY
# =========================================================

(
    current_messages,
    current_last_question,
    current_last_answer,
    current_last_context,
) = get_current_memory(selected_mode)


# =========================================================
# PDF QUESTION ANSWERING MODE
# =========================================================

if selected_mode == "📚 PDF Question Answering":

    st.header("📚 PDF Question Answering")

    st.info(
        "Upload your study material. The AI will retrieve relevant "
        "information from the PDF before generating an answer."
    )

    # -----------------------------------------------------
    # PDF UPLOAD
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "📄 Upload Study PDF",
        type=["pdf"],
        help="Upload a text-based PDF for question answering.",
    )

    # -----------------------------------------------------
    # PROCESS PDF
    # -----------------------------------------------------

    if uploaded_file is not None:

        # Detect a new PDF
        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner("📖 Reading PDF..."):

                try:

                    # Extract PDF text
                    document_text = extract_text_from_pdf(
                        uploaded_file
                    )

                    # Split into chunks
                    chunks = split_text(
                        document_text,
                        chunk_size=800,
                        chunk_overlap=100,
                    )

                    if not chunks:
                        raise ValueError(
                            "No usable text chunks were created from the PDF."
                        )

                    # Load embedding model
                    if st.session_state.embedding_model is None:

                        with st.spinner(
                            "🧠 Loading embedding model..."
                        ):
                            st.session_state.embedding_model = (
                                load_embedding_model()
                            )

                    # Build RAG knowledge base
                    with st.spinner(
                        "🔎 Building knowledge base..."
                    ):

                        knowledge_base = build_knowledge_base(
                            chunks,
                            st.session_state.embedding_model,
                        )

                    # Save RAG state
                    st.session_state.rag_index = knowledge_base
                    st.session_state.document_chunks = chunks
                    st.session_state.rag_ready = True
                    st.session_state.uploaded_file_name = (
                        uploaded_file.name
                    )

                    # IMPORTANT:
                    # New PDF clears ONLY PDF conversation.
                    # AI Tutor conversation remains untouched.
                    reset_pdf_after_new_upload()

                    st.success(
                        f"✅ PDF processed successfully: "
                        f"{uploaded_file.name}"
                    )

                    st.info(
                        f"📄 {len(chunks)} text chunks created."
                    )

                except Exception as error:

                    st.session_state.rag_ready = False
                    st.session_state.rag_index = None
                    st.session_state.document_chunks = []

                    st.error(
                        f"❌ PDF processing failed: {error}"
                    )

        else:

            # Existing PDF
            if st.session_state.rag_ready:

                st.success(
                    f"📄 Active PDF: "
                    f"{st.session_state.uploaded_file_name}"
                )


    # -----------------------------------------------------
    # PDF STATUS
    # -----------------------------------------------------

    if st.session_state.rag_ready:

        st.caption(
            "🔎 RAG knowledge base is ready. "
            "Questions will be answered using retrieved PDF content."
        )

    else:

        st.warning(
            "Please upload and process a PDF before asking "
            "questions in this mode."
        )


# =========================================================
# AI TUTOR MODE
# =========================================================

else:

    st.header("🤖 AI Tutor")

    st.info(
        "Ask general educational questions. "
        "No PDF is required in this mode."
    )

    st.markdown(
        """
        **AI Tutor workflow**

        Question → Tutor Agent → Research Agent → Evaluator Agent → Final Answer
        """
    )


# =========================================================
# DISPLAY CURRENT CONVERSATION
# =========================================================

st.divider()

st.subheader("💬 Conversation")


# Refresh current memory after possible PDF processing
(
    current_messages,
    current_last_question,
    current_last_answer,
    current_last_context,
) = get_current_memory(selected_mode)


if not current_messages:

    if selected_mode == "📚 PDF Question Answering":

        st.caption(
            "No PDF conversation yet. Ask a question about your study material."
        )

    else:

        st.caption(
            "No AI Tutor conversation yet. Ask your first question."
        )


else:

    for message in current_messages:

        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):

            st.markdown(content)


# =========================================================
# CHAT INPUT
# =========================================================

if selected_mode == "📚 PDF Question Answering":

    chat_placeholder = (
        "Ask a question about your uploaded PDF..."
    )

else:

    chat_placeholder = (
        "Ask your AI Tutor a question..."
    )


user_question = st.chat_input(
    chat_placeholder
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_question:

    user_question = user_question.strip()

    if not user_question:
        st.warning("Please enter a question.")

        st.stop()


    # -----------------------------------------------------
    # CHECK API KEY
    # -----------------------------------------------------

    if not api_key:

        st.error(
            "❌ GROQ_API_KEY is not configured.\n\n"
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

        st.stop()


    # -----------------------------------------------------
    # PDF MODE VALIDATION
    # -----------------------------------------------------

    if selected_mode == "📚 PDF Question Answering":

        if not st.session_state.rag_ready:

            st.warning(
                "📄 Please upload and process a PDF first."
            )

            st.stop()


    # -----------------------------------------------------
    # GET CURRENT MODE MEMORY
    # -----------------------------------------------------

    if selected_mode == "📚 PDF Question Answering":

        mode_messages = st.session_state.pdf_messages

    else:

        mode_messages = st.session_state.tutor_messages


    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    user_message = {
        "role": "user",
        "content": user_question,
    }

    mode_messages.append(user_message)


    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_question)


    # -----------------------------------------------------
    # RAG RETRIEVAL
    # -----------------------------------------------------

    retrieved_context = ""
    search_results = []

    if selected_mode == "📚 PDF Question Answering":

        try:

            search_results = search_knowledge_base(
                query=user_question,
                embedding_model=st.session_state.embedding_model,
                knowledge_base=st.session_state.rag_index,
                top_k=4,
            )

            best_score = get_best_relevance_score(
                search_results
            )

            if not search_results:

                answer = (
                    "I couldn't find enough information in the "
                    "provided study material to answer this confidently. "
                    "Please upload a relevant document or provide more context."
                )

                with st.chat_message("assistant"):

                    st.warning(answer)

                st.session_state.pdf_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.session_state.pdf_last_question = (
                    user_question
                )

                st.session_state.pdf_last_answer = answer
                st.session_state.pdf_last_context = ""

                st.stop()


            # Build retrieved context
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


            # Save current PDF context
            st.session_state.pdf_last_context = (
                retrieved_context
            )


            st.caption(
                f"🔎 Retrieved {len(search_results)} relevant "
                f"sections | Best relevance: {best_score:.2f}"
            )

        except Exception as error:

            error_message = (
                f"❌ RAG retrieval failed: {error}"
            )

            with st.chat_message("assistant"):

                st.error(error_message)

            # Save error to correct mode
            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

            st.stop()


    # =====================================================
    # BUILD REQUEST DATA
    # =====================================================

    request_data = {

        "mode": (
            "pdf"
            if selected_mode
            == "📚 PDF Question Answering"
            else "tutor"
        ),

        "question": user_question,

        "topic": subject,

        "subject": subject,

        "academic_level": academic_level,

        "language": language,

        "explanation_style": explanation_style,

        "context": retrieved_context,

        "history": mode_messages[:-1],
    }


    # =====================================================
    # MULTI-AGENT AI PIPELINE
    # =====================================================

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "🧠 Tutor Agent is thinking..."
            ):

                result = run_ai_tutor(
                    request_data=request_data,
                    api_key=api_key,
                    model=model,
                )


            # -------------------------------------------------
            # RESULT HANDLING
            # -------------------------------------------------

            if isinstance(result, dict):

                answer = result.get(
                    "answer",
                    "",
                )

                if not answer:

                    answer = result.get(
                        "final_answer",
                        "",
                    )

                if not answer:

                    answer = (
                        "The AI did not return a usable answer."
                    )

            else:

                answer = str(result)


            st.markdown(answer)


        # =====================================================
        # SAVE ASSISTANT ANSWER TO CORRECT MEMORY
        # =====================================================

        assistant_message = {
            "role": "assistant",
            "content": answer,
        }

        if selected_mode == "📚 PDF Question Answering":

            st.session_state.pdf_messages.append(
                assistant_message
            )

            st.session_state.pdf_last_question = (
                user_question
            )

            st.session_state.pdf_last_answer = answer

        else:

            st.session_state.tutor_messages.append(
                assistant_message
            )

            st.session_state.tutor_last_question = (
                user_question
            )

            st.session_state.tutor_last_answer = answer


    except Exception as error:

        error_message = (
            f"❌ AI processing failed: {error}"
        )

        with st.chat_message("assistant"):

            st.error(error_message)


        # Save error to correct conversation
        if selected_mode == "📚 PDF Question Answering":

            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

        else:

            st.session_state.tutor_messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


# =========================================================
# REFRESH CURRENT MEMORY FOR LEARNING TOOLS
# =========================================================

(
    current_messages,
    current_last_question,
    current_last_answer,
    current_last_context,
) = get_current_memory(selected_mode)


# =========================================================
# LEARNING TOOLS
# =========================================================

if current_last_question and current_last_answer:

    st.divider()

    st.subheader("🧠 Learning Tools")

    st.caption(
        "Use these tools with your most recent question and answer."
    )

    tool_columns = st.columns(3)


    # -----------------------------------------------------
    # EXPLAIN AGAIN
    # -----------------------------------------------------

    with tool_columns[0]:

        if st.button(
            "🔄 Explain Again",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": explanation_style,
            }


            try:

                with st.spinner(
                    "🔄 Explaining again..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Explain Again",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


    # -----------------------------------------------------
    # EXPLAIN SIMPLY
    # -----------------------------------------------------

    with tool_columns[1]:

        if st.button(
            "🧒 Explain Simply",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": "Simple",
            }


            try:

                with st.spinner(
                    "🧒 Making explanation simpler..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Explain Simply",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


    # -----------------------------------------------------
    # GIVE EXAMPLE
    # -----------------------------------------------------

    with tool_columns[2]:

        if st.button(
            "💡 Give Example",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": explanation_style,
            }


            try:

                with st.spinner(
                    "💡 Creating an example..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Give Example",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


    # =====================================================
    # SECOND ROW OF LEARNING TOOLS
    # =====================================================

    tool_columns_2 = st.columns(3)


    # -----------------------------------------------------
    # EXAM ANSWER
    # -----------------------------------------------------

    with tool_columns_2[0]:

        if st.button(
            "📝 Exam Answer",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": "Exam Focused",
            }


            try:

                with st.spinner(
                    "📝 Creating exam-style answer..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Exam Answer",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    with tool_columns_2[1]:

        if st.button(
            "📋 Create Summary",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": "Simple",
            }


            try:

                with st.spinner(
                    "📋 Creating summary..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Summary",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


    # -----------------------------------------------------
    # QUIZ
    # -----------------------------------------------------

    with tool_columns_2[2]:

        if st.button(
            "❓ Generate Quiz",
            use_container_width=True,
        ):

            tool_request = {

                "mode": (
                    "pdf"
                    if selected_mode
                    == "📚 PDF Question Answering"
                    else "tutor"
                ),

                "question": current_last_question,

                "answer": current_last_answer,

                "context": current_last_context,

                "history": current_messages,

                "academic_level": academic_level,

                "subject": subject,

                "language": language,

                "explanation_style": "Exam Focused",
            }


            try:

                with st.spinner(
                    "❓ Generating quiz..."
                ):

                    tool_answer = run_learning_tool(
                        tool="Quiz",
                        request_data=tool_request,
                        api_key=api_key,
                        model=model,
                    )


                with st.chat_message("assistant"):

                    st.markdown(tool_answer)


                if selected_mode == "📚 PDF Question Answering":

                    st.session_state.pdf_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                else:

                    st.session_state.tutor_messages.append(
                        {
                            "role": "assistant",
                            "content": tool_answer,
                        }
                    )

                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Learning tool failed: {error}"
                )


# =========================================================
# MEMORY STATUS
# =========================================================

st.divider()

with st.expander("🧠 Conversation Memory Status"):

    pdf_count = len(
        st.session_state.pdf_messages
    )

    tutor_count = len(
        st.session_state.tutor_messages
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "📚 PDF Conversation Messages",
            pdf_count,
        )

    with col2:

        st.metric(
            "🤖 AI Tutor Messages",
            tutor_count,
        )

    st.caption(
        "These two memories are completely separate. "
        "Switching modes does not transfer conversation history."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 AI Education / AI Tutor | "
    "RAG + Multi-Agent AI + Groq"
)
