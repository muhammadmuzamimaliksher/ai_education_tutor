# =========================================================
# SESSION STATE MANAGER
# =========================================================

import streamlit as st


def initialize_session_state():
    """
    Initialize all Streamlit session-state variables.
    """
    # =====================================================
    # QUIZ SESSION
    # =====================================================
    
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []
    
    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    
    if "quiz_source_mode" not in st.session_state:
        st.session_state.quiz_source_mode = ""
    # =====================================================
    # RAG / PDF STATE
    # =====================================================

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


    # =====================================================
    # PDF CONVERSATION MEMORY
    # =====================================================

    if "pdf_messages" not in st.session_state:
        st.session_state.pdf_messages = []

    if "pdf_last_question" not in st.session_state:
        st.session_state.pdf_last_question = ""

    if "pdf_last_answer" not in st.session_state:
        st.session_state.pdf_last_answer = ""

    if "pdf_last_context" not in st.session_state:
        st.session_state.pdf_last_context = ""


    # =====================================================
    # AI TUTOR CONVERSATION MEMORY
    # =====================================================

    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []

    if "tutor_last_question" not in st.session_state:
        st.session_state.tutor_last_question = ""

    if "tutor_last_answer" not in st.session_state:
        st.session_state.tutor_last_answer = ""


# =========================================================
# MEMORY FUNCTIONS
# =========================================================

def clear_pdf_conversation():

    st.session_state.pdf_messages = []
    st.session_state.pdf_last_question = ""
    st.session_state.pdf_last_answer = ""
    st.session_state.pdf_last_context = ""


def clear_tutor_conversation():

    st.session_state.tutor_messages = []
    st.session_state.tutor_last_question = ""
    st.session_state.tutor_last_answer = ""


def clear_current_conversation(selected_mode):

    if selected_mode == "📚 PDF Question Answering":

        clear_pdf_conversation()

    else:

        clear_tutor_conversation()


# =========================================================
# NEW PDF RESET
# =========================================================

def reset_pdf_conversation():

    """
    Clear PDF conversation when a new PDF is uploaded.

    AI Tutor memory is NOT affected.
    """

    clear_pdf_conversation()


# =========================================================
# CURRENT MEMORY
# =========================================================

def get_current_memory(selected_mode):

    if selected_mode == "📚 PDF Question Answering":

        return {
            "messages": st.session_state.pdf_messages,
            "last_question": st.session_state.pdf_last_question,
            "last_answer": st.session_state.pdf_last_answer,
            "last_context": st.session_state.pdf_last_context,
        }

    return {
        "messages": st.session_state.tutor_messages,
        "last_question": st.session_state.tutor_last_question,
        "last_answer": st.session_state.tutor_last_answer,
        "last_context": "",
    }
