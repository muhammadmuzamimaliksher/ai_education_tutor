# =========================================================
# SESSION STATE MANAGER
# =========================================================

import streamlit as st


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

def initialize_session_state():
    """
    Initialize all Streamlit session-state variables.
    """

    # =====================================================
    # PDF QUIZ SESSION
    # =====================================================

    if "pdf_quiz" not in st.session_state:
        st.session_state.pdf_quiz = {
            "history": [],
            "active_quiz": None,
            "answers": {},
            "score": 0,
            "submitted": False,
            "source_mode": "PDF Question Answering",
        }

    # =====================================================
    # AI TUTOR QUIZ SESSION
    # =====================================================

    if "tutor_quiz" not in st.session_state:
        st.session_state.tutor_quiz = {
            "history": [],
            "active_quiz": None,
            "answers": {},
            "score": 0,
            "submitted": False,
            "source_mode": "AI Tutor",
        }

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
# CONVERSATION CLEAR FUNCTIONS
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

    AI Tutor memory and AI Tutor quiz history
    are NOT affected.
    """

    clear_pdf_conversation()


# =========================================================
# CURRENT CONVERSATION MEMORY
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


# =========================================================
# QUIZ STATE FUNCTIONS
# =========================================================

def get_quiz_state(selected_mode):

    """
    Return the quiz state belonging ONLY to the current mode.

    PDF mode  -> pdf_quiz
    Tutor mode -> tutor_quiz
    """

    if selected_mode == "📚 PDF Question Answering":

        return st.session_state.pdf_quiz

    return st.session_state.tutor_quiz


# =========================================================
# CLEAR CURRENT MODE QUIZ
# =========================================================

def clear_quiz_session(selected_mode):

    """
    Clear ONLY the quiz belonging to the selected mode.

    This does NOT affect the other mode's quiz.
    """

    quiz = get_quiz_state(selected_mode)

    quiz["history"] = []

    quiz["active_quiz"] = None

    quiz["answers"] = {}

    quiz["score"] = 0

    quiz["submitted"] = False


# =========================================================
# RESET ACTIVE QUIZ ONLY
# =========================================================

def reset_active_quiz(selected_mode):

    """
    Remove only the currently active quiz.

    Quiz history remains محفوظ.
    """

    quiz = get_quiz_state(selected_mode)

    quiz["active_quiz"] = None

    quiz["answers"] = {}

    quiz["score"] = 0

    quiz["submitted"] = False


# =========================================================
# GET CURRENT QUIZ MEMORY
# =========================================================

def get_quiz_memory(selected_mode):

    """
    Return quiz information for the current mode only.
    """

    quiz = get_quiz_state(selected_mode)

    return {
        "messages": quiz["history"],
        "history": quiz["history"],
        "active_quiz": quiz["active_quiz"],
        "answers": quiz["answers"],
        "score": quiz["score"],
        "submitted": quiz["submitted"],
        "source_mode": quiz["source_mode"],
    }
