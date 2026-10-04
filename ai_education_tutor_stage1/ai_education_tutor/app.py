# =========================================================
# AI EDUCATION / AI TUTOR
# MAIN APPLICATION
# =========================================================

import streamlit as st

from config import APP_TITLE
from session_manager import initialize_session_state
from ui_components import render_sidebar
from pdf_mode import render_pdf_mode
from tutor_mode import render_tutor_mode
from learning_tools import render_learning_tools


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

initialize_session_state()


# =========================================================
# HEADER
# =========================================================

st.title("🎓 AI Education / AI Tutor")

st.markdown(
    """
    **Learn smarter with AI-powered tutoring, PDF question
    answering, RAG-based knowledge retrieval, and multi-agent AI.**
    """
)


# =========================================================
# SIDEBAR
# =========================================================

selected_mode, settings = render_sidebar()


# =========================================================
# MAIN CONTENT
# =========================================================

if selected_mode == "📚 PDF Question Answering":

    render_pdf_mode(settings)

else:

    render_tutor_mode(settings)


# =========================================================
# LEARNING TOOLS
# =========================================================

render_learning_tools(
    selected_mode=selected_mode,
    settings=settings,
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 AI Education / AI Tutor | "
    "RAG + Multi-Agent AI + Groq"
)
