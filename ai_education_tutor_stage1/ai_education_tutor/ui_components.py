# =========================================================
# UI COMPONENTS
# =========================================================

import streamlit as st

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    get_groq_api_key,
)

from session_manager import clear_current_conversation


def render_sidebar():

    with st.sidebar:

        st.header("⚙️ Settings")


        # =================================================
        # MODE
        # =================================================

        selected_mode = st.radio(
            "Choose Learning Mode",
            [
                "📚 PDF Question Answering",
                "🤖 AI Tutor",
            ],
        )


        st.divider()


        # =================================================
        # MODEL
        # =================================================

        model = st.selectbox(
            "🤖 AI Model",
            AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(DEFAULT_MODEL)
                if DEFAULT_MODEL in AVAILABLE_MODELS
                else 0
            ),
        )


        # =================================================
        # EDUCATION LEVEL
        # =================================================

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


        # =================================================
        # SUBJECT
        # =================================================

        subject = st.text_input(
            "📖 Subject",
            placeholder="e.g. Biology, Physics, SEO",
        )


        # =================================================
        # LANGUAGE
        # =================================================

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


        # =================================================
        # STYLE
        # =================================================

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


        # =================================================
        # CLEAR MEMORY
        # =================================================

        button_text = (
            "🗑️ Clear PDF Conversation"
            if selected_mode
            == "📚 PDF Question Answering"
            else
            "🗑️ Clear Tutor Conversation"
        )


        if st.button(
            button_text,
            use_container_width=True,
        ):

            clear_current_conversation(
                selected_mode
            )

            st.success(
                "Current conversation cleared."
            )

            st.rerun()


        st.divider()


        # =================================================
        # API STATUS
        # =================================================

        api_key = get_groq_api_key()


        if api_key:

            st.success(
                "✅ Groq API configured"
            )

        else:

            st.error(
                "❌ GROQ_API_KEY is not configured."
            )


    # =====================================================
    # SETTINGS OBJECT
    # =====================================================

    settings = {

        "model": model,

        "academic_level": academic_level,

        "subject": subject,

        "language": language,

        "explanation_style": explanation_style,

        "api_key": api_key,
    }


    return selected_mode, settings
