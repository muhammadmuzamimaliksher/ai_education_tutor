import streamlit as st

from config import APP_TITLE, GROQ_API_KEY, DEFAULT_MODEL, AVAILABLE_MODELS
from ai_engine import run_ai_tutor


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎓 AI Education / AI Tutor")
st.write(
    "An AI-powered learning assistant designed to explain academic "
    "concepts according to the student's educational level."
)


# ---------------------------------------------------------
# API Key Check
# ---------------------------------------------------------
if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY is not configured.\n\n"
        "Add GROQ_API_KEY to Streamlit Secrets and reboot the app."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
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
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL)
        if DEFAULT_MODEL in AVAILABLE_MODELS
        else 0,
    )

    st.divider()

    st.info(
        "Future versions can connect this application to a "
        "document-based RAG knowledge base."
    )


# ---------------------------------------------------------
# Main Question Area
# ---------------------------------------------------------
st.subheader("Ask Your AI Tutor")

question = st.text_area(
    "✏️ Enter your question",
    placeholder=(
        "Example: Explain photosynthesis step by step "
        "in simple words."
    ),
    height=150,
)


# ---------------------------------------------------------
# Ask Button
# ---------------------------------------------------------
if st.button("🚀 Ask AI Tutor", type="primary", use_container_width=True):

    if not question.strip():
        st.warning("⚠️ Please enter a question first.")
        st.stop()

    request_data = {
        "question": question.strip(),
        "academic_level": academic_level,
        "subject": subject,
        "language": language,
        "explanation_style": explanation_style,
    }

    with st.spinner("🤔 AI Tutor is thinking..."):

        try:
            answer = run_ai_tutor(
                request_data=request_data,
                api_key=GROQ_API_KEY,
                model=model,
            )

            if answer:
                st.subheader("📖 AI Tutor Answer")
                st.markdown(answer)

            else:
                st.error(
                    "❌ The AI returned an empty response. "
                    "Please try again."
                )

        except Exception as error:
            st.error(
                "❌ Unable to generate the answer."
            )

            with st.expander("Technical error"):
                st.code(str(error))


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "AI Education / AI Tutor • RAG + Multi-Agent Architecture"
)
