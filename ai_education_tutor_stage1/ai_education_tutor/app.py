import streamlit as st
from groq import Groq

from config import APP_NAME, GROQ_API_KEY, GROQ_MODEL


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def ask_groq(
    client,
    question: str,
    academic_level: str,
    subject: str,
    language: str,
    explanation_style: str,
) -> str:
    system_prompt = f"""
You are an AI Education Tutor.

Student academic level: {academic_level}
Subject: {subject}
Preferred language: {language}
Explanation style: {explanation_style}

Your job is to teach, not merely give a short answer.
Adapt vocabulary, depth, examples, and reasoning to the student's academic level.
For mathematics and technical problems, show the reasoning step by step.
Use examples when they improve understanding.
Do not intentionally invent facts. If information is uncertain, clearly say so.
For advanced university or PhD questions, provide appropriate technical depth.
Do not claim 100% accuracy or certainty.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_text = str(e)

        if "401" in error_text or "invalid_api_key" in error_text.lower():
            return "❌ Invalid Groq API key. Check GROQ_API_KEY in Streamlit Secrets."

        if "429" in error_text:
            return "⚠️ Groq rate limit reached. Please wait a moment and try again."

        if "404" in error_text or "model" in error_text.lower():
            return (
                f"❌ Groq model error. Current model is `{GROQ_MODEL}`. "
                "Check that this model is available to your Groq account."
            )

        return f"❌ An error occurred while generating the answer: {error_text}"


st.title("🎓 AI Education / AI Tutor")
st.caption("Ask. Understand. Practice. Learn.")

with st.sidebar:
    st.header("Learning Settings")

    academic_level = st.selectbox(
        "Academic Level",
        [
            "Grade 1–5",
            "Grade 6–8",
            "Grade 9–10",
            "Grade 11–12",
            "Undergraduate",
            "Master's",
            "PhD",
        ],
    )

    subject = st.selectbox(
        "Subject",
        [
            "General",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Computer Science",
            "English",
            "History",
            "Geography",
            "Business",
            "Engineering",
            "Other",
        ],
    )

    language = st.selectbox(
        "Language",
        ["English", "Urdu", "Roman Urdu"],
    )

    explanation_style = st.selectbox(
        "Explanation Style",
        [
            "Simple",
            "Step-by-step",
            "Detailed",
            "Academic",
            "Exam-focused",
        ],
    )

    st.divider()
    st.caption(f"AI Model: {GROQ_MODEL}")

if not GROQ_API_KEY:
    st.warning(
        "⚠️ GROQ_API_KEY is not configured. Add it to "
        "`.streamlit/secrets.toml` for local testing or Streamlit Secrets for deployment."
    )

question = st.text_area(
    "Ask your question",
    placeholder="Example: Explain photosynthesis in simple words.",
    height=160,
)

if st.button("🚀 Ask AI Tutor", type="primary", use_container_width=True):
    if not GROQ_API_KEY:
        st.error("Please configure GROQ_API_KEY first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        client = get_groq_client()

        with st.spinner("AI Tutor is thinking..."):
            answer = ask_groq(
                client=client,
                question=question.strip(),
                academic_level=academic_level,
                subject=subject,
                language=language,
                explanation_style=explanation_style,
            )

        st.session_state["last_question"] = question.strip()
        st.session_state["last_answer"] = answer

if "last_answer" in st.session_state:
    st.divider()
    st.subheader("📚 Tutor Answer")
    st.markdown(st.session_state["last_answer"])

st.divider()
st.caption(
    "Stage 1 foundation: Streamlit → Groq → AI Tutor. "
    "CrewAI, RAG, verification, memory, tools, and quizzes will be added in later stages."
)
