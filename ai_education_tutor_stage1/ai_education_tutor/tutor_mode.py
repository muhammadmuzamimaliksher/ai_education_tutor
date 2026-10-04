# =========================================================
# AI TUTOR MODE
# =========================================================

import streamlit as st

from ai_engine import run_ai_tutor


def render_tutor_mode(settings):

    st.header("🤖 AI Tutor")

    st.info(
        "Ask general educational questions. "
        "No PDF is required."
    )


    st.markdown(
        """
        **AI Tutor Workflow**

        Question
        → Tutor Agent
        → Research Agent
        → Evaluator Agent
        → Final Answer
        """
    )


    st.divider()


    # =====================================================
    # DISPLAY TUTOR MEMORY ONLY
    # =====================================================

    for message in st.session_state.tutor_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask your AI Tutor a question..."
    )


    if question:

        process_tutor_question(
            question,
            settings,
        )


# =========================================================
# PROCESS AI TUTOR QUESTION
# =========================================================

def process_tutor_question(
    question,
    settings,
):

    if not settings["api_key"]:

        st.error(
            "GROQ_API_KEY is not configured."
        )

        return


    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.tutor_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # =====================================================
    # REQUEST DATA
    # =====================================================

    request_data = {

        "mode": "tutor",

        "question": question,

        "topic": settings["subject"],

        "subject": settings["subject"],

        "academic_level": settings[
            "academic_level"
        ],

        "language": settings["language"],

        "explanation_style": settings[
            "explanation_style"
        ],

        "context": "",

        "history": (
            st.session_state.tutor_messages[:-1]
        ),
    }


    # =====================================================
    # AI PIPELINE
    # =====================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Tutor → Research → Evaluator..."
        ):

            try:

                result = run_ai_tutor(
                    request_data=request_data,
                    api_key=settings["api_key"],
                    model=settings["model"],
                )


                if isinstance(result, dict):

                    answer = result.get(
                        "answer",
                        result.get(
                            "final_answer",
                            "",
                        ),
                    )

                else:

                    answer = str(result)


                if not answer:

                    answer = (
                        "The AI did not return a usable answer."
                    )


                st.markdown(answer)


            except Exception as error:

                answer = (
                    f"❌ AI processing failed: {error}"
                )

                st.error(answer)


    # =====================================================
    # SAVE ANSWER
    # =====================================================

    st.session_state.tutor_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    st.session_state.tutor_last_question = (
        question
    )

    st.session_state.tutor_last_answer = (
        answer
    )


    st.rerun()
