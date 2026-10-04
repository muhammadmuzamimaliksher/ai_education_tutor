# learning_tools.py

# =========================================================
# LEARNING TOOLS
# =========================================================

import json
import re

import streamlit as st

from ai_engine import run_learning_tool
from session_manager import get_current_memory


# =========================================================
# QUIZ HELPERS
# =========================================================

def extract_json_from_response(text):
    """
    Extract JSON from an AI response.

    Handles:
    - normal JSON
    - ```json ... ```
    - ``` ... ```
    """

    if not text:
        raise ValueError("Empty quiz response received.")

    cleaned = text.strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    # First attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        candidate = cleaned[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The AI returned an invalid quiz format."
    )


def validate_quiz(quiz_data):

    if not isinstance(quiz_data, dict):
        raise ValueError(
            "Quiz data must be an object."
        )

    questions = quiz_data.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Quiz questions are missing."
        )

    if len(questions) != 5:
        raise ValueError(
            "Quiz must contain exactly 5 questions."
        )

    for index, question in enumerate(
        questions,
        start=1,
    ):

        if not isinstance(question, dict):
            raise ValueError(
                f"Question {index} is invalid."
            )

        if not question.get("question"):
            raise ValueError(
                f"Question {index} has no question text."
            )

        options = question.get("options")

        if not isinstance(options, list):
            raise ValueError(
                f"Question {index} has no options."
            )

        if len(options) != 4:
            raise ValueError(
                f"Question {index} must have 4 options."
            )

        correct_index = question.get(
            "correct_index"
        )

        if not isinstance(
            correct_index,
            int,
        ):
            raise ValueError(
                f"Question {index} has no valid answer."
            )

        if correct_index < 0 or correct_index > 3:
            raise ValueError(
                f"Question {index} has an invalid answer index."
            )

        # Explanation is optional.
        # The quiz must NOT fail just because the AI omitted it.
        if not question.get("explanation"):
            question["explanation"] = ""

    return True

def initialize_quiz_state():

    if "active_quiz" not in st.session_state:
        st.session_state.active_quiz = None

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0


def clear_quiz():

    st.session_state.active_quiz = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}
    st.session_state.quiz_score = 0


def render_quiz():

    quiz = st.session_state.active_quiz

    if not quiz:
        return

    questions = quiz.get(
        "questions",
        [],
    )

    st.divider()

    st.subheader(
        "📝 Quiz"
    )

    if not st.session_state.quiz_submitted:

        st.info(
            "Select one answer for each question, "
            "then click **Submit Quiz**."
        )

        for index, question in enumerate(
            questions,
            start=1,
        ):

            st.markdown(
                f"### Question {index} of {len(questions)}"
            )

            st.write(
                question["question"]
            )

            options = question["options"]

            selected = st.radio(
                "Choose your answer:",
                options,
                key=f"quiz_answer_{index}",
                index=None,
            )

            if selected is not None:
                st.session_state.quiz_answers[
                    index - 1
                ] = selected

            st.divider()

        if st.button(
            "✅ Submit Quiz",
            type="primary",
            use_container_width=True,
        ):

            if len(
                st.session_state.quiz_answers
            ) != len(questions):

                st.warning(
                    "Please answer all questions before "
                    "submitting the quiz."
                )

                return

            score = 0

            for index, question in enumerate(
                questions
            ):

                selected = (
                    st.session_state.quiz_answers[
                        index
                    ]
                )

                correct_index = question[
                    "correct_index"
                ]

                correct_answer = question[
                    "options"
                ][correct_index]

                if selected == correct_answer:
                    score += 1

            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True

            st.rerun()

    else:

        score = st.session_state.quiz_score
        total = len(questions)

        percentage = round(
            (score / total) * 100
        )

        st.success(
            f"🎯 Quiz Completed!\n\n"
            f"**Score: {score}/{total} "
            f"({percentage}%)**"
        )

        if percentage >= 80:
            st.balloons()
            st.success(
                "Excellent work! 🎉"
            )
        elif percentage >= 60:
            st.info(
                "Good effort! Review the incorrect answers."
            )
        else:
            st.warning(
                "Keep practicing and review the PDF material."
            )

        st.divider()

        st.subheader(
            "📊 Answer Review"
        )

        for index, question in enumerate(
            questions
        ):

            selected = (
                st.session_state.quiz_answers[
                    index
                ]
            )

            correct_answer = question[
                "options"
            ][
                question["correct_index"]
            ]

            if selected == correct_answer:

                st.success(
                    f"**Question {index + 1}: ✅ Correct**"
                )

            else:

                st.error(
                    f"**Question {index + 1}: ❌ Incorrect**"
                )

                st.write(
                    f"Your answer: **{selected}**"
                )

                st.write(
                    f"Correct answer: **{correct_answer}**"
                )

            st.write(
                f"**Explanation:** "
                f"{question['explanation']}"
            )

            st.divider()

        if st.button(
            "🔄 Create New Quiz",
            use_container_width=True,
        ):

            clear_quiz()
            st.rerun()


# =========================================================
# MAIN LEARNING TOOLS
# =========================================================

def render_learning_tools(
    selected_mode,
    settings,
):

    initialize_quiz_state()

    memory = get_current_memory(
        selected_mode
    )

    last_question = memory[
        "last_question"
    ]

    last_answer = memory[
        "last_answer"
    ]

    last_context = memory[
        "last_context"
    ]

    current_messages = memory[
        "messages"
    ]

    # =====================================================
    # SHOW ACTIVE QUIZ
    # =====================================================

    if st.session_state.active_quiz:

        render_quiz()

    # =====================================================
    # NOTHING TO PROCESS
    # =====================================================

    if not last_question or not last_answer:
        return

    st.divider()

    st.subheader(
        "🧠 Learning Tools"
    )

    st.caption(
        "Use these tools with your latest question."
    )

    # =====================================================
    # CORRECT MODE NAME
    # =====================================================

    if selected_mode == "📚 PDF Question Answering":

        mode = "PDF Question Answering"

    else:

        mode = "AI Tutor"

    # =====================================================
    # TOOL REQUEST
    # =====================================================

    def execute_tool(
        tool_name,
        explanation_style=None,
    ):

        # -------------------------------------------------
        # PDF EXAM ANSWER
        # -------------------------------------------------
        #
        # IMPORTANT:
        # Do NOT send this back to the AI.
        #
        # last_answer is already the exact answer extracted
        # from the PDF by PDF mode.
        #

        if (
            tool_name == "Exam Answer"
            and mode == "PDF Question Answering"
        ):

            result = last_answer

            with st.chat_message(
                "assistant"
            ):

                st.markdown(result)

            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

            st.rerun()

            return

        request_data = {

            "mode": mode,

            "question": last_question,

            "answer": last_answer,

            # IMPORTANT:
            # Engine expects retrieved_context.
            "retrieved_context": last_context,

            # Keep context key too for compatibility.
            "context": last_context,

            "history": current_messages,

            "academic_level": settings[
                "academic_level"
            ],

            "subject": settings[
                "subject"
            ],

            "language": settings[
                "language"
            ],

            "explanation_style": (
                explanation_style
                or settings[
                    "explanation_style"
                ]
            ),
        }

        try:

            with st.spinner(
                f"🧠 {tool_name}..."
            ):

                result = run_learning_tool(
                    tool=tool_name,
                    request_data=request_data,
                    api_key=settings["api_key"],
                    model=settings["model"],
                )

            # =================================================
            # QUIZ
            # =================================================

            if tool_name == "Quiz":

                quiz_data = extract_json_from_response(
                    result
                )

                validate_quiz(
                    quiz_data
                )

                clear_quiz()

                st.session_state.active_quiz = (
                    quiz_data
                )

                st.session_state.quiz_submitted = (
                    False
                )

                st.rerun()

                return

            # =================================================
            # NORMAL LEARNING TOOL
            # =================================================

            with st.chat_message(
                "assistant"
            ):

                st.markdown(result)

            message = {
                "role": "assistant",
                "content": result,
            }

            if mode == "PDF Question Answering":

                st.session_state.pdf_messages.append(
                    message
                )

            else:

                st.session_state.tutor_messages.append(
                    message
                )

            st.rerun()

        except Exception as error:

            st.error(
                f"❌ Learning tool failed: {error}"
            )

    # =====================================================
    # FIRST ROW
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🔄 Explain Again",
            use_container_width=True,
        ):

            execute_tool(
                "Explain Again"
            )

    with col2:

        if st.button(
            "🧒 Explain Simply",
            use_container_width=True,
        ):

            execute_tool(
                "Explain Simply",
                "Simple",
            )

    with col3:

        if st.button(
            "💡 Give Example",
            use_container_width=True,
        ):

            execute_tool(
                "Give Example"
            )

    # =====================================================
    # SECOND ROW
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📝 Exam Answer",
            use_container_width=True,
        ):

            execute_tool(
                "Exam Answer",
                "Exam Focused",
            )

    with col2:

        if st.button(
            "📋 Create Summary",
            use_container_width=True,
        ):

            execute_tool(
                "Summary",
                "Simple",
            )

    with col3:

        if st.button(
            "❓ Generate Quiz",
            use_container_width=True,
        ):

            execute_tool(
                "Quiz",
                "Exam Focused",
            )
