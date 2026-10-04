# learning_tools.py

# =========================================================
# LEARNING TOOLS
# =========================================================

import json
import re
from datetime import datetime

import streamlit as st

from ai_engine import run_learning_tool

from session_manager import (
    get_current_memory,
    get_quiz_state,
    clear_quiz_session,
)


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
        raise ValueError(
            "Empty quiz response received."
        )

    cleaned = text.strip()

    # -----------------------------------------------------
    # Remove markdown code fences
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # First attempt
    # -----------------------------------------------------

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:

        pass

    # -----------------------------------------------------
    # Find JSON object inside response
    # -----------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        try:

            return json.loads(candidate)

        except json.JSONDecodeError:

            pass

    raise ValueError(
        "The AI returned an invalid quiz format."
    )


# =========================================================
# QUIZ VALIDATION
# =========================================================

def validate_quiz(quiz_data):

    if not isinstance(
        quiz_data,
        dict,
    ):

        raise ValueError(
            "Quiz data must be an object."
        )

    questions = quiz_data.get(
        "questions"
    )

    if not isinstance(
        questions,
        list,
    ):

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

        if not isinstance(
            question,
            dict,
        ):

            raise ValueError(
                f"Question {index} is invalid."
            )

        # -------------------------------------------------
        # Question text
        # -------------------------------------------------

        if not question.get(
            "question"
        ):

            raise ValueError(
                f"Question {index} has no question text."
            )

        # -------------------------------------------------
        # Options
        # -------------------------------------------------

        options = question.get(
            "options"
        )

        if not isinstance(
            options,
            list,
        ):

            raise ValueError(
                f"Question {index} has no options."
            )

        if len(options) != 4:

            raise ValueError(
                f"Question {index} must have 4 options."
            )

        # -------------------------------------------------
        # Correct answer
        # -------------------------------------------------

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

        if (
            correct_index < 0
            or correct_index > 3
        ):

            raise ValueError(
                f"Question {index} has an invalid answer index."
            )

        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------
        #
        # Explanation is optional.
        # Do NOT fail the entire quiz if the AI omitted it.
        #

        if not question.get(
            "explanation"
        ):

            question["explanation"] = ""

    return True


# =========================================================
# QUIZ HISTORY
# =========================================================

def save_quiz_to_history(
    selected_mode,
    quiz_data,
):
    """
    Save a newly generated quiz into the history
    belonging ONLY to the current mode.

    PDF quiz -> pdf_quiz["history"]
    Tutor quiz -> tutor_quiz["history"]
    """

    quiz_state = get_quiz_state(
        selected_mode
    )

    quiz_record = {
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "quiz": quiz_data,
        "score": None,
        "total": len(
            quiz_data.get(
                "questions",
                [],
            )
        ),
        "submitted": False,
    }

    quiz_state["history"].append(
        quiz_record
    )


# =========================================================
# UPDATE LAST QUIZ HISTORY RECORD
# =========================================================

def update_latest_quiz_history(
    selected_mode,
):
    """
    Save the final score of the currently active
    quiz into the correct mode's quiz history.
    """

    quiz_state = get_quiz_state(
        selected_mode
    )

    history = quiz_state["history"]

    if not history:

        return

    latest = history[-1]

    latest["score"] = quiz_state[
        "score"
    ]

    latest["total"] = len(
        quiz_state[
            "active_quiz"
        ].get(
            "questions",
            [],
        )
    )

    latest["submitted"] = True

    latest["answers"] = dict(
        quiz_state["answers"]
    )


# =========================================================
# CLEAR ACTIVE QUIZ
# =========================================================

def clear_active_quiz(
    selected_mode,
):
    """
    Clear only the active quiz.

    Quiz history remains untouched.
    """

    quiz_state = get_quiz_state(
        selected_mode
    )

    quiz_state["active_quiz"] = None

    quiz_state["answers"] = {}

    quiz_state["score"] = 0

    quiz_state["submitted"] = False


# =========================================================
# RENDER QUIZ HISTORY
# =========================================================

def render_quiz_history(
    selected_mode,
):
    """
    Display quiz history ONLY for the selected mode.
    """

    quiz_state = get_quiz_state(
        selected_mode
    )

    history = quiz_state[
        "history"
    ]

    if not history:

        return

    st.divider()

    if (
        selected_mode
        == "📚 PDF Question Answering"
    ):

        st.subheader(
            "📚 PDF Quiz History"
        )

    else:

        st.subheader(
            "🤖 AI Tutor Quiz History"
        )

    for index, record in enumerate(
        reversed(history),
        start=1,
    ):

        score = record.get(
            "score"
        )

        total = record.get(
            "total",
            5,
        )

        created_at = record.get(
            "created_at",
            "",
        )

        if score is None:

            score_text = (
                "Not submitted"
            )

        else:

            percentage = round(
                (score / total) * 100
            )

            score_text = (
                f"{score}/{total} "
                f"({percentage}%)"
            )

        with st.expander(
            f"Quiz {len(history) - index + 1} "
            f"— {score_text}"
        ):

            st.caption(
                f"Created: {created_at}"
            )


# =========================================================
# RENDER ACTIVE QUIZ
# =========================================================

def render_quiz(
    selected_mode,
):

    quiz_state = get_quiz_state(
        selected_mode
    )

    quiz = quiz_state[
        "active_quiz"
    ]

    if not quiz:

        return

    questions = quiz.get(
        "questions",
        [],
    )

    if not questions:

        return

    st.divider()

    if (
        selected_mode
        == "📚 PDF Question Answering"
    ):

        st.subheader(
            "📚 PDF Quiz"
        )

        st.caption(
            "This quiz belongs only to "
            "PDF Question Answering mode."
        )

    else:

        st.subheader(
            "🤖 AI Tutor Quiz"
        )

        st.caption(
            "This quiz belongs only to "
            "AI Tutor mode."
        )

    # =====================================================
    # QUIZ NOT SUBMITTED
    # =====================================================

    if not quiz_state[
        "submitted"
    ]:

        st.info(
            "Select one answer for each question, "
            "then click **Submit Quiz**."
        )

        for index, question in enumerate(
            questions,
            start=1,
        ):

            st.markdown(
                f"### Question {index} "
                f"of {len(questions)}"
            )

            st.write(
                question["question"]
            )

            options = question[
                "options"
            ]

            selected = st.radio(
                "Choose your answer:",
                options,
                key=(
                    f"{selected_mode}_"
                    f"quiz_answer_{index}"
                ),
                index=None,
            )

            if selected is not None:

                quiz_state[
                    "answers"
                ][
                    index - 1
                ] = selected

            st.divider()

        # -------------------------------------------------
        # SUBMIT
        # -------------------------------------------------

        if st.button(
            "✅ Submit Quiz",
            type="primary",
            use_container_width=True,
            key=(
                f"{selected_mode}_"
                "submit_quiz"
            ),
        ):

            if len(
                quiz_state["answers"]
            ) != len(questions):

                st.warning(
                    "Please answer all questions "
                    "before submitting the quiz."
                )

                return

            score = 0

            for index, question in enumerate(
                questions
            ):

                selected = (
                    quiz_state[
                        "answers"
                    ][index]
                )

                correct_index = (
                    question[
                        "correct_index"
                    ]
                )

                correct_answer = (
                    question[
                        "options"
                    ][correct_index]
                )

                if (
                    selected
                    == correct_answer
                ):

                    score += 1

            quiz_state[
                "score"
            ] = score

            quiz_state[
                "submitted"
            ] = True

            # Save result to correct mode history
            update_latest_quiz_history(
                selected_mode
            )

            st.rerun()

    # =====================================================
    # QUIZ SUBMITTED
    # =====================================================

    else:

        score = quiz_state[
            "score"
        ]

        total = len(
            questions
        )

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
                "Good effort! Review the "
                "incorrect answers."
            )

        else:

            if (
                selected_mode
                == "📚 PDF Question Answering"
            ):

                st.warning(
                    "Keep practicing and review "
                    "the PDF material."
                )

            else:

                st.warning(
                    "Keep practicing and review "
                    "the topic again."
                )

        st.divider()

        st.subheader(
            "📊 Answer Review"
        )

        for index, question in enumerate(
            questions
        ):

            selected = (
                quiz_state[
                    "answers"
                ][index]
            )

            correct_answer = (
                question[
                    "options"
                ][
                    question[
                        "correct_index"
                    ]
                ]
            )

            if (
                selected
                == correct_answer
            ):

                st.success(
                    f"**Question {index + 1}: "
                    f"✅ Correct**"
                )

            else:

                st.error(
                    f"**Question {index + 1}: "
                    f"❌ Incorrect**"
                )

                st.write(
                    f"Your answer: "
                    f"**{selected}**"
                )

                st.write(
                    f"Correct answer: "
                    f"**{correct_answer}**"
                )

            explanation = (
                question.get(
                    "explanation",
                    "",
                )
                or ""
            ).strip()

            if explanation:

                st.write(
                    f"**Explanation:** "
                    f"{explanation}"
                )

            st.divider()

        # -------------------------------------------------
        # NEW QUIZ
        # -------------------------------------------------

        if st.button(
            "🔄 Create New Quiz",
            use_container_width=True,
            key=(
                f"{selected_mode}_"
                "new_quiz"
            ),
        ):

            clear_active_quiz(
                selected_mode
            )

            st.rerun()


# =========================================================
# MAIN LEARNING TOOLS
# =========================================================

def render_learning_tools(
    selected_mode,
    settings,
):

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

    quiz_state = get_quiz_state(
        selected_mode
    )

    if quiz_state[
        "active_quiz"
    ]:

        render_quiz(
            selected_mode
        )

    # =====================================================
    # QUIZ HISTORY
    # =====================================================

    render_quiz_history(
        selected_mode
    )

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

    if (
        selected_mode
        == "📚 PDF Question Answering"
    ):

        mode = (
            "PDF Question Answering"
        )

    else:

        mode = "AI Tutor"

    # =====================================================
    # TOOL EXECUTION
    # =====================================================

    def execute_tool(
        tool_name,
        explanation_style=None,
    ):

        # -------------------------------------------------
        # PDF EXAM ANSWER
        # -------------------------------------------------

        if (
            tool_name
            == "Exam Answer"
            and mode
            == "PDF Question Answering"
        ):

            result = last_answer

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    result
                )

            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

            st.rerun()

            return

        # -------------------------------------------------
        # REQUEST DATA
        # -------------------------------------------------

        request_data = {

            "mode": mode,

            "question": last_question,

            "answer": last_answer,

            "retrieved_context": (
                last_context
            ),

            "context": last_context,

            "history": current_messages,

            "academic_level": (
                settings[
                    "academic_level"
                ]
            ),

            "subject": (
                settings[
                    "subject"
                ]
            ),

            "language": (
                settings[
                    "language"
                ]
            ),

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
                    api_key=settings[
                        "api_key"
                    ],
                    model=settings[
                        "model"
                    ],
                )

            # =================================================
            # QUIZ
            # =================================================

            if tool_name == "Quiz":

                quiz_data = (
                    extract_json_from_response(
                        result
                    )
                )

                validate_quiz(
                    quiz_data
                )

                # ---------------------------------------------
                # Get ONLY current mode quiz state
                # ---------------------------------------------

                quiz_state = get_quiz_state(
                    selected_mode
                )

                # ---------------------------------------------
                # Clear old ACTIVE quiz only.
                #
                # History is NOT deleted.
                # ---------------------------------------------

                clear_active_quiz(
                    selected_mode
                )

                # ---------------------------------------------
                # Store new active quiz
                # ---------------------------------------------

                quiz_state[
                    "active_quiz"
                ] = quiz_data

                quiz_state[
                    "answers"
                ] = {}

                quiz_state[
                    "score"
                ] = 0

                quiz_state[
                    "submitted"
                ] = False

                # ---------------------------------------------
                # Save to correct mode history
                # ---------------------------------------------

                save_quiz_to_history(
                    selected_mode,
                    quiz_data,
                )

                # ---------------------------------------------
                # IMPORTANT:
                #
                # DO NOT add Quiz to:
                # pdf_messages
                # tutor_messages
                # ---------------------------------------------

                st.rerun()

                return

            # =================================================
            # NORMAL LEARNING TOOL
            # =================================================

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    result
                )

            message = {
                "role": "assistant",
                "content": result,
            }

            # -------------------------------------------------
            # Save normal learning-tool response
            # to the correct conversation.
            #
            # Quiz is intentionally excluded above.
            # -------------------------------------------------

            if (
                mode
                == "PDF Question Answering"
            ):

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
