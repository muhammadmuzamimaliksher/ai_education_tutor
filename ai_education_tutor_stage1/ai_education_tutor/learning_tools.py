# =========================================================
# LEARNING TOOLS
# =========================================================

import streamlit as st

from ai_engine import run_learning_tool
from session_manager import get_current_memory


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
    # MODE
    # =====================================================

    mode = (
        "pdf"
        if selected_mode
        == "📚 PDF Question Answering"
        else
        "tutor"
    )


    # =====================================================
    # TOOL REQUEST
    # =====================================================

    def execute_tool(
        tool_name,
        explanation_style=None,
    ):

        request_data = {

            "mode": mode,

            "question": last_question,

            "answer": last_answer,

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


            with st.chat_message("assistant"):

                st.markdown(result)


            # ---------------------------------------------
            # SAVE TO CORRECT MEMORY
            # ---------------------------------------------

            message = {
                "role": "assistant",
                "content": result,
            }


            if mode == "pdf":

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
