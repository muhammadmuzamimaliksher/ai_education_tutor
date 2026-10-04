def format_history(
    history,
    max_messages=8,
):
    """
    Convert conversation history into compact text.

    Conversation history is context only.
    In PDF mode it is NOT treated as factual evidence.
    """

    if not history:
        return "No previous conversation."

    recent_history = history[-max_messages:]

    formatted = []

    for message in recent_history:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":
            label = "Student"
        else:
            label = "AI Tutor"

        formatted.append(
            f"{label}: {content}"
        )

    return "\n\n".join(formatted)


def validate_pdf_context(
    retrieved_context,
):
    """
    Check whether retrieved PDF evidence exists.
    """

    if not retrieved_context:
        return False

    if not retrieved_context.strip():
        return False

    return True
