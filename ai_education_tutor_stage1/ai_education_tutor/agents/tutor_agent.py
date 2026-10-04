# =========================================================
# TUTOR AGENT
# =========================================================

from ai_engine import create_client, call_groq


def run_tutor_agent(
    question,
    plan,
    context="",
    api_key=None,
    model=None,
):
    """
    Tutor Agent

    Responsibilities:
    - Teach the student
    - Follow the planner's instructions
    - Use provided context
    - Produce an educational draft
    """

    if not question:
        raise ValueError("Tutor Agent received an empty question.")

    client = create_client(api_key)

    system_prompt = """
You are the Tutor Agent in an AI Education system.

Your job is to teach the student clearly and accurately.

Rules:
- Follow the learning plan.
- Explain concepts step by step.
- Use simple language where possible.
- Use examples when helpful.
- Do not invent facts.
- If source context is provided, prioritize it.
- Clearly state when the available context is insufficient.
- Do not mention internal agents or internal processing.
"""

    user_prompt = f"""
STUDENT QUESTION:
{question}

LEARNING PLAN:
{plan}

AVAILABLE CONTEXT:
{context}

Create the best educational answer draft.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
