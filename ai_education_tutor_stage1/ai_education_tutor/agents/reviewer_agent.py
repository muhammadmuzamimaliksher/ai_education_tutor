# =========================================================
# REVIEWER AGENT
# =========================================================

from ai_engine import create_client, call_groq


def run_reviewer_agent(
    question,
    tutor_answer,
    context="",
    api_key=None,
    model=None,
):
    """
    Reviewer Agent

    Responsibilities:
    - Check the Tutor Agent's answer
    - Identify errors
    - Check relevance
    - Check hallucinations
    - Suggest corrections
    """

    if not tutor_answer:
        raise ValueError("Reviewer Agent received an empty tutor answer.")

    client = create_client(api_key)

    system_prompt = """
You are the Reviewer Agent in an AI Education system.

Your job is to critically review an educational answer.

Check:

1. Accuracy
2. Relevance to the question
3. Logical consistency
4. Clarity
5. Unsupported claims
6. Hallucinations
7. Whether the provided source context was followed

Return a structured review using exactly this format:

VERDICT: PASS or FAIL

ISSUES:
- issue 1
- issue 2

CORRECTIONS:
- correction 1
- correction 2

FINAL_RECOMMENDATION:
A short recommendation for the Final Answer Agent.

If there are no significant issues, use:

VERDICT: PASS
ISSUES:
- None
CORRECTIONS:
- None
"""

    user_prompt = f"""
STUDENT QUESTION:
{question}

SOURCE CONTEXT:
{context}

TUTOR ANSWER:
{tutor_answer}

Review the Tutor answer carefully.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
