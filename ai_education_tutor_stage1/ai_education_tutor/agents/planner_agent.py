# =========================================================
# PLANNER AGENT
# =========================================================

from ai_engine import create_client, call_groq


def run_planner_agent(
    question,
    context="",
    api_key=None,
    model=None,
):
    """
    Planner Agent

    Responsibilities:
    - Understand the student's question
    - Identify what needs to be explained
    - Create a clear learning plan
    """

    if not question:
        raise ValueError("Planner Agent received an empty question.")

    client = create_client(api_key)

    system_prompt = """
You are the Planner Agent in an AI Education system.

Your job is to analyze a student's question and create a
clear plan for another AI Tutor Agent.

Do NOT answer the student's question directly.

Identify:
1. What the student is asking
2. The main concept involved
3. Important subtopics
4. What information should be used
5. The best teaching approach

Keep the plan concise and educational.
"""

    user_prompt = f"""
STUDENT QUESTION:
{question}

AVAILABLE CONTEXT:
{context}

Create a learning plan for the Tutor Agent.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
