from .groq_client import call_groq
from .history import validate_pdf_context
from .rag_policy import (
    RAG_POLICY,
    PDF_FALLBACK,
)


def research_agent(
    client,
    model,
    request_data,
    tutor_answer,
):
    """
    Research / verification agent.

    PDF mode:
        PDF Evidence Checker.

    General mode:
        General answer reviewer.
    """

    mode = request_data.get(
        "mode",
        "AI Tutor",
    )

    question = request_data.get(
        "question",
        "",
    )

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )


    # =====================================================
    # PDF MODE
    # =====================================================

    if mode == "PDF Question Answering":

        if not validate_pdf_context(
            retrieved_context
        ):
            return (
                "No retrieved PDF evidence was available. "
                f"The final answer should be: {PDF_FALLBACK}"
            )

        system_prompt = f"""
You are the PDF Evidence Verification Agent.

You are NOT an external research agent.

Your ONLY responsibility is to verify whether
the Tutor Agent answer is supported by the
Retrieved Study Material.

RAG POLICY:

{RAG_POLICY}

STRICT RULES:

1. Use ONLY the Retrieved Study Material.
2. Do NOT use outside knowledge.
3. Do NOT correct the Tutor using general knowledge.
4. Do NOT introduce new facts.
5. Do NOT introduce new examples.
6. Do NOT invent citations.
7. Do NOT invent page numbers.
8. Identify unsupported claims.
9. Identify missing evidence.
10. If evidence is insufficient, say so clearly.

Do NOT perform internet research.
Do NOT use general knowledge.
"""

        user_prompt = f"""
STUDENT QUESTION:

{question}

RETRIEVED PDF STUDY MATERIAL:

{retrieved_context}

TUTOR AGENT ANSWER:

{tutor_answer}

Verify the Tutor Agent answer against ONLY
the Retrieved PDF Study Material.

Return a concise verification report.
"""


    # =====================================================
    # GENERAL MODE
    # =====================================================

    else:

        system_prompt = """
You are the Research Agent in an educational
multi-agent AI tutor.

Review the Tutor Agent answer.

Check:

- Logical correctness
- Educational usefulness
- Internal consistency
- Important factual problems
- Whether the question was answered

Do not unnecessarily rewrite the answer.

Return a concise verification report.
"""

        user_prompt = f"""
Student Question:

{question}

Tutor Agent Answer:

{tutor_answer}

Review this answer and identify important
problems or improvements.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
