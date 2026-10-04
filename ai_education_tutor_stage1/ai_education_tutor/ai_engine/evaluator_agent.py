from .groq_client import call_groq
from .history import validate_pdf_context
from .rag_policy import (
    RAG_POLICY,
    PDF_FALLBACK,
)


def evaluator_agent(
    client,
    model,
    request_data,
    tutor_answer,
    research_report,
):
    """
    Final Evaluator Agent.

    PDF mode:
        Only PDF-supported information is allowed.

    General mode:
        Normal educational evaluation.
    """

    mode = request_data.get(
        "mode",
        "AI Tutor",
    )

    question = request_data.get(
        "question",
        "",
    )

    academic_level = request_data.get(
        "academic_level",
        "General",
    )

    language = request_data.get(
        "language",
        "English",
    )

    explanation_style = request_data.get(
        "explanation_style",
        "Simple",
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
            return PDF_FALLBACK

        system_prompt = f"""
You are the FINAL EVALUATOR AGENT for a
STRICT PDF-ONLY educational system.

The Retrieved Study Material is the ONLY
factual source.

RAG POLICY:

{RAG_POLICY}

ABSOLUTE RULES:

1. Use ONLY information in the Retrieved Study Material.

2. Do NOT use general knowledge.

3. Do NOT use internet knowledge.

4. Do NOT use information from your training data
   unless it is explicitly present in the PDF context.

5. Do NOT guess.

6. Do NOT fill missing information.

7. Do NOT add helpful information from outside the PDF.

8. Do NOT invent examples.

9. Do NOT invent definitions.

10. Do NOT invent formulas.

11. Do NOT invent dates, names, statistics,
    procedures, references, citations, quotations,
    URLs, or page numbers.

12. Conversation history is NOT evidence.

13. Tutor Agent output is NOT evidence.

14. Research Agent output is NOT evidence.

15. ONLY Retrieved Study Material is evidence.

16. If the requested information is not clearly
    supported by the PDF, return exactly:

{PDF_FALLBACK}

17. If only part of the question is supported,
    provide only the supported part.

18. Do not mention internal agents.

FINAL CHECK:

Before answering, verify that every factual
claim is supported by the PDF.

If unsupported content exists, remove it.

If the remaining evidence cannot answer the
question, return:

{PDF_FALLBACK}

Student academic level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}
"""

        user_prompt = f"""
STUDENT QUESTION:

{question}

RETRIEVED STUDY MATERIAL:

{retrieved_context}

TUTOR AGENT ANSWER:

{tutor_answer}

PDF EVIDENCE VERIFICATION REPORT:

{research_report}

Create the final answer.

IMPORTANT:

The Retrieved Study Material is the ONLY
factual source.

If the answer is not supported by the PDF,
return exactly:

{PDF_FALLBACK}
"""


    # =====================================================
    # GENERAL MODE
    # =====================================================

    else:

        system_prompt = f"""
You are the final Evaluator Agent of an
AI Education Tutor.

Create the best final educational response.

Student academic level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}

Requirements:

- Directly answer the question.
- Correct important issues identified by Research.
- Keep the answer educational and clear.
- Use examples when useful.
- Do not mention internal agents.
- Do not deliberately invent facts.
"""

        user_prompt = f"""
Student Question:

{question}

Tutor Agent Answer:

{tutor_answer}

Research Agent Report:

{research_report}

Create the final student-friendly answer.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
