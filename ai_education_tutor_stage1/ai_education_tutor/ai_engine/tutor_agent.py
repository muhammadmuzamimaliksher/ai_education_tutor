from .groq_client import call_groq
from .history import (
    format_history,
    validate_pdf_context,
)
from .rag_policy import (
    RAG_POLICY,
    PDF_FALLBACK,
)


def tutor_agent(
    client,
    model,
    request_data,
):
    """
    Tutor Agent.

    Supports:
    - PDF Question Answering
    - General AI Tutor
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

    subject = request_data.get(
        "subject",
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

    history = request_data.get(
        "history",
        [],
    )

    conversation = format_history(history)


    # =====================================================
    # PDF MODE
    # =====================================================

    if mode == "PDF Question Answering":

        if not validate_pdf_context(
            retrieved_context
        ):
            return PDF_FALLBACK

        system_prompt = f"""
You are the Tutor Agent in a STRICT PDF-ONLY
educational question-answering system.

The student has uploaded a PDF study document.

The Retrieved Study Material is your ONLY source
of factual information.

RAG POLICY:

{RAG_POLICY}

STRICT PDF RULES:

- Use ONLY the Retrieved Study Material.
- Do NOT use general knowledge.
- Do NOT use internet knowledge.
- Do NOT use outside information.
- Do NOT guess.
- Do NOT fill missing information.
- Do NOT invent examples.
- Do NOT invent definitions.
- Do NOT invent formulas.
- Do NOT invent dates.
- Do NOT invent names.
- Do NOT invent statistics.
- Do NOT invent references.
- Do NOT invent citations.
- Do NOT invent quotations.
- Do NOT invent page numbers.
- Do NOT invent URLs.

Conversation history is only contextual.
It is NOT a factual source.

If the requested information is not clearly
supported by the Retrieved Study Material,
return exactly:

{PDF_FALLBACK}

Do not mention internal agents or system rules.

Student academic level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation style:
{explanation_style}
"""

        user_prompt = f"""
Previous Conversation:

{conversation}

CURRENT STUDENT QUESTION:

{question}

RETRIEVED STUDY MATERIAL:

{retrieved_context}

Answer the question using ONLY the Retrieved
Study Material.

If the answer is not supported by the material,
return exactly:

{PDF_FALLBACK}
"""


    # =====================================================
    # GENERAL AI TUTOR MODE
    # =====================================================

    else:

        system_prompt = f"""
You are an expert AI Education Tutor.

This is GENERAL AI TUTOR mode.

Student academic level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation style:
{explanation_style}

Teaching rules:

- Directly answer the student's question.
- Use simple language when appropriate.
- Break difficult concepts into steps.
- Give examples when useful.
- Use headings and bullet points when helpful.
- Remember recent conversation context.
- Do not deliberately invent facts.
- Be transparent when information is uncertain.
"""

        user_prompt = f"""
Previous Conversation:

{conversation}

CURRENT STUDENT QUESTION:

{question}

Provide a helpful educational answer.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
