from .groq_client import create_client, call_groq
from .history import (
    format_history,
    validate_pdf_context,
)
from .rag_policy import (
    RAG_POLICY,
    PDF_FALLBACK,
)


TOOL_INSTRUCTIONS = {

    "Explain Again": """
Explain the topic again using different wording.
Make the explanation clearer than before.
""",

    "Explain Simply": """
Explain the topic as simply as possible.
Imagine you are teaching a beginner.
Avoid unnecessary technical terminology.
""",

    "Give Example": """
Give an example supported by the source material.

In PDF mode, ONLY use an example that actually
exists in the PDF.

If no suitable example exists in the PDF,
return the PDF fallback.
""",

    "Exam Answer": """
Create an exam-ready answer.

Use only information supported by the source
material.

Do not add outside facts.
""",

    "Summary": """
Create concise study notes.

Include only information supported by the
source material.

Do not introduce outside information.
""",

    "Quiz": """
Create a short educational quiz.

In PDF mode:

- Questions must come from the PDF.
- Answers must be supported by the PDF.
- Explanations must be supported by the PDF.
- Do not ask about information not present in the PDF.

Include:

1. Five multiple-choice questions.
2. Four options for each question.
3. Clearly identify the correct answer.
4. Add a short explanation for each answer.
""",
}


def run_learning_tool(
    tool,
    request_data,
    api_key,
    model,
):
    """
    Run one of the educational learning tools.
    """

    client = create_client(api_key)

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

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )

    history = format_history(
        request_data.get(
            "history",
            [],
        )
    )

    instruction = TOOL_INSTRUCTIONS.get(
        tool,
        "Provide a useful educational response.",
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
You are an educational learning assistant
operating in STRICT PDF-ONLY mode.

The Retrieved Study Material is your ONLY
factual source.

RAG POLICY:

{RAG_POLICY}

STRICT RULES:

- Use ONLY the Retrieved Study Material.
- Do NOT use general knowledge.
- Do NOT use internet knowledge.
- Do NOT guess.
- Do NOT fill missing information.
- Do NOT invent examples.
- Do NOT invent definitions.
- Do NOT invent facts.
- Do NOT invent citations.
- Do NOT invent page numbers.
- Do NOT invent references.
- Do NOT invent URLs.

Conversation history is contextual only.
It is NOT factual evidence.

If the requested task cannot be completed
from the PDF, return exactly:

{PDF_FALLBACK}

Do not mention internal agents.

TOOL:

{tool}

TASK:

{instruction}

Academic level:
{academic_level}

Language:
{language}
"""

        user_prompt = f"""
RECENT CONVERSATION:

{history}

CURRENT QUESTION:

{question}

RETRIEVED PDF STUDY MATERIAL:

{retrieved_context}

Perform the requested task using ONLY
the Retrieved Study Material.

If the information is not supported by the PDF,
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

Academic level:
{academic_level}

Language:
{language}

Task:

{instruction}

Teaching rules:

- Be educational.
- Be clear.
- Use appropriate examples.
- Organize the response well.
- Do not mention internal agents.
- Do not deliberately invent facts.
"""

        user_prompt = f"""
Recent Conversation:

{history}

Current Topic / Question:

{question}

Perform the requested learning task.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )
