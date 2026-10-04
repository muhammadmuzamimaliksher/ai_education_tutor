# ai_engine/learning_tools_engine.py

import json

from .groq_client import create_client, call_groq

from .history import (
    format_history,
    validate_pdf_context,
)

from .rag_policy import (
    RAG_POLICY,
    PDF_FALLBACK,
)


# =========================================================
# TOOL INSTRUCTIONS
# =========================================================

TOOL_INSTRUCTIONS = {

    "Explain Again": """
Explain the topic again using only the provided source
material.

In PDF mode:
- Do not add information.
- Do not introduce outside facts.
- Preserve the meaning of the PDF.
""",

    "Explain Simply": """
Explain the topic as simply as possible.

In PDF mode:
- Use ONLY the PDF material.
- Do not introduce outside information.
- Do not invent examples or facts.
""",

    "Give Example": """
Give an example supported by the source material.

In PDF mode:
- ONLY use an example that actually exists in the PDF.
- Do not create a new example.
- If no suitable example exists, return the PDF fallback.
""",

    "Exam Answer": """
Provide the answer using ONLY the source material.

In PDF mode:
- Preserve the original wording whenever possible.
- Do not add facts.
- Do not create additional explanations.
- Do not expand the answer using general knowledge.
- Do not invent information.
""",

    "Summary": """
Create concise study notes.

In PDF mode:
- Include ONLY information supported by the PDF.
- Do not introduce outside information.
""",

    "Quiz": """
Create a 5-question multiple-choice quiz.

Every question must be answerable from the provided
source material.

Each question MUST contain:

- question
- four options
- correct_index
- explanation

The explanation MUST be based only on the provided
source material.

Do not omit the explanation field.

If the source material does not contain enough
information for a useful explanation, use a short
explanation based directly on the relevant source
passage.

IMPORTANT:

The correct_index must be the zero-based index of the
correct option.

0 = first option
1 = second option
2 = third option
3 = fourth option

Do NOT write the correct answer separately.

Return exactly 5 questions.

Return ONLY valid JSON.

Required format:

{
    "title": "Quiz title",
    "questions": [
        {
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "correct_index": 1,
            "explanation": "Source-supported explanation."
        }
    ]
}
""",
}


# =========================================================
# JSON CLEANING
# =========================================================

def clean_json_response(response):

    if not response:
        raise ValueError(
            "The AI returned an empty response."
        )

    response = response.strip()

    if response.startswith("```"):

        lines = response.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response = "\n".join(lines).strip()

    # Validate JSON before returning
    try:
        json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Quiz response was not valid JSON: {error}"
        )

    return response


# =========================================================
# MAIN FUNCTION
# =========================================================

def run_learning_tool(
    tool,
    request_data,
    api_key,
    model,
):
    """
    Run one of the educational learning tools.
    """

    client = create_client(
        api_key
    )

    mode = request_data.get(
        "mode",
        "AI Tutor",
    )

    question = request_data.get(
        "question",
        "",
    )

    answer = request_data.get(
        "answer",
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

    # -----------------------------------------------------
    # IMPORTANT
    # Accept both names for compatibility.
    # -----------------------------------------------------

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )

    if not retrieved_context:

        retrieved_context = request_data.get(
            "context",
            "",
        )

    history = format_history(
        request_data.get(
            "history",
            [],
        )
    )

    explanation_style = request_data.get(
        "explanation_style",
        "Normal",
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

        # -------------------------------------------------
        # SPECIAL PDF QUIZ
        # -------------------------------------------------

        if tool == "Quiz":

            system_prompt = f"""
You are a strict PDF-based educational quiz generator.

The Retrieved PDF Study Material is your ONLY factual
source.

RAG POLICY:

{RAG_POLICY}

STRICT RULES:

- Use ONLY the Retrieved PDF Study Material.
- Do NOT use general knowledge.
- Do NOT use internet knowledge.
- Do NOT guess.
- Do NOT fill missing information.
- Do NOT invent facts.
- Do NOT invent dates.
- Do NOT invent names.
- Do NOT invent statistics.
- Do NOT invent historical information.
- Every correct answer MUST be directly supported
  by the PDF.
- Every explanation MUST be supported by the PDF.
- Do not ask questions whose answers are absent
  from the PDF.

Return ONLY valid JSON.

No markdown.
No ```json.
No introductory text.

Exactly 5 questions.
Exactly 4 options per question.

Use zero-based correct_index:

0 = A
1 = B
2 = C
3 = D

The correct answer must be supported by the PDF.
"""

            user_prompt = f"""
CURRENT QUESTION:

{question}

RETRIEVED PDF STUDY MATERIAL:

{retrieved_context}

Create the 5-question quiz now.

Return ONLY the required JSON object.
"""

            result = call_groq(
                client,
                model,
                system_prompt,
                user_prompt,
            )

            return clean_json_response(
                result
            )

        # -------------------------------------------------
        # NORMAL PDF LEARNING TOOLS
        # -------------------------------------------------

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

For Exam Answer:

- Answer only from the PDF.
- Preserve the source wording whenever possible.
- Do not add an explanation that is not present
  in the PDF.
- Do not expand the answer with outside knowledge.
- Do not invent exam facts.

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

Explanation style:

{explanation_style}
"""

        user_prompt = f"""
RECENT CONVERSATION:

{history}

CURRENT QUESTION:

{question}

PREVIOUS PDF ANSWER:

{answer}

RETRIEVED PDF STUDY MATERIAL:

{retrieved_context}

Perform the requested task using ONLY
the Retrieved PDF Study Material.

If the information is not supported by the PDF,
return exactly:

{PDF_FALLBACK}
"""

        return call_groq(
            client,
            model,
            system_prompt,
            user_prompt,
        )

    # =====================================================
    # GENERAL AI TUTOR MODE
    # =====================================================

    system_prompt = f"""
You are an expert AI Education Tutor.

This is GENERAL AI TUTOR mode.

Academic level:

{academic_level}

Language:

{language}

Task:

{instruction}

Explanation style:

{explanation_style}

Teaching rules:

- Be educational.
- Be clear.
- Use appropriate examples.
- Organize the response well.
- Do not mention internal agents.
- Do not deliberately invent facts.

For Quiz:

Return ONLY valid JSON using this structure:

{{
    "title": "Quiz title",
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "correct_index": 0,
            "explanation": "Explanation."
        }}
    ]
}}

Exactly 5 questions.
Exactly 4 options per question.

correct_index:
0 = A
1 = B
2 = C
3 = D

Do not use markdown for Quiz output.
"""

    user_prompt = f"""
Recent Conversation:

{history}

Current Topic / Question:

{question}

Perform the requested learning task.
"""

    result = call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )

    if tool == "Quiz":

        return clean_json_response(
            result
        )

    return result
