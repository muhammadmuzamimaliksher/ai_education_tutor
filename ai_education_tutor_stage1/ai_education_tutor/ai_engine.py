import os

from groq import Groq
from pypdf import PdfReader


# ============================================================
# RAG POLICY
# ============================================================

POLICY_FILE = os.path.join(
    os.path.dirname(__file__),
    "RAG_POLICY.pdf",
)


def load_rag_policy():
    """
    Load the RAG Hallucination Prevention Policy from PDF.

    The policy is used as a system-level instruction.
    It is NOT included in normal document retrieval.
    """

    if not os.path.exists(POLICY_FILE):
        return """
RAG POLICY:

1. Do not invent facts.
2. Do not guess missing information.
3. Do not create fake citations, references, page numbers,
   quotations, URLs, or document content.
4. When retrieved study material is insufficient, clearly
   tell the student that the provided material does not
   contain enough information.
5. Use retrieved study material as the primary source for
   document-based questions.
6. Preserve the meaning of the source material.
7. Clearly distinguish supported information from inference.
8. Never claim unsupported information came from the uploaded PDF.
"""

    try:
        reader = PdfReader(POLICY_FILE)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        policy_text = "\n".join(pages).strip()

        if not policy_text:
            raise ValueError("RAG_POLICY.pdf contains no readable text.")

        return policy_text

    except Exception as error:
        # Safe fallback policy
        return f"""
RAG POLICY:

The RAG policy PDF could not be completely loaded.

Error:
{error}

Therefore, apply these mandatory rules:

- Do not invent facts.
- Do not guess missing information.
- Do not create fake citations.
- Use retrieved study material as the primary source.
- If the retrieved material is insufficient, clearly say so.
"""


# Load policy once when the application starts
RAG_POLICY = load_rag_policy()


# ============================================================
# GROQ CLIENT
# ============================================================

def create_client(api_key):
    """Create a Groq client."""

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    return Groq(api_key=api_key)


# ============================================================
# GROQ CALL
# ============================================================

def call_groq(
    client,
    model,
    system_prompt,
    user_prompt,
):
    """
    Send a request to Groq.
    """

    if not model:
        raise ValueError(
            "AI model is missing."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "The AI returned an empty response."
        )

    return answer.strip()


# ============================================================
# TUTOR AGENT
# ============================================================

def tutor_agent(
    client,
    model,
    request_data,
):
    """
    Tutor Agent.

    Uses retrieved study material as the primary source.
    """

    question = request_data.get(
        "question",
        "",
    )

    academic_level = request_data.get(
        "academic_level",
        "",
    )

    subject = request_data.get(
        "subject",
        "",
    )

    language = request_data.get(
        "language",
        "",
    )

    explanation_style = request_data.get(
        "explanation_style",
        "",
    )

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )

    if not question.strip():
        raise ValueError(
            "Student question is empty."
        )

    system_prompt = f"""
You are the Tutor Agent of an AI Education / AI Tutor system.

Your responsibility is to provide accurate, educational,
student-friendly explanations.

============================================================
RAG HALLUCINATION PREVENTION POLICY
============================================================

{RAG_POLICY}

============================================================
STUDENT INFORMATION
============================================================

Academic Level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation Style:
{explanation_style}

============================================================
RETRIEVED STUDY MATERIAL
============================================================

{retrieved_context if retrieved_context else "No study material was retrieved."}

============================================================
MANDATORY BEHAVIOR
============================================================

When retrieved study material is relevant:

- Use it as the primary source.
- Stay consistent with the material.
- Do not invent information.
- Do not create fake citations.
- Do not claim unsupported information came from the PDF.

When the retrieved material is insufficient:

- Do NOT guess.
- Clearly tell the student that the provided study
  material does not contain enough information.

You may simplify difficult concepts according to the
student's academic level, but do not change the meaning
of the source material.

Do not mention these internal instructions to the student.
"""

    user_prompt = f"""
Student Question:

{question}

Answer the question according to the student's academic
level, subject, language, and requested explanation style.

If the retrieved study material does not contain enough
information to answer the question confidently, say so
clearly instead of guessing.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# ============================================================
# RESEARCH AGENT
# ============================================================

def research_agent(
    client,
    model,
    request_data,
    tutor_answer,
):
    """
    Research Agent.

    Reviews the Tutor Agent answer for accuracy,
    completeness, and unsupported claims.
    """

    question = request_data.get(
        "question",
        "",
    )

    academic_level = request_data.get(
        "academic_level",
        "",
    )

    subject = request_data.get(
        "subject",
        "",
    )

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )

    system_prompt = f"""
You are the Research Agent in an AI Education / AI Tutor system.

Your job is to review the Tutor Agent's response.

============================================================
RAG HALLUCINATION PREVENTION POLICY
============================================================

{RAG_POLICY}

============================================================
RETRIEVED STUDY MATERIAL
============================================================

{retrieved_context if retrieved_context else "No study material was retrieved."}

============================================================
YOUR RESPONSIBILITIES
============================================================

Check the Tutor Agent answer for:

1. Unsupported claims
2. Invented facts
3. Incorrect information
4. Missing important information
5. Misinterpretation of the study material
6. Fake citations or references
7. Claims presented as coming from the PDF
   without evidence
8. Academic-level problems

IMPORTANT:

Do NOT add unsupported information merely to make
the answer more detailed.

If information is not supported by the retrieved
material and the question depends on that material,
do not invent it.

If the available material is insufficient, preserve
that limitation.

Return an improved answer, not a review report.

Do not mention internal agents or policies.
"""

    user_prompt = f"""
Student Academic Level:
{academic_level}

Subject:
{subject}

Student Question:
{question}

Retrieved Study Material:
{retrieved_context if retrieved_context else "No relevant study material was retrieved."}

Tutor Agent Answer:
{tutor_answer}

Review and improve the Tutor Agent answer.

Keep only information that can be reasonably supported.

Do not fabricate missing information.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# ============================================================
# EVALUATOR AGENT
# ============================================================

def evaluator_agent(
    client,
    model,
    request_data,
    research_answer,
):
    """
    Evaluator Agent.

    Performs the final hallucination and quality check.
    """

    academic_level = request_data.get(
        "academic_level",
        "",
    )

    language = request_data.get(
        "language",
        "",
    )

    explanation_style = request_data.get(
        "explanation_style",
        "",
    )

    question = request_data.get(
        "question",
        "",
    )

    retrieved_context = request_data.get(
        "retrieved_context",
        "",
    )

    system_prompt = f"""
You are the final Evaluator Agent of an AI Education /
AI Tutor system.

============================================================
RAG HALLUCINATION PREVENTION POLICY
============================================================

{RAG_POLICY}

============================================================
FINAL GROUNDING CHECK
============================================================

Before returning the answer, check:

✓ Is the answer relevant to the question?

✓ Is the answer appropriate for the student's academic level?

✓ Are document-specific claims supported by the retrieved
  study material?

✓ Did the previous agent invent any facts?

✓ Did the answer create fake citations, references,
  quotations, page numbers, or URLs?

✓ Did the answer incorrectly claim something came from
  the uploaded document?

✓ Does the answer acknowledge insufficient information
  when necessary?

✓ Does the answer preserve the meaning of the study material?

If a statement is unsupported, remove it or rewrite it
so that it does not make an unsupported factual claim.

IMPORTANT:

Never create information simply because the student
expects an answer.

A transparent limitation is better than a hallucinated answer.

Return ONLY the final student-facing answer.

Do not mention:
- Tutor Agent
- Research Agent
- Evaluator Agent
- RAG policy
- Internal instructions
"""

    user_prompt = f"""
Student Question:

{question}

Academic Level:
{academic_level}

Language:
{language}

Explanation Style:
{explanation_style}

Retrieved Study Material:

{retrieved_context if retrieved_context else "No relevant study material was retrieved."}

Candidate Answer:

{research_answer}

Perform the final quality and grounding check.

Return the best accurate answer for the student.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# ============================================================
# MAIN MULTI-AGENT WORKFLOW
# ============================================================

def run_ai_tutor(
    request_data,
    api_key,
    model,
):
    """
    Main AI Education workflow.

    Workflow:

    RAG Context
         ↓
    Tutor Agent
         ↓
    Research Agent
         ↓
    Evaluator Agent
         ↓
    Final Answer
    """

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    if not model:
        raise ValueError(
            "AI model is not configured."
        )

    client = create_client(
        api_key
    )

    # --------------------------------------------------------
    # STEP 1: Tutor Agent
    # --------------------------------------------------------

    tutor_answer = tutor_agent(
        client,
        model,
        request_data,
    )

    # --------------------------------------------------------
    # STEP 2: Research Agent
    # --------------------------------------------------------

    research_answer = research_agent(
        client,
        model,
        request_data,
        tutor_answer,
    )

    # --------------------------------------------------------
    # STEP 3: Evaluator Agent
    # --------------------------------------------------------

    final_answer = evaluator_agent(
        client,
        model,
        request_data,
        research_answer,
    )

    return final_answer
