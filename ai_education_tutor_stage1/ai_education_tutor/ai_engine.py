import os
from groq import Groq
from pypdf import PdfReader


# =========================================================
# RAG POLICY
# =========================================================

POLICY_FILE = os.path.join(
    os.path.dirname(__file__),
    "RAG_POLICY.pdf"
)


def load_rag_policy():
    """Load the anti-hallucination policy PDF."""

    if not os.path.exists(POLICY_FILE):
        return """
        Follow these rules:
        1. Do not invent facts.
        2. For PDF questions, use only retrieved study material.
        3. If the study material does not contain enough information,
           clearly say that the information is unavailable.
        4. Do not fabricate citations, page numbers, quotations,
           references, URLs, statistics, or facts.
        """

    try:
        reader = PdfReader(POLICY_FILE)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        policy = "\n".join(pages).strip()

        if policy:
            return policy

    except Exception:
        pass

    return """
    Do not invent facts.
    For PDF questions, answer only from retrieved study material.
    If information is unavailable, say so clearly.
    """


RAG_POLICY = load_rag_policy()


# =========================================================
# GROQ CLIENT
# =========================================================

def create_client(api_key):
    """Create Groq client."""

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(api_key=api_key)


# =========================================================
# GROQ CALL
# =========================================================

def call_groq(
    client,
    model,
    system_prompt,
    user_prompt,
):
    """Send a request to Groq."""

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

    if not response.choices:
        raise RuntimeError(
            "Groq returned no response."
        )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return answer.strip()


# =========================================================
# TUTOR AGENT
# =========================================================

def tutor_agent(
    client,
    model,
    request_data,
):
    """
    Tutor Agent.

    Works in both:
    - PDF Q&A mode
    - General AI Tutor mode
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

    # -----------------------------------------------------
    # PDF MODE
    # -----------------------------------------------------

    if mode == "PDF Question Answering":

        system_prompt = f"""
You are the Tutor Agent in a grounded educational
question-answering system.

The user uploaded a study document.

You MUST answer using the retrieved study material
provided below.

RAG POLICY:

{RAG_POLICY}

IMPORTANT RULES:

- Use the retrieved study material as the primary source.
- Do not invent information.
- Do not add unsupported facts.
- Do not fabricate citations or page numbers.
- Explain the material clearly.
- If the retrieved material does not contain enough
  information, say so instead of guessing.

Student level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation style:
{explanation_style}
"""

        user_prompt = f"""
Student Question:

{question}

Retrieved Study Material:

{retrieved_context}

Provide a clear educational answer based on the
retrieved study material.
"""

    # -----------------------------------------------------
    # GENERAL AI TUTOR MODE
    # -----------------------------------------------------

    else:

        system_prompt = f"""
You are an expert AI Education Tutor.

Your job is to help students understand academic
and educational topics.

You may answer general educational questions in
this mode.

Student level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation style:
{explanation_style}

Teaching rules:

- Explain concepts clearly.
- Start with the direct answer.
- Use simple language when appropriate.
- Give examples when useful.
- Break difficult concepts into steps.
- Use headings and bullet points when helpful.
- Do not deliberately make up facts.
- If you are uncertain about a fact, be transparent.
"""

        user_prompt = f"""
Student Question:

{question}

Provide a helpful educational explanation.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# =========================================================
# RESEARCH AGENT
# =========================================================

def research_agent(
    client,
    model,
    request_data,
    tutor_answer,
):
    """
    Research Agent checks the Tutor Agent response.
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

    # PDF MODE
    if mode == "PDF Question Answering":

        system_prompt = f"""
You are the Research Agent.

Your responsibility is to verify the Tutor Agent's
answer against the retrieved study material.

RAG POLICY:

{RAG_POLICY}

Rules:

- Check whether important claims are supported.
- Do not introduce new unsupported information.
- Do not fabricate references.
- Remove or identify unsupported claims.
- If the material is insufficient, say so clearly.
"""

        user_prompt = f"""
Question:

{question}

Retrieved Study Material:

{retrieved_context}

Tutor Agent Answer:

{tutor_answer}

Review the answer for grounding and factual support.

Return a concise verification/research report.
"""

    # GENERAL MODE
    else:

        system_prompt = """
You are the Research Agent in an educational
multi-agent AI tutor.

Review the Tutor Agent's answer.

Check:

- Logical correctness
- Educational usefulness
- Internal consistency
- Potential unsupported claims
- Whether the answer actually addresses the question

Do not unnecessarily rewrite the answer.

Return a concise verification report.
"""

        user_prompt = f"""
Student Question:

{question}

Tutor Agent Answer:

{tutor_answer}

Review this answer and identify any important
problems or improvements.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# =========================================================
# EVALUATOR AGENT
# =========================================================

def evaluator_agent(
    client,
    model,
    request_data,
    tutor_answer,
    research_report,
):
    """
    Evaluator Agent produces the final answer.
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

    # -----------------------------------------------------
    # PDF MODE
    # -----------------------------------------------------

    if mode == "PDF Question Answering":

        system_prompt = f"""
You are the final Evaluator Agent.

Your job is to produce the final answer for the
student.

RAG POLICY:

{RAG_POLICY}

STRICT PDF RULES:

1. The answer must be grounded in the retrieved
   study material.

2. Do not introduce unsupported facts.

3. Do not invent citations, references, page numbers,
   quotations, statistics, or URLs.

4. If the retrieved study material is insufficient,
   clearly state that.

5. Do not pretend that information exists in the
   document when it does not.

6. Make the final answer educational and easy to
   understand.

Student level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}
"""

        user_prompt = f"""
Student Question:

{question}

Retrieved Study Material:

{retrieved_context}

Tutor Agent Answer:

{tutor_answer}

Research Agent Report:

{research_report}

Create the final grounded answer.
"""

    # -----------------------------------------------------
    # GENERAL AI TUTOR MODE
    # -----------------------------------------------------

    else:

        system_prompt = f"""
You are the final Evaluator Agent of an AI Education
Tutor.

Create the best final educational response.

Student level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}

Requirements:

- Directly answer the student's question.
- Make the explanation clear.
- Correct important issues identified by Research.
- Do not include unnecessary internal agent discussion.
- Do not mention "Tutor Agent", "Research Agent",
  or "Evaluator Agent" in the final answer.
- Use examples or steps when useful.
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


# =========================================================
# MAIN AI TUTOR PIPELINE
# =========================================================

def run_ai_tutor(
    request_data,
    api_key,
    model,
):
    """
    Main multi-agent workflow:

    Tutor → Research → Evaluator
    """

    client = create_client(api_key)

    tutor_answer = tutor_agent(
        client,
        model,
        request_data,
    )

    research_report = research_agent(
        client,
        model,
        request_data,
        tutor_answer,
    )

    final_answer = evaluator_agent(
        client,
        model,
        request_data,
        tutor_answer,
        research_report,
    )

    return final_answer
