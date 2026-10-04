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
    """Load the anti-hallucination policy."""

    if not os.path.exists(POLICY_FILE):
        return """
        Rules:
        1. Do not invent facts.
        2. For PDF questions, use only retrieved study material.
        3. If the study material is insufficient, say so clearly.
        4. Never fabricate citations, page numbers, references,
           quotations, URLs, statistics, or facts.
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
# FORMAT CONVERSATION HISTORY
# =========================================================

def format_history(history, max_messages=8):
    """
    Convert Streamlit conversation history into
    a compact text format.
    """

    if not history:
        return "No previous conversation."

    recent_history = history[-max_messages:]

    formatted = []

    for message in recent_history:

        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "user":
            label = "Student"
        else:
            label = "AI Tutor"

        formatted.append(
            f"{label}: {content}"
        )

    return "\n\n".join(formatted)


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

    Supports:
    - PDF Question Answering
    - General AI Tutor
    - Conversation memory
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

        system_prompt = f"""
You are the Tutor Agent in a grounded educational
question-answering system.

The student has uploaded a study document.

You MUST use the retrieved study material as the
primary source.

RAG POLICY:

{RAG_POLICY}

STRICT RULES:

- Do not invent information.
- Do not add unsupported facts.
- Do not fabricate citations.
- Do not fabricate page numbers.
- Do not fabricate quotations.
- Do not fabricate references.
- Do not fabricate URLs.
- If the retrieved material is insufficient,
  clearly say so.
- Use previous conversation only to understand
  context and references.
- The study material remains the authority for
  document-based answers.

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
Previous Conversation:

{conversation}

Current Student Question:

{question}

Retrieved Study Material:

{retrieved_context}

Answer the current question clearly using the
retrieved study material.
"""


    # =====================================================
    # GENERAL AI TUTOR MODE
    # =====================================================

    else:

        system_prompt = f"""
You are an expert AI Education Tutor.

You help students understand academic and
educational topics.

This is GENERAL AI TUTOR mode.

The student does not need to upload a PDF.

Student level:
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
- Remember the recent conversation context.
- Do not deliberately invent facts.
- Be transparent when information is uncertain.
"""

        user_prompt = f"""
Previous Conversation:

{conversation}

Current Student Question:

{question}

Provide a helpful educational answer.
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
    Research Agent verifies the Tutor Agent response.
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

        system_prompt = f"""
You are the Research Agent.

Verify the Tutor Agent's answer against the
retrieved study material.

RAG POLICY:

{RAG_POLICY}

Rules:

- Check important claims against the material.
- Do not introduce unsupported information.
- Do not fabricate references.
- Identify unsupported claims.
- Prefer transparency when evidence is insufficient.
"""

        user_prompt = f"""
Student Question:

{question}

Retrieved Study Material:

{retrieved_context}

Tutor Agent Answer:

{tutor_answer}

Verify the answer against the study material.

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
    Evaluator Agent creates final answer.
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

        system_prompt = f"""
You are the final Evaluator Agent.

Create the final answer for the student.

RAG POLICY:

{RAG_POLICY}

STRICT PDF RULES:

- Ground the answer in the retrieved material.
- Remove unsupported claims.
- Do not invent information.
- Do not invent citations.
- Do not invent page numbers.
- Do not invent references.
- Do not invent URLs.
- If evidence is insufficient, say so.
- Make the final answer easy to understand.

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

Create the final grounded educational answer.
"""


    # =====================================================
    # GENERAL MODE
    # =====================================================

    else:

        system_prompt = f"""
You are the final Evaluator Agent of an
AI Education Tutor.

Create the best final educational response.

Student level:
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
- Do not mention the internal agents.
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
# MAIN AI PIPELINE
# =========================================================

def run_ai_tutor(
    request_data,
    api_key,
    model,
):
    """
    Main workflow:

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


# =========================================================
# LEARNING TOOLS
# =========================================================

def run_learning_tool(
    tool,
    request_data,
    api_key,
    model,
):
    """
    Generate learning content:

    - Explain Again
    - Explain Simply
    - Give Example
    - Exam Answer
    - Summary
    - Quiz
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


    # =====================================================
    # TOOL INSTRUCTIONS
    # =====================================================

    tool_instructions = {

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
Explain the concept using practical,
easy-to-understand examples.
""",

        "Exam Answer": """
Create an exam-ready answer.
Use a clear definition, explanation,
important points, and examples where appropriate.
""",

        "Summary": """
Create concise study notes.
Include:
- Main concept
- Important points
- Key definitions
- Important facts
- Useful examples
""",

        "Quiz": """
Create a short educational quiz.

Include:
1. Five multiple-choice questions.
2. Four options for each question.
3. Clearly identify the correct answer.
4. Add a short explanation for each answer.
""",
    }


    instruction = tool_instructions.get(
        tool,
        "Provide a useful educational response."
    )


    # =====================================================
    # PDF MODE
    # =====================================================

    if mode == "PDF Question Answering":

        system_prompt = f"""
You are an educational learning assistant.

The student is working from an uploaded PDF.

RAG POLICY:

{RAG_POLICY}

STRICT RULES:

- Use only the retrieved study material.
- Do not introduce unsupported information.
- Do not fabricate facts.
- Do not fabricate citations or page numbers.
- Do not fabricate references.
- If the material is insufficient, say so.

Student level:
{academic_level}

Language:
{language}

Task:

{instruction}
"""

        user_prompt = f"""
Recent Conversation:

{history}

Current Topic / Question:

{question}

Retrieved Study Material:

{retrieved_context}

Perform the requested learning task using
the study material.
"""


    # =====================================================
    # GENERAL AI TUTOR MODE
    # =====================================================

    else:

        system_prompt = f"""
You are an expert AI Education Tutor.

Student level:
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
