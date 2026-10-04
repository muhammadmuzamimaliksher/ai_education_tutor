from groq import Groq


# =========================================================
# Groq Client
# =========================================================

def create_client(api_key: str):
    """
    Create and return a Groq client.
    """

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return Groq(api_key=api_key)


# =========================================================
# Generic Groq Call
# =========================================================

def call_groq(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Send a request to Groq and return the generated text.
    """

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
        temperature=0.4,
    )

    if not response.choices:
        raise RuntimeError("Groq returned no choices.")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()


# =========================================================
# Tutor Agent
# =========================================================

def tutor_agent(
    client,
    model: str,
    request_data: dict,
) -> str:
    """
    Tutor Agent creates the initial educational explanation.
    """

    question = request_data["question"]
    academic_level = request_data["academic_level"]
    subject = request_data["subject"]
    language = request_data["language"]
    explanation_style = request_data["explanation_style"]

system_prompt = """
You are the Tutor Agent in an AI Education platform.

You teach students using retrieved educational
material whenever it is available.

Your priorities are:

1. Use retrieved study material as the primary source.
2. Explain concepts clearly.
3. Adapt the explanation to the student's level.
4. Do not invent facts.
5. Do not pretend unsupported information came from
   the uploaded document.
6. If the retrieved material is insufficient, say so.
7. Use examples when useful.
8. For mathematics, show steps.
9. For science, explain cause and effect clearly.
10. Follow the requested language and style.

You are part of a multi-agent educational system.
"""

   retrieved_context = request_data.get(
    "retrieved_context",
    "",
)

user_prompt = f"""
Student Academic Level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation Style:
{explanation_style}

Student Question:
{question}

Retrieved Study Material:
{retrieved_context if retrieved_context else "No study material was uploaded."}

Instructions:

If study material is provided, use it as the
primary source for answering the student's question.

Do not invent information that conflicts with
the retrieved study material.

If the study material does not contain enough
information to answer the question, clearly say
that the provided material does not contain
enough information and then give a general
explanation only when appropriate.

Provide a useful educational explanation.
"""

    return call_groq(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# =========================================================
# Research Agent
# =========================================================

def research_agent(
    client,
    model: str,
    request_data: dict,
    tutor_answer: str,
) -> str:
    """
    Research Agent checks the answer for factual quality.

    This is the foundation for the future RAG layer.
    """

    question = request_data["question"]
    subject = request_data["subject"]

    system_prompt = """
You are the Research Agent for an AI education system.

Your task is to analyze the proposed tutor answer.

Check:

- factual consistency
- logical consistency
- missing important information
- unsupported claims
- possible misconceptions
- whether the answer matches the question

Do not unnecessarily rewrite the entire answer.

Return concise research feedback that another agent can use.
"""

    user_prompt = f"""
Subject:
{subject}

Question:
{question}

Proposed Tutor Answer:
{tutor_answer}

Analyze the answer and identify corrections or improvements.
"""

    return call_groq(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# =========================================================
# Evaluator Agent
# =========================================================

def evaluator_agent(
    client,
    model: str,
    request_data: dict,
    tutor_answer: str,
    research_feedback: str,
) -> str:
    """
    Evaluator Agent produces the final student-friendly answer.
    """

    system_prompt = """
You are the Final Evaluator Agent in an AI Education platform.

Your job is to produce the final answer for the student.

Use the tutor answer and research feedback.

Requirements:

1. Correct factual problems identified by the Research Agent.
2. Do not add unsupported information.
3. Keep the answer appropriate for the student's academic level.
4. Make the explanation easy to understand.
5. Use headings and bullet points where useful.
6. Show steps for problems that require steps.
7. Include examples where useful.
8. Do not mention internal agents.
9. Do not mention this evaluation process.
10. Answer directly.
"""

    user_prompt = f"""
Academic Level:
{request_data["academic_level"]}

Subject:
{request_data["subject"]}

Language:
{request_data["language"]}

Explanation Style:
{request_data["explanation_style"]}

Student Question:
{request_data["question"]}

Tutor Answer:
{tutor_answer}

Research Feedback:
{research_feedback}

Produce the final answer for the student.
"""

    return call_groq(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# =========================================================
# Multi-Agent Orchestrator
# =========================================================

def run_ai_tutor(
    request_data: dict,
    api_key: str,
    model: str,
) -> str:
    """
    Run the complete multi-agent AI Tutor workflow.

    Workflow:

    User Question
          ↓
    Tutor Agent
          ↓
    Research Agent
          ↓
    Evaluator Agent
          ↓
    Final Answer
    """

    if not request_data.get("question", "").strip():
        raise ValueError("Question cannot be empty.")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    if not model:
        raise ValueError("AI model is not configured.")

    client = create_client(api_key)

    # Agent 1
    tutor_answer = tutor_agent(
        client=client,
        model=model,
        request_data=request_data,
    )

    # Agent 2
    research_feedback = research_agent(
        client=client,
        model=model,
        request_data=request_data,
        tutor_answer=tutor_answer,
    )

    # Agent 3
    final_answer = evaluator_agent(
        client=client,
        model=model,
        request_data=request_data,
        tutor_answer=tutor_answer,
        research_feedback=research_feedback,
    )

    return final_answer
