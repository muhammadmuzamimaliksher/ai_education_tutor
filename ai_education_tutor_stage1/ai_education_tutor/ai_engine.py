from groq import Groq


def create_client(api_key):
    """Create Groq client."""
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return Groq(api_key=api_key)


def call_groq(client, model, system_prompt, user_prompt):
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
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


def tutor_agent(client, model, request_data):
    """Tutor Agent: creates the main educational explanation."""

    question = request_data.get("question", "")
    academic_level = request_data.get("academic_level", "")
    subject = request_data.get("subject", "")
    language = request_data.get("language", "")
    explanation_style = request_data.get("explanation_style", "")
    retrieved_context = request_data.get("retrieved_context", "")

    system_prompt = f"""
You are the Tutor Agent of an AI Education / AI Tutor system.

Student academic level:
{academic_level}

Subject:
{subject}

Language:
{language}

Explanation style:
{explanation_style}

Your job is to explain the student's question clearly and accurately.

If study material is provided below, use it as the primary source.

Do not invent facts that are not supported by the study material when the
question is related to that material.

Retrieved study material:
{retrieved_context}
"""

    user_prompt = f"""
Student Question:

{question}

Provide a clear educational answer suitable for the student's academic level.

Use examples where helpful.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


def research_agent(client, model, request_data, tutor_answer):
    """Research Agent: improves completeness and factual quality."""

    question = request_data.get("question", "")
    subject = request_data.get("subject", "")
    academic_level = request_data.get("academic_level", "")

    system_prompt = """
You are the Research Agent in an AI Education system.

Review the Tutor Agent's answer.

Identify missing important information, unclear explanations,
possible factual problems, and areas that need improvement.

Do not unnecessarily change a correct answer.
Return an improved educational answer.
"""

    user_prompt = f"""
Student academic level:
{academic_level}

Subject:
{subject}

Question:
{question}

Tutor Agent Answer:
{tutor_answer}

Improve the answer while keeping it appropriate for the student's level.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


def evaluator_agent(client, model, request_data, research_answer):
    """Evaluator Agent: performs final quality check."""

    academic_level = request_data.get("academic_level", "")
    language = request_data.get("language", "")
    explanation_style = request_data.get("explanation_style", "")

    system_prompt = """
You are the Evaluator Agent of an AI Education / AI Tutor system.

Your job is to perform the final quality check.

Check:

1. Accuracy
2. Clarity
3. Relevance
4. Academic-level suitability
5. Logical structure
6. Language quality

Return ONLY the final answer for the student.

Do not mention that multiple AI agents were used.
"""

    user_prompt = f"""
Academic level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}

Candidate answer:

{research_answer}

Improve it if necessary and return the final student-friendly answer.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


def run_ai_tutor(request_data, api_key, model):
    """
    Main multi-agent workflow:

    Tutor Agent
        ↓
    Research Agent
        ↓
    Evaluator Agent
        ↓
    Final Answer
    """

    client = create_client(api_key)

    tutor_answer = tutor_agent(
        client,
        model,
        request_data,
    )

    research_answer = research_agent(
        client,
        model,
        request_data,
        tutor_answer,
    )

    final_answer = evaluator_agent(
        client,
        model,
        request_data,
        research_answer,
    )

    return final_answer
