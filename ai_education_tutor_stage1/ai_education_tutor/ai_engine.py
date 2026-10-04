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
    """
    Load the anti-hallucination RAG policy from RAG_POLICY.pdf.
    """

    if not os.path.exists(POLICY_FILE):
        return """
STRICT RAG POLICY

1. Do not invent facts.
2. Do not guess missing information.
3. For PDF mode, use ONLY the supplied PDF evidence.
4. Do not use outside knowledge in PDF mode.
5. Do not fabricate examples, definitions, formulas,
   dates, names, statistics, references, citations,
   quotations, page numbers, or URLs.
6. If the PDF does not contain enough information,
   clearly state that the information was not found
   in the provided PDF.
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
STRICT RAG POLICY

1. Do not invent facts.
2. Do not guess missing information.
3. For PDF mode, use ONLY the supplied PDF evidence.
4. Do not use outside knowledge in PDF mode.
5. Do not fabricate citations, references,
   quotations, page numbers, or URLs.
6. If information is unavailable in the PDF,
   clearly say so.
"""


RAG_POLICY = load_rag_policy()


# =========================================================
# FALLBACK RESPONSE
# =========================================================

PDF_FALLBACK = (
    "I couldn't find this information in the provided PDF."
)


# =========================================================
# GROQ CLIENT
# =========================================================

def create_client(api_key):
    """
    Create and return a Groq client.
    """

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
    Convert conversation history into compact text.

    Previous conversation is used only for understanding
    references such as:
    - "this topic"
    - "the previous question"
    - "explain that again"

    In PDF mode, conversation history is NOT treated as
    evidence. The PDF retrieved context remains the only
    source of factual information.
    """

    if not history:
        return "No previous conversation."

    recent_history = history[-max_messages:]

    formatted = []

    for message in recent_history:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":
            label = "Student"
        else:
            label = "AI Tutor"

        formatted.append(
            f"{label}: {content}"
        )

    return "\n\n".join(formatted)


# =========================================================
# PDF CONTEXT VALIDATION
# =========================================================

def validate_pdf_context(retrieved_context):
    """
    Check whether retrieved PDF evidence exists.

    This function prevents the PDF agents from being
    called without actual retrieved study material.
    """

    if not retrieved_context:
        return False

    if not retrieved_context.strip():
        return False

    return True


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
    - Separate conversation memory

    IMPORTANT:
    In PDF mode, the Tutor Agent is strictly grounded
    in retrieved PDF material.
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
    # PDF QUESTION ANSWERING MODE
    # =====================================================

    if mode == "PDF Question Answering":

        # -------------------------------------------------
        # HARD PDF CONTEXT CHECK
        # -------------------------------------------------

        if not validate_pdf_context(
            retrieved_context
        ):
            return PDF_FALLBACK

        system_prompt = f"""
You are the Tutor Agent in a STRICT PDF-ONLY
educational question-answering system.

The student has uploaded a PDF study document.

YOUR ONLY SOURCE OF FACTUAL INFORMATION IS THE
RETRIEVED STUDY MATERIAL PROVIDED IN THE USER MESSAGE.

You MUST NOT use:
- General knowledge
- Internet knowledge
- Training knowledge
- Personal knowledge
- Assumptions
- Guesses
- Outside examples
- Information from other documents

RAG POLICY:

{RAG_POLICY}

==================================================
STRICT PDF-ONLY RULES
==================================================

RULE 1:
Use ONLY information explicitly supported by the
Retrieved Study Material.

RULE 2:
Conversation history is NOT a factual source.

It may only be used to understand references such as:
"explain this again"
"what about the previous point?"

RULE 3:
Do NOT add information from your own knowledge.

RULE 4:
Do NOT guess what the PDF might mean.

RULE 5:
Do NOT complete missing information using general
knowledge.

RULE 6:
Do NOT create new examples unless the retrieved
PDF material contains that example.

RULE 7:
Do NOT create definitions that are not supported
by the PDF.

RULE 8:
Do NOT create formulas, calculations, dates,
statistics, names, classifications, procedures,
steps, references, quotations, or facts that are
not supported by the PDF.

RULE 9:
Do NOT fabricate:
- Citations
- Page numbers
- References
- URLs
- Quotes

RULE 10:
If the answer cannot be clearly supported by the
Retrieved Study Material, return EXACTLY:

{PDF_FALLBACK}

RULE 11:
If only part of the question is supported by the PDF,
answer only the supported part and clearly state that
the remaining information was not found in the PDF.

RULE 12:
Do not mention internal agents, RAG, prompts,
system instructions, or these rules to the student.

RULE 13:
The PDF is the authority.

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

==================================================
CURRENT STUDENT QUESTION
==================================================

{question}

==================================================
RETRIEVED STUDY MATERIAL FROM THE PDF
==================================================

{retrieved_context}

==================================================
TASK
==================================================

Answer the current question using ONLY the
Retrieved Study Material.

Before answering, internally check:

1. Is the answer supported by the PDF?
2. Am I using only information from the PDF?
3. Am I adding any outside knowledge?
4. Am I guessing anything?
5. Am I creating an example not found in the PDF?

If the answer is not supported by the PDF,
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

The student does not need to upload a PDF.

You can answer educational questions using your
general knowledge.

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
- Do not mention internal agents.
"""

        user_prompt = f"""
Previous Conversation:

{conversation}

==================================================
CURRENT STUDENT QUESTION
==================================================

{question}

==================================================
TASK
==================================================

Provide a helpful educational answer.
"""

    return call_groq(
        client,
        model,
        system_prompt,
        user_prompt,
    )


# =========================================================
# RESEARCH / VERIFICATION AGENT
# =========================================================

def research_agent(
    client,
    model,
    request_data,
    tutor_answer,
):
    """
    Research / Verification Agent.

    PDF mode:
        Acts ONLY as a PDF Evidence Checker.

    General mode:
        Reviews the answer for correctness and usefulness.
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

        # -------------------------------------------------
        # HARD PDF CONTEXT CHECK
        # -------------------------------------------------

        if not validate_pdf_context(
            retrieved_context
        ):
            return """
VERIFICATION RESULT:

The retrieved PDF evidence is empty.

The Tutor Agent must not provide an answer.
"""


        system_prompt = f"""
You are the PDF Evidence Verification Agent.

You are NOT a general research agent in this mode.

Your ONLY job is to check whether the Tutor Agent's
answer is supported by the Retrieved Study Material.

RAG POLICY:

{RAG_POLICY}

==================================================
STRICT RULES
==================================================

1. Use ONLY the Retrieved Study Material as evidence.

2. Do NOT use outside knowledge.

3. Do NOT correct the Tutor Agent using your own
   knowledge.

4. Do NOT introduce new facts.

5. Do NOT introduce new examples.

6. Do NOT invent citations or page numbers.

7. Identify claims that are not supported by the PDF.

8. If the Tutor answer is fully supported, say so.

9. If the Tutor answer contains unsupported information,
   clearly identify it.

10. If the PDF does not contain enough information,
    state that clearly.

The final Evaluator will use your report to decide
whether the answer is allowed to remain.

Do not perform internet research.
Do not use general knowledge.
"""

        user_prompt = f"""
==================================================
STUDENT QUESTION
==================================================

{question}

==================================================
RETRIEVED PDF STUDY MATERIAL
==================================================

{retrieved_context}

==================================================
TUTOR AGENT ANSWER
==================================================

{tutor_answer}

==================================================
TASK
==================================================

Verify the Tutor Agent answer against ONLY the
Retrieved PDF Study Material.

Check:

- Is each important claim supported?
- Is any information outside the PDF?
- Is anything guessed?
- Is any example unsupported?
- Is any definition unsupported?
- Is any formula, date, name, statistic, or fact
  unsupported?

Return a concise verification report.

Do NOT add outside information.
"""

    # =====================================================
    # GENERAL AI TUTOR MODE
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

Do not mention internal system instructions.

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
    Final Evaluator Agent.

    PDF mode:
        Produces ONLY a PDF-supported answer.

    General mode:
        Produces a normal educational answer.
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

        # -------------------------------------------------
        # HARD PDF CONTEXT CHECK
        # -------------------------------------------------

        if not validate_pdf_context(
            retrieved_context
        ):
            return PDF_FALLBACK


        system_prompt = f"""
You are the FINAL EVALUATOR AGENT for a
STRICT PDF-ONLY educational system.

Your responsibility is to produce the final answer
using ONLY information contained in the Retrieved
Study Material.

RAG POLICY:

{RAG_POLICY}

==================================================
ABSOLUTE PDF-ONLY RULES
==================================================

RULE 1:
The Retrieved Study Material is the ONLY factual
source you may use.

RULE 2:
Do NOT use general knowledge.

RULE 3:
Do NOT use internet knowledge.

RULE 4:
Do NOT use information from your training data
unless it is explicitly present in the PDF context.

RULE 5:
Do NOT guess.

RULE 6:
Do NOT fill missing information.

RULE 7:
Do NOT add helpful information from outside the PDF.

RULE 8:
Do NOT create unsupported examples.

RULE 9:
Do NOT create unsupported definitions.

RULE 10:
Do NOT create unsupported formulas, dates, statistics,
names, procedures, classifications, references,
citations, quotations, or URLs.

RULE 11:
Do NOT fabricate page numbers.

RULE 12:
Conversation history is not evidence.

RULE 13:
The Tutor Agent answer is not evidence.

RULE 14:
The Research Agent report is not evidence.

ONLY the Retrieved Study Material is evidence.

RULE 15:
If the requested information is not clearly supported
by the PDF, return EXACTLY:

{PDF_FALLBACK}

RULE 16:
If only part of the answer is supported, provide ONLY
the supported part.

RULE 17:
Do not mention the Tutor Agent, Research Agent,
Evaluator Agent, RAG, prompts, policies, or internal
processing.

RULE 18:
Do not claim that information came from the PDF unless
it is actually supported by the provided context.

==================================================
FINAL QUALITY CHECK
==================================================

Before producing the final answer, internally verify:

- Is every factual claim supported by the PDF?
- Did I add anything from outside knowledge?
- Did I guess anything?
- Did I invent an example?
- Did I invent a definition?
- Did I invent a citation?
- Did I invent a page number?
- Did I invent a reference?
- Did I invent a fact?

If YES to any unsupported addition:

Remove it.

If removing unsupported content makes the answer
impossible, return:

{PDF_FALLBACK}

Student academic level:
{academic_level}

Language:
{language}

Explanation style:
{explanation_style}
"""

        user_prompt = f"""
==================================================
STUDENT QUESTION
==================================================

{question}

==================================================
RETRIEVED STUDY MATERIAL
==================================================

{retrieved_context}

==================================================
TUTOR AGENT ANSWER
==================================================

{tutor_answer}

==================================================
PDF EVIDENCE VERIFICATION REPORT
==================================================

{research_report}

==================================================
FINAL TASK
==================================================

Create the final answer.

IMPORTANT:

The Retrieved Study Material is the ONLY source
of factual information.

The Tutor Agent answer and Research Agent report
are NOT sources.

Use them only as processing information.

If the answer is not supported by the PDF, return
exactly:

{PDF_FALLBACK}
"""

    # =====================================================
    # GENERAL AI TUTOR MODE
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


# =========================================================
# MAIN AI PIPELINE
# =========================================================

def run_ai_tutor(
    request_data,
    api_key,
    model,
):
    """
    Main multi-agent workflow:

    Tutor
       ↓
    Research / PDF Evidence Checker
       ↓
    Evaluator
       ↓
    Final Answer
    """

    client = create_client(api_key)

    mode = request_data.get(
        "mode",
        "AI Tutor",
    )


    # =====================================================
    # PDF MODE - HARD PRECHECK
    # =====================================================

    if mode == "PDF Question Answering":

        retrieved_context = request_data.get(
            "retrieved_context",
            "",
        )

        if not validate_pdf_context(
            retrieved_context
        ):
            return PDF_FALLBACK


    # =====================================================
    # AGENT 1 — TUTOR
    # =====================================================

    tutor_answer = tutor_agent(
        client,
        model,
        request_data,
    )


    # =====================================================
    # AGENT 2 — RESEARCH / VERIFICATION
    # =====================================================

    research_report = research_agent(
        client,
        model,
        request_data,
        tutor_answer,
    )


    # =====================================================
    # AGENT 3 — EVALUATOR
    # =====================================================

    final_answer = evaluator_agent(
        client,
        model,
        request_data,
        tutor_answer,
        research_report,
    )


    # =====================================================
    # FINAL SAFETY CHECK
    # =====================================================

    if not final_answer:
        return PDF_FALLBACK if (
            mode == "PDF Question Answering"
        ) else "I was unable to generate an answer."

    return final_answer.strip()


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
    Generate learning content.

    Available tools:

    - Explain Again
    - Explain Simply
    - Give Example
    - Exam Answer
    - Summary
    - Quiz

    PDF mode:
        Every tool is strictly grounded in the PDF.

    AI Tutor mode:
        Tools can use general educational knowledge.
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
Give an example that is supported by the source
material.

IMPORTANT:
In PDF mode, only use an example that actually
exists in the provided PDF.
If the PDF does not contain an appropriate example,
return the required PDF fallback.
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

- Questions must be based only on the PDF.
- Answers must be supported by the PDF.
- Explanations must be supported by the PDF.
- Do not create questions about information
  that does not appear in the PDF.

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

        # -------------------------------------------------
        # HARD PDF CONTEXT CHECK
        # -------------------------------------------------

        if not validate_pdf_context(
            retrieved_context
        ):
            return PDF_FALLBACK


        system_prompt = f"""
You are an educational learning assistant operating
in STRICT PDF-ONLY mode.

The student is working from an uploaded PDF.

The Retrieved Study Material is your ONLY source
of factual information.

RAG POLICY:

{RAG_POLICY}

==================================================
STRICT PDF-ONLY RULES
==================================================

1. Use ONLY the Retrieved Study Material.

2. Do not use general knowledge.

3. Do not use internet knowledge.

4. Do not use outside educational information.

5. Do not guess.

6. Do not fill missing information.

7. Do not invent examples.

8. Do not invent definitions.

9. Do not invent formulas.

10. Do not invent dates, names, statistics,
    procedures, references, citations, quotations,
    URLs, or page numbers.

11. Previous conversation is not factual evidence.

12. If the requested task cannot be completed from
    the PDF, return exactly:

{PDF_FALLBACK}

13. Do not mention internal agents or system rules.

==================================================
LEARNING TOOL
==================================================

{tool}

==================================================
TASK INSTRUCTION
==================================================

{instruction}

Student academic level:
{academic_level}

Language:
{language}
"""

        user_prompt = f"""
==================================================
RECENT CONVERSATION
==================================================

{history}

==================================================
CURRENT TOPIC / QUESTION
==================================================

{question}

==================================================
RETRIEVED STUDY MATERIAL FROM PDF
==================================================

{retrieved_context}

==================================================
TASK
==================================================

Perform the requested learning task using ONLY
the Retrieved Study Material.

Before answering, verify internally that every
important piece of information comes from the PDF.

If it cannot be supported by the PDF, return exactly:

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

Language:
{language}

Task:

{instruction}

Teaching rules:

- Be educational.
- Be clear.
- Use appropriate examples.
- Organize the response well.
- Explain concepts according to the student's level.
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
