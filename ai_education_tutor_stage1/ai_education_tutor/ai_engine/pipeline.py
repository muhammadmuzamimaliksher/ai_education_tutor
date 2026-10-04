from .groq_client import create_client
from .history import validate_pdf_context
from .rag_policy import PDF_FALLBACK
from .tutor_agent import tutor_agent
from .research_agent import research_agent
from .evaluator_agent import evaluator_agent


def run_ai_tutor(
    request_data,
    api_key,
    model,
):
    """
    Main multi-agent AI workflow.

    Tutor
       ↓
    Research / Evidence Verification
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
    # PDF PRECHECK
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
    # AGENT 1
    # =====================================================

    tutor_answer = tutor_agent(
        client,
        model,
        request_data,
    )


    # =====================================================
    # AGENT 2
    # =====================================================

    research_report = research_agent(
        client,
        model,
        request_data,
        tutor_answer,
    )


    # =====================================================
    # AGENT 3
    # =====================================================

    final_answer = evaluator_agent(
        client,
        model,
        request_data,
        tutor_answer,
        research_report,
    )


    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if not final_answer:
        if mode == "PDF Question Answering":
            return PDF_FALLBACK

        return "I was unable to generate an answer."

    return final_answer.strip()
