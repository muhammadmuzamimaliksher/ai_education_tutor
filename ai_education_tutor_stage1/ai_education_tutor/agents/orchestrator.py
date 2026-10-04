# =========================================================
# MULTI-AGENT ORCHESTRATOR
# =========================================================

from .planner_agent import run_planner_agent
from .tutor_agent import run_tutor_agent
from .reviewer_agent import run_reviewer_agent
from .final_answer_agent import run_final_answer_agent


def run_multi_agent(
    question,
    context="",
    api_key=None,
    model=None,
):
    """
    Multi-Agent Educational Workflow

    Planner
        ↓
    Tutor
        ↓
    Reviewer
        ↓
    Final Answer
    """

    if not question:
        raise ValueError("Multi-agent workflow received an empty question.")

    # -----------------------------------------------------
    # STEP 1: PLANNER
    # -----------------------------------------------------

    plan = run_planner_agent(
        question=question,
        context=context,
        api_key=api_key,
        model=model,
    )

    # -----------------------------------------------------
    # STEP 2: TUTOR
    # -----------------------------------------------------

    tutor_answer = run_tutor_agent(
        question=question,
        plan=plan,
        context=context,
        api_key=api_key,
        model=model,
    )

    # -----------------------------------------------------
    # STEP 3: REVIEWER
    # -----------------------------------------------------

    review = run_reviewer_agent(
        question=question,
        tutor_answer=tutor_answer,
        context=context,
        api_key=api_key,
        model=model,
    )

    # -----------------------------------------------------
    # STEP 4: FINAL ANSWER
    # -----------------------------------------------------

    final_answer = run_final_answer_agent(
        question=question,
        tutor_answer=tutor_answer,
        review=review,
        context=context,
        api_key=api_key,
        model=model,
    )

    # -----------------------------------------------------
    # RETURN COMPLETE WORKFLOW
    # -----------------------------------------------------

    return {
        "question": question,
        "plan": plan,
        "tutor_answer": tutor_answer,
        "review": review,
        "final_answer": final_answer,
    }
