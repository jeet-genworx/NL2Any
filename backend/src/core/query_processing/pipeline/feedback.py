"""Feedback coordination, sliding window retention, and guidance prompt composition."""

from typing import Any


def _get_stage_feedback_items(
    feedback_history: list[dict[str, Any]],
    current_feedback: str | None,
    stage: str,
) -> list[str]:
    """Extract a sliding window of at most 1 previous feedback item + current feedback item for a stage."""
    items: list[str] = []
    prior_for_stage = [
        str(h["feedback"]).strip()
        for h in feedback_history
        if isinstance(h, dict) and h.get("stage") == stage and h.get("feedback")
    ]
    if prior_for_stage:
        items.append(prior_for_stage[-1])
    if current_feedback and current_feedback.strip():
        cf = current_feedback.strip()
        if not items or items[-1] != cf:
            items.append(cf)
    return items


def _format_stage_user_guidance(
    feedback_history: list[dict[str, Any]],
    current_feedback: str | None,
    stage: str,
) -> str | None:
    """Format user guidance with a sliding window of at most 1 previous feedback item + current feedback item."""
    items = _get_stage_feedback_items(feedback_history, current_feedback, stage)
    if not items:
        return None

    joined_items = "\n".join(f"- {item}" for item in items)
    if stage == "table_selection":
        note = "(Note: User guidance is an advisory hint. You must strictly ground table selection in the canonical schema and adhere to all relationship rules.)"
    elif stage == "query_generation":
        note = "(Note: User guidance provides domain/intent clarification. All generated queries must strictly adhere to the Selected Schema and read-only safety rules.)"
    elif stage == "validation":
        note = "(Note: User validation guidance clarifies user intent or acceptable caveats. Ensure the query remains strictly read-only and free of syntax errors.)"
    else:
        note = "(Note: User guidance is an advisory hint.)"

    return f"[USER GUIDANCE / HINT]:\n{joined_items}\n{note}"


def build_cascading_guidance(
    feedback_history: list[dict[str, Any]],
    applicable_feedback: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Build cascading user guidance blocks for Stage 1, Stage 2, and Stage 3.

    Returns:
        (s1_user_guidance, s2_user_guidance, s3_val_guidance)
    """
    s1_items = _get_stage_feedback_items(
        feedback_history, applicable_feedback.get("table_selection"), "table_selection"
    )
    s2_items = _get_stage_feedback_items(
        feedback_history, applicable_feedback.get("query_generation"), "query_generation"
    )
    s3_items = _get_stage_feedback_items(
        feedback_history, applicable_feedback.get("validation"), "validation"
    )

    # Stage 1: Table Selection guidance
    if s1_items:
        joined_s1 = "\n".join(f"- {it}" for it in s1_items)
        s1_user_guidance = (
            f"[USER GUIDANCE / HINT]:\n{joined_s1}\n"
            "(Note: User guidance is an advisory hint. You must strictly ground table selection in the canonical schema and adhere to all relationship rules.)"
        )
    else:
        s1_user_guidance = None

    # Stage 2: Query Generation guidance (receives Stage 2 feedback plus active Stage 1 context)
    gen_guidance_items: list[str] = []
    if s1_items:
        gen_guidance_items.extend([f"Context from table selection: {it}" for it in s1_items])
    if s2_items:
        gen_guidance_items.extend(s2_items)

    if gen_guidance_items:
        joined_s2 = "\n".join(f"- {it}" for it in gen_guidance_items)
        s2_user_guidance = (
            f"[USER GUIDANCE / HINT]:\n{joined_s2}\n"
            "(Note: User guidance provides domain/intent clarification. All generated queries must strictly adhere to the Selected Schema and read-only safety rules.)"
        )
    else:
        s2_user_guidance = None

    # Stage 3: Validation guidance (receives Stage 3 feedback plus all applied query guidance)
    val_sections: list[str] = []
    applied_query_items: list[str] = []
    if s1_items:
        applied_query_items.extend([f"Context from table selection: {it}" for it in s1_items])
    if s2_items:
        applied_query_items.extend(s2_items)

    if applied_query_items:
        joined_applied = "\n".join(f"- {it}" for it in applied_query_items)
        val_sections.append(
            f"[APPLIED QUERY GUIDANCE]:\n{joined_applied}\n"
            "(Note: The above user guidance was explicitly requested by the user to scope query generation. All requested filters, status codes, grains, and constraints are authorized user refinements and MUST be accepted as correct.)"
        )
    if s3_items:
        joined_s3 = "\n".join(f"- {it}" for it in s3_items)
        val_sections.append(
            f"[USER VALIDATION GUIDANCE]:\n{joined_s3}\n"
            "(Note: User validation guidance clarifies user intent or acceptable caveats. Ensure the query remains strictly read-only and free of syntax errors.)"
        )

    s3_val_guidance = "\n\n".join(val_sections) if val_sections else None

    return s1_user_guidance, s2_user_guidance, s3_val_guidance


def compose_feedback_prompt(
    user_guidance: str | None,
    system_retry: str | None,
) -> str | None:
    """Compose user guidance and system retry feedback cleanly without recursive stacking."""
    if user_guidance and system_retry:
        return f"{user_guidance}\n\n{system_retry}"
    return user_guidance or system_retry
