import asyncio
import time
from typing import Any

from backend.src.core.query_processing.pipeline.fsm.adapters import StageAdapters
from backend.src.core.query_processing.pipeline.fsm.context import (
    ContextUpdates,
    PipelineContext,
)
from backend.src.core.query_processing.pipeline.fsm.states import PipelineState
from backend.src.core.query_processing.pipeline.fsm.transition import (
    calculate_transition_cap,
    transition,
)


class PipelineFSMRunner:
    """Drives the FSM loop across states until reaching a terminal state."""

    def __init__(self, adapters: StageAdapters) -> None:
        self.adapters = adapters

    async def run(
        self,
        context: PipelineContext,
        initial_state: PipelineState,
    ) -> PipelineContext:
        """Run the FSM starting from initial_state until reaching a terminal state."""
        state = initial_state
        cap = calculate_transition_cap(context.max_retries)
        transition_count = 0

        while not state.is_terminal:
            # 0. Notify stage started
            if context.on_stage_event:
                try:
                    res = context.on_stage_event(state.value, "started", None)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

            # 1. Run adapter
            t0 = time.perf_counter()
            event, tokens = await self.adapters.execute(state, context)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Record event artifacts
            context.record_event_artifacts(event)

            # 2b. Notify stage completed
            if context.on_stage_event:
                summary: dict[str, Any] = {}
                try:
                    if state == PipelineState.SPELLING_CHECK and context.spelling_result:
                        summary["has_changes"] = bool(context.spelling_result.corrections)
                    elif state == PipelineState.QUESTION_GUARDRAIL and context.guardrail_result:
                        decision_val = getattr(
                            context.guardrail_result.decision,
                            "value",
                            str(context.guardrail_result.decision),
                        )
                        summary["decision"] = decision_val
                    elif state == PipelineState.FEEDBACK_GUARDRAIL:
                        summary["passed"] = True
                    elif state == PipelineState.TABLE_SELECTION:
                        summary["selected_objects"] = list(context.selected_objects)
                    elif state == PipelineState.QUERY_GENERATION:
                        summary["has_query"] = bool(context.generated_query)
                    elif state == PipelineState.QUERY_VALIDATION and context.validation_result:
                        summary["is_valid"] = context.validation_result.valid
                    elif state == PipelineState.POLICY_CHECK and context.policy_result:
                        summary["passed"] = context.policy_result.allowed
                    elif state == PipelineState.QUERY_EXECUTION and context.execution_result:
                        summary["row_count"] = context.execution_result.row_count
                except Exception:
                    pass

                try:
                    res = context.on_stage_event(state.value, "completed", summary)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

            # 3. Record tokens
            context.record_tokens(state, tokens)

            # 4. Accumulate latency
            context.accum_stage_latency_ms[state.value] += latency_ms

            # 5. Call transition
            decision = transition(state, event, context)

            # 6. Apply updates
            context.apply_updates(decision.updates)

            # 7. Advance state
            state = decision.next_state

            # 8. Enforce the transition cap
            transition_count += 1
            if transition_count > cap:
                context.apply_updates(
                    ContextUpdates(error="Pipeline exceeded maximum transition safety cap.")
                )
                state = PipelineState.TERMINAL_ERROR
                break

        return context
