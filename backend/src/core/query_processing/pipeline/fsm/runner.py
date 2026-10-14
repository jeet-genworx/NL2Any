"""Generic FSM runner driving stage execution, telemetry, and deterministic transitions."""

import time

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
            # 1. Run adapter
            t0 = time.perf_counter()
            event, tokens = await self.adapters.execute(state, context)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Record event artifacts
            context.record_event_artifacts(event)

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
