"""Human-in-the-loop gate for table selection.

Everything downstream of the selector is scoped by the tables it picked: the
schema slice, the plan, the query. A wrong pick therefore does not produce an
error, it produces a confident answer to a different question, which is the one
failure the pipeline cannot catch on its own.

There is no confidence score to threshold on -- the provider returns text, not
logprobs, and the stage runs once at temperature zero -- so certainty is read
off evidence instead: the selector's own `sufficient` flag, and the repairs the
stage makes silently when the model misread the schema it was shown. When any
of that fires the run is parked here rather than carried on with a warning
attached, and a human settles which tables it may use.

This module holds the assessment and the parked runs. Resuming one is the
orchestrator's job, because only it knows what comes next.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.src.config import settings
from backend.src.schemas.pipeline import (
    GuardrailResult,
    QuestionAnalysis,
    SelectionUncertainty,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    TableSelectionResult,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle: the orchestrator imports this module
    from backend.src.core.query_processing.pipeline.orchestrator import _SelectionOutcome


# Reason codes. They are strings rather than an enum because they travel to the
# frontend on the response and are read there as data, not matched on.
INSUFFICIENT_AFTER_RETRIES = "insufficient_after_retries"
SELECTOR_ERROR_FALLBACK = "selector_error_fallback"
HALLUCINATED_TABLES = "hallucinated_tables"
UNJOINABLE_DROPPED = "unjoinable_dropped"


class PendingRunNotFound(KeyError):
    """A run id that was never issued, or whose run has already expired or resumed."""


def assess_selection(
    selection: TableSelectionResult,
    unjoinable: list[str],
    exhausted_retries: bool,
) -> SelectionUncertainty | None:
    """What, if anything, makes this selection worth a human's attention.

    Returns None for the ordinary case -- the model chose tables it was shown,
    they join, and it said so -- which is what keeps the gate off the hot path.
    """
    reasons: list[str] = []
    details: list[str] = []

    if exhausted_retries:
        reasons.append(INSUFFICIENT_AFTER_RETRIES)
        details.append(
            "The table selector never became confident the schema covers this question"
            + (f": {selection.reason}" if selection.reason else ".")
        )
        if selection.missing_objects:
            details.append(f"It reported missing: {', '.join(selection.missing_objects)}.")

    if selection.fallback:
        reasons.append(SELECTOR_ERROR_FALLBACK)
        details.append(
            "The table selector failed to answer and the closest embedding match was "
            "used instead, so nothing actually read the question against the schema."
        )

    if selection.discarded_objects:
        reasons.append(HALLUCINATED_TABLES)
        details.append(
            "The selector named table(s) that do not exist in the candidate set and "
            f"were discarded: {', '.join(selection.discarded_objects)}."
        )

    if unjoinable:
        reasons.append(UNJOINABLE_DROPPED)
        details.append(
            "Table(s) it chose were dropped because no foreign-key path connects them "
            f"to the rest of the query: {', '.join(unjoinable)}."
        )

    if not reasons:
        return None

    return SelectionUncertainty(
        reasons=reasons,
        detail=" ".join(details),
        selector_reason=selection.reason,
        missing_objects=list(selection.missing_objects),
        discarded_objects=list(selection.discarded_objects),
        unjoinable_objects=list(unjoinable),
        fallback=selection.fallback,
    )


@dataclass
class PendingRun:
    """A run stopped at the table gate, with everything needed to finish it.

    The providers are kept as the arguments the caller passed rather than as
    the bound stage objects, so resuming re-resolves them the same way the first
    call did and a run cannot finish against a provider the request never asked
    for. That includes the stage-specific ones: a run paused with selection on
    one model and generation on another resumes with exactly that split.
    """

    target: Any
    db_type: Any
    provider_name: str
    provider_arg: Any
    question: str
    processed_question: str
    schema: Any
    guardrail_result: GuardrailResult
    spelling_res: SpellingCorrectionResult | None
    semantic_res: SemanticAnalysisResult | None
    linguistic_res: Any
    analysis: QuestionAnalysis
    outcome: "_SelectionOutcome"
    uncertainty: SelectionUncertainty
    created_at: float = field(default_factory=time.time)
    provider_map: dict[str, str] = field(default_factory=dict)
    selector_provider_arg: Any = None
    generator_provider_arg: Any = None

    def expires_at(self, ttl_seconds: int | None = None) -> float:
        ttl = settings.table_approval_ttl_seconds if ttl_seconds is None else ttl_seconds
        return self.created_at + ttl


class PendingRunStore:
    """Runs waiting on a human, held in memory and expired on their own.

    In-process, which matches how the orchestrator itself is held -- a single
    module-level instance behind the API. Running the API under more than one
    worker would need this moved to shared storage, since the resume call can
    land on a process that never issued the id.
    """

    def __init__(self, ttl_seconds: int | None = None, max_entries: int = 64) -> None:
        self.ttl_seconds = (
            settings.table_approval_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        self.max_entries = max_entries
        self._runs: dict[str, PendingRun] = {}

    def _evict_expired(self, now: float | None = None) -> None:
        cutoff = time.time() if now is None else now
        for run_id in [
            run_id
            for run_id, run in self._runs.items()
            if run.expires_at(self.ttl_seconds) <= cutoff
        ]:
            del self._runs[run_id]

    def put(self, run: PendingRun) -> str:
        """Park a run and return the id a human answers with."""
        self._evict_expired()
        # A gate nobody answers must not be able to pin the neighborhood of
        # every unanswered question before it in memory.
        while len(self._runs) >= self.max_entries:
            oldest = min(self._runs, key=lambda k: self._runs[k].created_at)
            del self._runs[oldest]
        run_id = uuid.uuid4().hex
        self._runs[run_id] = run
        return run_id

    def get(self, run_id: str) -> PendingRun:
        """Look at a parked run without consuming it.

        An answer that turns out to be invalid should cost the person a typo,
        not the whole analysis sitting behind the gate, so a run is only taken
        once its answer has been accepted.
        """
        self._evict_expired()
        try:
            return self._runs[run_id]
        except KeyError:
            raise PendingRunNotFound(run_id) from None

    def pop(self, run_id: str) -> PendingRun:
        """Take a parked run, which may only be resumed once."""
        self._evict_expired()
        try:
            return self._runs.pop(run_id)
        except KeyError:
            raise PendingRunNotFound(run_id) from None

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._runs)
