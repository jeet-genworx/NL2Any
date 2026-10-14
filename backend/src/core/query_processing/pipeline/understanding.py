"""Everything the pipeline does with the question before it touches the schema.

Four stages that only read the question: the guardrail decides whether it is a
database question at all, spelling correction fixes it, and then the embedding,
the semantic SLM and spaCy read the corrected text concurrently -- three
independent views of one sentence, so there is no reason to run them in series.

Two of these stages can end the run on their own, which is why the result
carries either an `analysis` for the schema stages to use or the reason there
is none.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from backend.src.control.providers.embedding.base import EmbeddingProvider
from backend.src.core.query_processing.nlp.linguistic import LinguisticAnalyzer
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.semantic import SemanticAnalyzer
from backend.src.core.query_processing.pipeline.spelling import SpellingChecker
from backend.src.schemas.pipeline import (
    GuardrailDecision,
    GuardrailResult,
    QuestionAnalysis,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class Understanding:
    """What the question-reading stages produced, or why they stopped the run."""

    guardrail: GuardrailResult
    spelling: SpellingCorrectionResult | None = None
    semantic: SemanticAnalysisResult | None = None
    linguistic: Any = None
    analysis: QuestionAnalysis | None = None
    processed_question: str = ""
    query_vector: list[float] = field(default_factory=list)
    basic_answer: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """True when the run ends here: rejected, answered, or unreadable."""
        return self.analysis is None

    def response_fields(self) -> dict[str, Any]:
        """What every later response repeats back about the question."""
        return {
            "guardrail": self.guardrail,
            "spelling_correction": self.spelling,
            "semantic": self.semantic,
            "linguistic": self.linguistic.model_dump() if self.linguistic else None,
        }

    def terminal_fields(self) -> dict[str, Any]:
        """The full response for a run that ends in this phase."""
        return {
            **self.response_fields(),
            "basic_answer": self.basic_answer,
            "error": self.error,
        }


async def understand_question(
    *,
    question: str,
    target_key: str,
    database_name: str,
    guardrail: GuardrailClassifier,
    semantic: SemanticAnalyzer,
    linguistic: LinguisticAnalyzer,
    spelling_checker: SpellingChecker,
    embedding_provider: EmbeddingProvider,
    jargons: list[str] | None = None,
) -> Understanding:
    """Run the guardrail, spelling and analysis stages over the raw question."""
    # Stage 1: guardrail
    guardrail_result = await guardrail.classify(question)
    if guardrail_result.decision == GuardrailDecision.REJECT:
        return Understanding(
            guardrail=guardrail_result,
            error="Sorry, I can't help with this.",
        )

    if guardrail_result.decision == GuardrailDecision.BASIC:
        return Understanding(
            guardrail=guardrail_result,
            basic_answer=guardrail.handle_basic_question(
                question=question,
                database_type=target_key,
                database_name=database_name,
            ),
        )

    # Stage 1.5: SymSpell spelling checker, before semantic analysis so every
    # later stage reads the same corrected question.
    spelling_res = spelling_checker.check(question=question, jargons=jargons)
    processed_question = spelling_res.corrected_question

    # Stages 2 & 3: concurrent embedding, semantic SLM and spaCy analysis.
    try:
        query_vector, semantic_res, linguistic_res = await asyncio.gather(
            embedding_provider.embed(processed_question),
            semantic.analyze(processed_question),
            asyncio.to_thread(linguistic.analyze, processed_question),
        )
    except Exception as err:
        return Understanding(
            guardrail=guardrail_result,
            spelling=spelling_res,
            error=f"Failed to analyze query: {err}",
        )

    return Understanding(
        guardrail=guardrail_result,
        spelling=spelling_res,
        semantic=semantic_res,
        linguistic=linguistic_res,
        processed_question=processed_question,
        query_vector=query_vector,
        analysis=QuestionAnalysis(
            question=processed_question,
            subjective=semantic_res.subjective,
            objective=semantic_res.objective,
            nouns=linguistic_res.nouns,
            verbs=linguistic_res.verbs,
            entities=linguistic_res.entities,
        ),
    )
