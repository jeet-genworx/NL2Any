"""Query processing pipeline stages package.

The stages, in the order a run uses them: `guardrail`, `spelling`, `semantic`,
`selection` (retrieval, neighborhood, table choice), `approval` (the
human-in-the-loop gate), `materialization`, `planning`, `generation`, `policy`,
`executor`, `results`. `orchestrator` sequences them, `stages` binds each to
its model provider, `resources` caches what they read per database, and
`response` assembles the one shape every exit returns.
"""

from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.core.query_processing.pipeline.expansion import SchemaExpander
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.policy import SafetyPolicyValidator
from backend.src.core.query_processing.pipeline.resources import SchemaResources
from backend.src.core.query_processing.pipeline.response import build_response
from backend.src.core.query_processing.pipeline.results import ResultProcessor
from backend.src.core.query_processing.pipeline.selection import SelectionOutcome
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.core.query_processing.pipeline.semantic import SemanticAnalyzer
from backend.src.core.query_processing.pipeline.stages import StageSet
from backend.src.core.query_processing.pipeline.validator import QueryValidator

__all__ = [
    "GuardrailClassifier",
    "NL2AnyQueryOrchestrator",
    "QueryExecutor",
    "QueryPlanner",
    "QueryValidator",
    "ResultProcessor",
    "SafetyPolicyValidator",
    "SchemaExpander",
    "SchemaResources",
    "SelectionOutcome",
    "SemanticAnalyzer",
    "StageSet",
    "TableSelector",
    "build_response",
]
