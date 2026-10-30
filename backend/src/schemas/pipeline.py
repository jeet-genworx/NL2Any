"""Pipeline Pydantic models for NL2AnyQuery Part 2."""

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field as PydanticField, model_validator

from backend.src.data.models.schema import DatabaseType, Relationship, SchemaObject


class LinguisticEntity(BaseModel):
    """Named entity extracted from text."""

    text: str
    label: str


class GuardrailDecision(str, Enum):
    """Guardrail classification decisions."""

    READ_QUERY = "READ_QUERY"
    BASIC = "BASIC"
    REJECT = "REJECT"


class GuardrailResult(BaseModel):
    """Result of Guardrail classification."""

    decision: GuardrailDecision
    reason: str = ""


class SpellingCorrectionItem(BaseModel):
    """Single token spelling correction entry."""

    original: str
    corrected: str


class SpellingCorrectionResult(BaseModel):
    """Result of SymSpell spelling checking stage."""

    original_question: str
    corrected_question: str
    corrections: list[SpellingCorrectionItem] = PydanticField(default_factory=list)
    jargons_applied: list[str] = PydanticField(default_factory=list)


class SemanticAnalysisResult(BaseModel):
    """Subjective and objective concepts extracted from user question."""

    subjective: list[str] = PydanticField(default_factory=list)
    objective: list[str] = PydanticField(default_factory=list)


class QuestionAnalysis(BaseModel):
    """Combined question analysis joining semantic SLM and spaCy linguistic data."""

    question: str
    subjective: list[str] = PydanticField(default_factory=list)
    objective: list[str] = PydanticField(default_factory=list)
    nouns: list[str] = PydanticField(default_factory=list)
    verbs: list[str] = PydanticField(default_factory=list)
    entities: list[LinguisticEntity] = PydanticField(default_factory=list)


class CandidateTable(BaseModel):
    """Candidate table retrieved via cosine similarity."""

    table_name: str
    similarity: float
    rank: int


class EmbeddingRetrievalResult(BaseModel):
    """Result of embedding-based candidate retrieval."""

    candidates: list[CandidateTable] = PydanticField(default_factory=list)


class NeighborhoodTable(BaseModel):
    """One schema object in the table neighborhood handed to the table selector.

    `level` is how many foreign-key hops away from the nearest embedding match
    the object sits: 0 means it *is* an embedding match, 1 a direct neighbor of
    one, and so on up to the traversal limit. `reached_from` names the embedding
    matches whose breadth-first walk discovered it, which is what explains to
    the selector why a table it never searched for is on the list at all.
    """

    name: str
    description: str = ""
    kind: str = "table"
    level: int = 0
    similarity: float | None = None
    rank: int | None = None
    reached_from: list[str] = PydanticField(default_factory=list)


class TableNeighborhood(BaseModel):
    """Embedding matches plus every table reachable from them within N hops.

    The candidate space the table selector chooses from when the BFS strategy
    is active: the embedding matches are the seeds, and breadth-first traversal
    of the foreign-key graph supplies the tables a question needs but never
    names -- the bridge tables, and the entity tables one hop past them.
    """

    seed_objects: list[str] = PydanticField(default_factory=list)
    tables: list[NeighborhoodTable] = PydanticField(default_factory=list)
    relationships: list[Relationship] = PydanticField(default_factory=list)
    max_levels: int = 0
    max_tables: int = 0
    truncated: bool = False
    discarded_objects: list[str] = PydanticField(default_factory=list)

    def table_names(self) -> list[str]:
        """Names of every table in the neighborhood, seeds first."""
        return [t.name for t in self.tables]


class TableSelectionResult(BaseModel):
    """Result of Table Selector SLM."""

    selected_objects: list[str] = PydanticField(default_factory=list)
    sufficient: bool = True
    missing_objects: list[str] = PydanticField(default_factory=list)
    reason: str = ""
    retrieval_hint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_tables_key(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "selected_tables" in data and "selected_objects" not in data:
                data["selected_objects"] = data["selected_tables"]
            if "missing_tables" in data and "missing_objects" not in data:
                data["missing_objects"] = data["missing_tables"]
            if "missing_information" in data and "missing_objects" not in data:
                missing_info = data["missing_information"]
                if isinstance(missing_info, list):
                    data["missing_objects"] = missing_info
                elif isinstance(missing_info, str) and missing_info:
                    data["missing_objects"] = [missing_info]
            # The prompt asks for prose here and the model sometimes answers
            # with its reasoning steps as a list instead. The tables it chose
            # are still valid, so the selection is kept rather than rejected
            # over the shape of a field nothing branches on.
            reason = data.get("reason")
            if isinstance(reason, list):
                data["reason"] = " ".join(str(part) for part in reason if part)
            elif reason is not None and not isinstance(reason, str):
                data["reason"] = str(reason)
        return data


class GeneratorMissingInfo(BaseModel):
    """Structured feedback from Query Generator when required schema information is missing."""

    status: Literal["missing_information"] = "missing_information"
    missing_tables: list[str] = PydanticField(default_factory=list)
    missing_columns: list[str] = PydanticField(default_factory=list)
    reason: str = ""


class RelevantSchema(BaseModel):
    """Subset of database schema relevant to the question."""

    objects: list[SchemaObject] = PydanticField(default_factory=list)
    relationships: list[Relationship] = PydanticField(default_factory=list)

    def get_object_names(self) -> set[str]:
        return {obj.name.lower() for obj in self.objects}

    def get_field_names_for_object(self, obj_name: str) -> set[str]:
        for obj in self.objects:
            if obj.name.lower() == obj_name.lower():
                paths = set(obj.all_field_paths())
                # Also include short leaf names for convenient matching
                for f in obj.fields:
                    paths.add(f.name)
                return paths
        return set()


class SemanticFilter(BaseModel):
    """Database-independent filter intent."""

    field: str
    operator: str = "equals"  # e.g., equals, not_equals, greater_than, less_than, in, like, between
    value: Any = None


class SemanticOrdering(BaseModel):
    """Database-independent ordering intent."""

    field: str
    direction: Literal["asc", "desc"] = "asc"


class SemanticAggregation(BaseModel):
    """Database-independent aggregation intent."""

    field: str | None = None
    function: str = "count"  # e.g., count, sum, avg, min, max
    alias: str | None = None


class QueryPlan(BaseModel):
    """Database-independent, semantic query plan containing NO raw SQL/Mongo syntax."""

    operation: str = "select"  # select, aggregate, find
    sources: list[str] = PydanticField(default_factory=list)
    projections: list[str] = PydanticField(default_factory=list)
    filters: list[SemanticFilter] = PydanticField(default_factory=list)
    aggregations: list[SemanticAggregation] = PydanticField(default_factory=list)
    group_by: list[str] = PydanticField(default_factory=list)
    order_by: list[SemanticOrdering] = PydanticField(default_factory=list)
    limit: int | None = None
    relationships_used: list[str] = PydanticField(default_factory=list)
    missing_tables: list[str] = PydanticField(default_factory=list)


class MongoQuery(BaseModel):
    """Strictly typed MongoDB query representation (no arbitrary JSON execution)."""

    operation: Literal["find", "aggregate"]
    collection: str
    filter: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    sort: list[list[Any]] | dict[str, int] | None = None
    limit: int | None = None
    pipeline: list[dict[str, Any]] | None = None


class GeneratedQuery(BaseModel):
    """Generated database query."""

    database_type: DatabaseType
    raw_query: str | MongoQuery
    formatted_query: str


class ValidationErrorType(str, Enum):
    """Validation error categories for targeted retry routing."""

    VALID = "VALID"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    PLANNER_ERROR = "PLANNER_ERROR"
    UNSAFE = "UNSAFE"
    GENERATION_ERROR = "GENERATION_ERROR"
    SEMANTIC_ERROR = "SEMANTIC_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"


class ValidationResult(BaseModel):
    """Result of query validation stage."""

    valid: bool
    verdict: str | None = None
    error_type: ValidationErrorType = ValidationErrorType.VALID
    issues: list[str] = PydanticField(default_factory=list)
    suggestion: str | None = None


class PolicyResult(BaseModel):
    """Result of deterministic read-only safety policy check."""

    allowed: bool
    reason: str = ""


class ExecutionResult(BaseModel):
    """Result of database query execution."""

    row_count: int
    columns: list[str] = PydanticField(default_factory=list)
    preview_rows: list[dict[str, Any]] = PydanticField(default_factory=list)
    truncated: bool = False
    csv_data: str | None = None


class PipelineResponse(BaseModel):
    """Comprehensive trace and response of the NL2AnyQuery pipeline."""

    run_id: str | None = PydanticField(
        default=None,
        description="Unique 32-char hex identifier for the pipeline run",
    )
    database: str
    question: str
    provider: str | None = None
    stage_providers: dict[str, str] | None = PydanticField(
        default=None,
        description="Resolved model provider for each stage: table_selection, query_generation, validation",
    )
    stage_tokens: dict[str, int] | None = PydanticField(
        default=None,
        description="Estimated token consumption per stage and total",
    )
    guardrail: GuardrailResult
    spelling_correction: SpellingCorrectionResult | None = None
    basic_answer: str | None = None
    semantic_analysis: SemanticAnalysisResult | None = None
    linguistic_analysis: dict[str, Any] | None = None
    candidate_objects: list[str] = PydanticField(default_factory=list)
    candidate_tables: list[CandidateTable] = PydanticField(default_factory=list)
    # Which view of the schema the table selector was shown, and why it may
    # not be the one that was asked for.
    table_selection_strategy: str = "whole_schema"
    table_neighborhood: TableNeighborhood | None = None
    selection_notice: str | None = None
    selected_objects: list[str] = PydanticField(default_factory=list)
    relevant_schema: dict[str, Any] | None = None
    query_plan: QueryPlan | None = None
    generated_query: str | dict[str, Any] | None = None
    attempts: int = 1
    selection_retries: int = 0
    validation_retries: int = 0
    validation: ValidationResult | None = None
    policy: PolicyResult | None = None
    results: ExecutionResult | None = None
    error: str | None = None
