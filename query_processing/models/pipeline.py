"""Pipeline Pydantic models for NL2AnyQuery Part 2."""

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field as PydanticField

from query_processing.models.schema import DatabaseType, Relationship, SchemaObject
from query_processing.nlp.linguistic import LinguisticEntity


class GuardrailDecision(str, Enum):
    """Guardrail classification decisions."""

    READ_QUERY = "READ_QUERY"
    BASIC = "BASIC"
    REJECT = "REJECT"


class GuardrailResult(BaseModel):
    """Result of Guardrail classification."""

    decision: GuardrailDecision
    reason: str = ""


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

    def to_retrieval_query(self) -> str:
        """Compose search text for semantic schema candidate retrieval."""
        parts = [self.question]
        parts.extend(self.subjective)
        parts.extend(self.objective)
        parts.extend(self.nouns)
        for ent in self.entities:
            parts.append(ent.text)
        return " ".join(parts)


class CandidateTable(BaseModel):
    """Candidate table retrieved via cosine similarity."""

    table_name: str
    similarity: float
    rank: int


class EmbeddingRetrievalResult(BaseModel):
    """Result of embedding-based candidate retrieval."""

    candidates: list[CandidateTable] = PydanticField(default_factory=list)


class TableSelectionResult(BaseModel):
    """Result of Table Selector SLM."""

    selected_objects: list[str] = PydanticField(default_factory=list)
    sufficient: bool = True
    missing_objects: list[str] = PydanticField(default_factory=list)
    reason: str = ""
    retrieval_hint: str | None = None


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


class ValidationResult(BaseModel):
    """Result of query validation stage."""

    valid: bool
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

    database: str
    question: str
    guardrail: GuardrailResult
    basic_answer: str | None = None
    semantic_analysis: SemanticAnalysisResult | None = None
    linguistic_analysis: dict[str, Any] | None = None
    candidate_objects: list[str] = PydanticField(default_factory=list)
    candidate_tables: list[CandidateTable] = PydanticField(default_factory=list)
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
