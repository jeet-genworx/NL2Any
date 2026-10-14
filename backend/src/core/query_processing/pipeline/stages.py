"""Which model provider runs which stage.

A request can point table selection and query generation at different models,
because they are different jobs: selection reads a whole schema neighborhood
and answers with a table list, while generation writes one query and reacts to
validation feedback. This module turns the request's provider arguments into
the set of stage instances a run uses.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.factory import resolve_model_provider
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.core.query_processing.pipeline.semantic import SemanticAnalyzer
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.data.models.schema import DatabaseType

DEFAULT_PROVIDER_NAME = "koboldcpp"


class StageSet(NamedTuple):
    """The provider-bound stages one run uses, in pipeline order."""

    guardrail: GuardrailClassifier
    semantic: SemanticAnalyzer
    selector: TableSelector
    planner: QueryPlanner
    postgres_gen: PostgresQueryGenerator
    mongo_gen: MongoQueryGenerator
    validator: QueryValidator

    def generator_for(self, db_type: DatabaseType) -> Any:
        """The query generator for this engine."""
        return self.postgres_gen if db_type == DatabaseType.POSTGRESQL else self.mongo_gen


def resolve_provider(
    provider: str | ModelProvider | None,
) -> tuple[str, ModelProvider] | tuple[None, None]:
    """Turn a request's provider argument into (name, instance).

    Returns (None, None) when nothing was asked for, which is how a
    stage-specific override says "fall back to the default".
    """
    if provider is None:
        return (None, None)
    if isinstance(provider, str):
        name = provider.strip().lower()
        if not name:
            return (None, None)
        return (name, resolve_model_provider(name))
    return (getattr(provider, "model", "custom"), provider)


class ProviderPlan(NamedTuple):
    """The providers a run resolved to, plus the names to report back."""

    name: str
    default: ModelProvider
    selector: ModelProvider | None
    generator: ModelProvider | None
    provider_map: dict[str, str]


def plan_providers(
    fallback: ModelProvider,
    provider: str | ModelProvider | None = None,
    selector_provider: str | ModelProvider | None = None,
    generator_provider: str | ModelProvider | None = None,
) -> ProviderPlan:
    """Resolve a request's three provider arguments in one go.

    `provider_map` is reported on the response rather than inferred by the
    caller, so a response says which model wrote the query it is carrying even
    when the stages differ.
    """
    default_name, resolved_default = resolve_provider(provider)
    provider_name = default_name or DEFAULT_PROVIDER_NAME
    sel_name, sel_obj = resolve_provider(selector_provider)
    gen_name, gen_obj = resolve_provider(generator_provider)
    return ProviderPlan(
        name=provider_name,
        default=resolved_default or fallback,
        selector=sel_obj,
        generator=gen_obj,
        provider_map={
            "default": provider_name,
            "selector": sel_name or provider_name,
            "generator": gen_name or provider_name,
        },
    )


class StageBinder:
    """Binds each stage to its provider, honouring constructor overrides.

    A stage that was injected into the orchestrator is always that instance,
    whatever the request asked for: an explicitly supplied stage is a deliberate
    override, and tests rely on it. The defaults are already bound to the
    orchestrator's own provider, so they are reused rather than rebuilt whenever
    that is the stage's provider.
    """

    def __init__(
        self,
        provider: ModelProvider,
        defaults: StageSet,
        injected: dict[str, bool],
    ) -> None:
        self.provider = provider
        self.defaults = defaults
        self.injected = injected

    _FACTORIES: dict[str, Any] = {
        "guardrail": GuardrailClassifier,
        "semantic": SemanticAnalyzer,
        "selector": TableSelector,
        "planner": QueryPlanner,
        "postgres_gen": PostgresQueryGenerator,
        "mongo_gen": MongoQueryGenerator,
        "validator": QueryValidator,
    }

    # The stages that read the question run on the default provider; selection
    # and the generation loop can each be pointed somewhere else.
    _ROLES: dict[str, str] = {
        "guardrail": "default",
        "semantic": "default",
        "selector": "selector",
        "planner": "generator",
        "postgres_gen": "generator",
        "mongo_gen": "generator",
        "validator": "generator",
    }

    def bind(
        self,
        active_provider: ModelProvider,
        selector_provider: ModelProvider | None = None,
        generator_provider: ModelProvider | None = None,
    ) -> StageSet:
        """Return stage instances, each bound to the provider chosen for its stage.

        `active_provider` is the default. Either stage-specific provider may be
        left unset, which means "use the default".
        """
        by_role = {
            "default": active_provider,
            "selector": selector_provider or active_provider,
            "generator": generator_provider or active_provider,
        }
        bound = []
        for name in StageSet._fields:
            stage = getattr(self.defaults, name)
            target = by_role[self._ROLES[name]]
            if self.injected.get(name) or target is self.provider:
                bound.append(stage)
            else:
                bound.append(self._FACTORIES[name](provider=target))
        return StageSet(*bound)
