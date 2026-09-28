"""Tests for selectable database targets.

The point of a target is that `postgres` and `finops` share the PostgreSQL
engine but are different databases. Anything that would make them collide --
one schema file, one embeddings file, one DSN -- is a bug these tests guard.
"""

import pytest

from backend.src.config import settings
from backend.src.data.clients.factory import get_adapter
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import (
    FINOPS_TARGET,
    MONGO_TARGET,
    POSTGRES_TARGET,
    TARGETS,
    as_target,
    default_target_for_type,
    resolve_target,
)
from backend.src.data.repositories import paths


def test_finops_is_a_selectable_target():
    assert "finops" in TARGETS
    assert FINOPS_TARGET.db_type == DatabaseType.POSTGRESQL
    assert FINOPS_TARGET.has_mst is True


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("postgres", "postgres"),
        ("postgresql", "postgres"),
        ("mongo", "mongo"),
        ("mongodb", "mongo"),
        ("finops", "finops"),
        ("finopsiq", "finops"),
        ("finopsiq_be", "finops"),
        ("  FinOps  ", "finops"),
    ],
)
def test_resolve_target_accepts_known_spellings(spelling, expected):
    assert resolve_target(spelling).key == expected


def test_resolve_target_rejects_unknown_and_names_the_valid_keys():
    with pytest.raises(ValueError) as err:
        resolve_target("nope")
    assert "finops" in str(err.value)


def test_finops_does_not_share_artifacts_with_the_demo_postgres_database():
    """The whole reason targets exist: two PostgreSQL databases, separate files."""
    for resolver in (paths.schema_path, paths.graph_path, paths.mst_path,
                     paths.descriptions_path, paths.embeddings_path):
        assert resolver(FINOPS_TARGET) != resolver(POSTGRES_TARGET), resolver.__name__
    assert paths.schema_path(FINOPS_TARGET).name == "finops.toml"
    assert paths.embeddings_path(FINOPS_TARGET).name == "finops_embeddings.json"


def test_finops_uses_its_own_dsn(monkeypatch):
    monkeypatch.setattr(settings, "postgres_dsn", "postgresql://demo/one")
    monkeypatch.setattr(settings, "finops_dsn", "postgresql://finops/two")

    assert POSTGRES_TARGET.connection == "postgresql://demo/one"
    assert FINOPS_TARGET.connection == "postgresql://finops/two"
    assert get_adapter(FINOPS_TARGET).dsn == "postgresql://finops/two"


def test_unconfigured_target_reports_itself(monkeypatch):
    """An empty DSN makes the picker say why the database is unusable."""
    monkeypatch.setattr(settings, "finops_dsn", "")
    assert FINOPS_TARGET.configured is False
    monkeypatch.setattr(settings, "finops_dsn", "postgresql://finops/two")
    assert FINOPS_TARGET.configured is True


def test_bare_database_type_still_means_the_demo_databases():
    """Type-keyed callers (query processing, older tests) must not drift onto
    FinOps just because it shares the PostgreSQL engine."""
    assert default_target_for_type(DatabaseType.POSTGRESQL) is POSTGRES_TARGET
    assert default_target_for_type(DatabaseType.MONGODB) is MONGO_TARGET
    assert as_target(DatabaseType.POSTGRESQL) is POSTGRES_TARGET
    assert as_target(FINOPS_TARGET) is FINOPS_TARGET
    assert paths.schema_path(DatabaseType.POSTGRESQL).name == "postgres.toml"
