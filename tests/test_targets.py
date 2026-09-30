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
    all_targets,
    as_target,
    clear_dynamic_targets,
    default_target_for_type,
    find_by_connection,
    register_target,
    resolve_target,
    target_from_connection,
    unregister_target,
)
from backend.src.data.repositories import paths


@pytest.fixture(autouse=True)
def _no_leftover_dynamic_targets():
    """Targets registered from a connection string live in module state; keep
    each test's registrations out of the next one."""
    clear_dynamic_targets()
    yield
    clear_dynamic_targets()


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


# --- Targets built from a user-supplied connection string ---------------------


def test_the_same_database_always_derives_the_same_key():
    """The guarantee that stops a database being ingested twice: supply the same
    connection string again and it resolves to the artifacts already on disk."""
    first = target_from_connection("postgresql://u:p@localhost:5432/shop")
    second = target_from_connection("postgresql://u:p@localhost:5432/shop")
    assert first.key == second.key


@pytest.mark.parametrize(
    "variant",
    [
        # Rotated password: the artifacts describe the database, not the session.
        "postgresql://u:rotated@localhost:5432/shop",
        # Different user, same database.
        "postgresql://someone:else@localhost:5432/shop",
        # Host case is not significant.
        "postgresql://u:p@LOCALHOST:5432/shop",
        # Either accepted spelling of the scheme.
        "postgres://u:p@localhost:5432/shop",
        # A SQLAlchemy URL names the same database; the driver suffix only says
        # which Python library SQLAlchemy would import.
        "postgresql+asyncpg://u:p@localhost:5432/shop",
        "postgresql+psycopg2://u:p@localhost:5432/shop",
    ],
)
def test_credentials_and_spelling_do_not_change_the_key(variant):
    baseline = target_from_connection("postgresql://u:p@localhost:5432/shop")
    assert target_from_connection(variant).key == baseline.key


def test_a_different_database_derives_a_different_key():
    shop = target_from_connection("postgresql://u:p@localhost:5432/shop")
    other = target_from_connection("postgresql://u:p@localhost:5432/warehouse")
    elsewhere = target_from_connection("postgresql://u:p@other-host:5432/shop")
    assert len({shop.key, other.key, elsewhere.key}) == 3


def test_a_user_supplied_target_carries_its_own_connection_and_artifacts(monkeypatch):
    monkeypatch.setattr(settings, "postgres_dsn", "postgresql://configured/demo")
    target = target_from_connection("postgresql://u:p@localhost:5432/shop", label="Shop")

    assert target.connection == "postgresql://u:p@localhost:5432/shop"
    assert target.label == "Shop"
    assert target.source == "user"
    # Nothing it owns may collide with a built-in database's files.
    for resolver in (paths.schema_path, paths.embeddings_path, paths.descriptions_path):
        assert resolver(target) != resolver(POSTGRES_TARGET), resolver.__name__
    assert get_adapter(target).dsn == "postgresql://u:p@localhost:5432/shop"


def test_a_mongo_target_takes_its_database_name_from_the_uri():
    """pymongo needs the database separately, so it has to be parsed out; without
    this a user-supplied URI would query the database named in settings."""
    target = target_from_connection("mongodb://host:27017/analytics?authSource=admin")

    assert target.db_type == DatabaseType.MONGODB
    assert target.mongo_database == "analytics"
    assert get_adapter(target).database_name == "analytics"


def test_a_postgres_target_has_no_separate_database_name():
    """psycopg parses the DSN itself, so nothing needs to be pulled out of it."""
    assert target_from_connection("postgresql://u:p@host:5432/shop").mongo_database is None


def test_the_label_falls_back_to_the_database_name():
    assert target_from_connection("postgresql://u:p@host:5432/shop").label == "shop (PostgreSQL)"
    assert target_from_connection("mongodb://host:27017/shop").label == "shop (MongoDB)"


def test_an_unsupported_engine_is_refused():
    with pytest.raises(ValueError):
        target_from_connection("redis://localhost:6379")


def test_a_sqlalchemy_url_is_stored_in_the_form_the_driver_accepts():
    """psycopg is handed the target's connection directly, and libpq does not
    understand `postgresql+asyncpg://`."""
    target = target_from_connection("postgresql+asyncpg://u:p@host:5432/shop")

    assert target.connection == "postgresql://u:p@host:5432/shop"
    assert get_adapter(target).dsn == "postgresql://u:p@host:5432/shop"


def test_a_mongodb_srv_uri_reaches_pymongo_unchanged():
    """`+srv` changes how the hosts are resolved, so it must survive."""
    target = target_from_connection("mongodb+srv://u:p@cluster.mongodb.net/shop")

    assert target.connection == "mongodb+srv://u:p@cluster.mongodb.net/shop"
    assert target.mongo_database == "shop"


def test_registering_is_idempotent_and_keeps_the_original():
    first = register_target(target_from_connection("postgresql://u:p@h:5432/shop", label="First"))
    second = register_target(target_from_connection("postgresql://u:p@h:5432/shop", label="Second"))

    assert second is first
    assert second.label == "First"
    assert len(all_targets()) == len(TARGETS) + 1


def test_a_registered_target_resolves_like_a_built_in():
    target = register_target(target_from_connection("postgresql://u:p@h:5432/shop"))
    assert resolve_target(target.key) is target
    assert target.key in all_targets()


def test_unregistering_removes_only_user_supplied_targets():
    target = register_target(target_from_connection("postgresql://u:p@h:5432/shop"))

    assert unregister_target(target.key) is target
    assert unregister_target(target.key) is None
    with pytest.raises(ValueError):
        unregister_target("postgres")


def test_find_by_connection_matches_a_configured_built_in(monkeypatch):
    """Pasting the DSN behind POSTGRES_DSN must select that database, not save a
    second copy of it and ingest the same schema again."""
    monkeypatch.setattr(settings, "postgres_dsn", "postgresql://u:p@localhost:5432/demo")

    assert find_by_connection("postgresql://other:creds@localhost:5432/demo") is POSTGRES_TARGET
    assert find_by_connection("postgresql://u:p@localhost:5432/somewhere-else") is None


def test_find_by_connection_ignores_unconfigured_targets(monkeypatch):
    """An empty DSN must not match an empty-ish parse of the supplied string."""
    monkeypatch.setattr(settings, "finops_dsn", "")
    assert find_by_connection("postgresql://u:p@localhost:5432/shop") is None
