"""Tests for the generated documentation TOML: completeness and caching.

Ingestion writes this file once; every later reader is expected to take what it
holds rather than recompute it, and to reuse the parse until the file changes.
"""

import tomllib

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.data.repositories import description_repository as dr
from backend.src.data.repositories import files
from backend.src.schemas.ingestion import TableDescription


def _schema(db_type: DatabaseType = DatabaseType.POSTGRESQL) -> DatabaseSchema:
    kind = (
        SchemaObjectKind.TABLE
        if db_type == DatabaseType.POSTGRESQL
        else SchemaObjectKind.COLLECTION
    )
    return DatabaseSchema(
        database_type=db_type,
        database_name="shop_db",
        objects=[
            SchemaObject(
                name="customers",
                kind=kind,
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="email", type="text"),
                    Field(
                        name="address",
                        type="object",
                        nested=[Field(name="city", type="text")],
                    ),
                ],
            )
        ],
    )


def test_documentation_lists_every_column_when_written_with_the_schema(tmp_path):
    """The model described one column; the file still records all of them, so a
    reader never has to consult the schema to learn the shape."""
    path = tmp_path / "postgres_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Customers.", columns={"id": "Key."})},
        path,
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        schema=_schema(),
    )

    columns = tomllib.loads(path.read_text())["tables"]["customers"]["columns"]
    assert columns == {"id": "Key.", "email": "", "address": "", "address.city": ""}


def test_documentation_without_schema_records_only_what_was_generated(tmp_path):
    path = tmp_path / "postgres_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Customers.", columns={"id": "Key."})},
        path,
        database_type=DatabaseType.POSTGRESQL,
    )

    assert tomllib.loads(path.read_text())["tables"]["customers"]["columns"] == {"id": "Key."}


def test_load_documentation_carries_database_metadata_and_kind(tmp_path):
    """Everything the API needs comes from this one file."""
    path = tmp_path / "mongo_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Customer documents.")},
        path,
        database_type=DatabaseType.MONGODB,
        database_name="shop_demo",
        schema=_schema(DatabaseType.MONGODB),
    )

    documentation = dr.load_documentation(path)

    assert documentation.database_type == DatabaseType.MONGODB
    assert documentation.database_name == "shop_demo"
    assert documentation.object_kind == SchemaObjectKind.COLLECTION
    assert documentation.generated_at
    assert "address.city" in documentation.objects["customers"].columns


def test_load_documentation_reuses_the_parse(tmp_path):
    """An unchanged file is parsed once, however many times it is requested."""
    path = tmp_path / "postgres_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Customers.")},
        path,
        database_type=DatabaseType.POSTGRESQL,
    )
    files.clear_cache()

    first = dr.load_documentation(path)
    assert dr.load_documentation(path) is first


def test_rewriting_the_file_invalidates_the_cache(tmp_path):
    """A later ingestion run is picked up without restarting the process."""
    path = tmp_path / "postgres_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Before.")},
        path,
        database_type=DatabaseType.POSTGRESQL,
    )
    assert dr.load_documentation(path).objects["customers"].description == "Before."

    dr.save_descriptions(
        {"customers": TableDescription(description="After.")},
        path,
        database_type=DatabaseType.POSTGRESQL,
    )

    assert dr.load_documentation(path).objects["customers"].description == "After."


def test_load_descriptions_drops_undescribed_columns(tmp_path):
    """The embedding/apply view must never see an empty string as a description,
    even though the file keeps the column slot."""
    path = tmp_path / "postgres_descriptions.toml"
    dr.save_descriptions(
        {"customers": TableDescription(description="Customers.", columns={"id": "Key."})},
        path,
        database_type=DatabaseType.POSTGRESQL,
        schema=_schema(),
    )

    assert dr.load_documentation(path).objects["customers"].columns == {
        "id": "Key.",
        "email": "",
        "address": "",
        "address.city": "",
    }
    assert dr.load_descriptions(path)["customers"].columns == {"id": "Key."}
