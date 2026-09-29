"""Universal, deterministic PostgreSQL metadata extraction.

Extracts every table and column reachable via the connection's information_schema.
Makes no assumptions about schema names, table names, or column names/content.
"""

from datetime import datetime, timezone
from typing import Any
import psycopg

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)

# Built-in schemas that ship with every PostgreSQL installation; never user data.
SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")


def _format_table_name(schema: str, table: str) -> str:
    """Format table name: unqualified for default 'public' schema, schema-qualified otherwise."""
    return table if schema == "public" else f"{schema}.{table}"


class PostgreSQLMetadataExtractor:
    """Extracts every table and column visible to the connection, with foreign keys."""

    def __init__(self, connection: psycopg.Connection[Any]) -> None:
        self.conn = connection

    def extract_schema(self) -> DatabaseSchema:
        """Extract all tables and columns from every non-system schema."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            db_name_row = cur.fetchone()
            db_name = db_name_row[0] if db_name_row else "postgres"

            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema != ALL(%s) AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name;
                """,
                (list(SYSTEM_SCHEMAS),),
            )
            tables = cur.fetchall()
            table_map = {
                (schema, tbl): _format_table_name(schema, tbl)
                for schema, tbl in tables
            }

            cur.execute(
                """
                SELECT table_schema, table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema != ALL(%s)
                ORDER BY table_schema, table_name, ordinal_position;
                """,
                (list(SYSTEM_SCHEMAS),),
            )
            columns_by_table: dict[str, list[Field]] = {name: [] for name in table_map.values()}
            for schema, tbl, col_name, data_type, is_nullable in cur.fetchall():
                full_name = table_map.get((schema, tbl))
                if full_name is None:
                    continue
                columns_by_table[full_name].append(
                    Field(
                        name=col_name,
                        type=data_type,
                        nullable=(is_nullable == "YES"),
                    )
                )

            cur.execute(
                """
                SELECT
                    kcu.table_schema AS from_schema,
                    kcu.table_name AS from_table,
                    kcu.column_name AS from_column,
                    ccu.table_schema AS to_schema,
                    ccu.table_name AS to_table,
                    ccu.column_name AS to_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.referential_constraints rc
                  ON tc.constraint_name = rc.constraint_name
                  AND tc.table_schema = rc.constraint_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON rc.unique_constraint_name = ccu.constraint_name
                  AND rc.unique_constraint_schema = ccu.table_schema
                WHERE tc.table_schema != ALL(%s)
                  AND tc.constraint_type = 'FOREIGN KEY'
                ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name;
                """,
                (list(SYSTEM_SCHEMAS),),
            )
            relationships: list[Relationship] = []
            for from_s, from_tbl, from_col, to_s, to_tbl, to_col in cur.fetchall():
                from_obj = table_map.get((from_s, from_tbl))
                to_obj = table_map.get((to_s, to_tbl))
                if from_obj and to_obj:
                    relationships.append(
                        Relationship(
                            from_object=from_obj,
                            from_field=from_col,
                            to_object=to_obj,
                            to_field=to_col,
                            relationship_type="many_to_one",
                        )
                    )

        schema_objects = [
            SchemaObject(
                name=name,
                kind=SchemaObjectKind.TABLE,
                description="",
                fields=columns_by_table[name],
            )
            for name in table_map.values()
        ]

        now_iso = datetime.now(timezone.utc).isoformat()
        schema = DatabaseSchema(
            database_type=DatabaseType.POSTGRESQL,
            database_name=db_name,
            schema_version="1.0",
            last_updated=now_iso,
            objects=schema_objects,
            relationships=relationships,
        )
        schema.validate_consistency()
        return schema
