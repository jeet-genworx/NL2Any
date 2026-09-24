"""Universal, deterministic PostgreSQL metadata extraction.

Extracts every table and column reachable via the connection's information_schema.
Makes no assumptions about schema names, table names, or column names/content.
"""

from datetime import datetime, timezone
from typing import Any
import psycopg

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)

# Built-in schemas that ship with every PostgreSQL installation; never user data.
SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")


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
            table_names = [row[1] for row in tables]

            cur.execute(
                """
                SELECT table_schema, table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema != ALL(%s)
                ORDER BY table_schema, table_name, ordinal_position;
                """,
                (list(SYSTEM_SCHEMAS),),
            )
            columns_by_table: dict[str, list[Field]] = {tbl: [] for tbl in table_names}
            for _schema, tbl, col_name, data_type, is_nullable in cur.fetchall():
                if tbl not in columns_by_table:
                    continue
                columns_by_table[tbl].append(
                    Field(
                        name=col_name,
                        type=data_type,
                        nullable=(is_nullable == "YES"),
                    )
                )

            cur.execute(
                """
                SELECT
                    kcu.table_name AS from_table,
                    kcu.column_name AS from_column,
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
                ORDER BY kcu.table_name, kcu.column_name;
                """,
                (list(SYSTEM_SCHEMAS),),
            )
            relationships: list[Relationship] = []
            for from_tbl, from_col, to_tbl, to_col in cur.fetchall():
                if from_tbl in columns_by_table and to_tbl in columns_by_table:
                    relationships.append(
                        Relationship(
                            from_object=from_tbl,
                            from_field=from_col,
                            to_object=to_tbl,
                            to_field=to_col,
                            relationship_type="many_to_one",
                        )
                    )

        schema_objects = [
            SchemaObject(
                name=tbl,
                kind=SchemaObjectKind.TABLE,
                description="",
                fields=columns_by_table[tbl],
            )
            for tbl in table_names
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
