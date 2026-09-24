"""Deterministic PostgreSQL metadata extraction."""

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

SAFE_CATEGORICAL_KEYWORDS = {
    "status",
    "category",
    "priority",
    "city",
    "state",
    "country",
    "role",
    "department",
    "type",
    "kind",
    "gender",
}

SENSITIVE_KEYWORDS = {
    "password",
    "secret",
    "token",
    "key",
    "auth",
    "hash",
    "credential",
    "salt",
    "email",
    "phone",
    "ssn",
    "address",
    "card",
    "cvv",
    "birth",
    "salary",
}


class PostgreSQLMetadataExtractor:
    """Extracts tables, columns, primary keys, foreign keys, and safe sample values from PostgreSQL."""

    def __init__(self, connection: psycopg.Connection[Any]) -> None:
        self.conn = connection

    def _is_safe_for_sampling(self, column_name: str, data_type: str) -> bool:
        col_lower = column_name.lower()
        # Strictly exclude sensitive fields
        if any(sens in col_lower for sens in SENSITIVE_KEYWORDS):
            return False

        # Only sample safe categorical string/enum-like fields
        is_string_type = any(
            t in data_type.lower()
            for t in ("character", "text", "varchar", "enum")
        )
        if not is_string_type:
            return False

        return any(safe in col_lower for safe in SAFE_CATEGORICAL_KEYWORDS)

    def _fetch_sample_values(self, table_name: str, column_name: str) -> list[Any]:
        try:
            # Safe bounded query
            query = f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL LIMIT 4;'
            with self.conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                return [str(r[0]) for r in rows if r[0] is not None]
        except Exception:
            return []

    def extract_schema(self) -> DatabaseSchema:
        """Extract authoritative PostgreSQL schema metadata."""
        with self.conn.cursor() as cur:
            # 1. Database name
            cur.execute("SELECT current_database();")
            db_name_row = cur.fetchone()
            db_name = db_name_row[0] if db_name_row else "postgres"

            # 2. Tables in public schema
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            table_names = [row[0] for row in cur.fetchall()]

            # 3. Columns
            cur.execute("""
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            columns_by_table: dict[str, list[Field]] = {tbl: [] for tbl in table_names}
            for tbl, col_name, data_type, is_nullable in cur.fetchall():
                if tbl not in columns_by_table:
                    continue

                sample_vals = []
                if self._is_safe_for_sampling(col_name, data_type):
                    sample_vals = self._fetch_sample_values(tbl, col_name)

                columns_by_table[tbl].append(
                    Field(
                        name=col_name,
                        type=data_type,
                        nullable=(is_nullable == "YES"),
                        sample_values=sample_vals,
                    )
                )

            # 4. Foreign Key Relationships
            cur.execute("""
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
                JOIN information_schema.constraint_column_usage ccu
                  ON rc.unique_constraint_name = ccu.constraint_name
                WHERE tc.table_schema = 'public'
                  AND tc.constraint_type = 'FOREIGN KEY'
                ORDER BY kcu.table_name, kcu.column_name;
            """)
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
