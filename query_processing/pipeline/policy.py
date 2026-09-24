from __future__ import annotations

import sqlglot
from sqlglot import exp
from query_processing.models.pipeline import GeneratedQuery, MongoQuery, PolicyResult, RelevantSchema
from query_processing.models.schema import DatabaseType

PROHIBITED_SQL_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Transaction,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
)


class SafetyPolicyValidator:
    """Enforces non-negotiable deterministic read-only boundaries."""

    def check(
        self,
        query: GeneratedQuery,
        schema: RelevantSchema | None = None,
    ) -> PolicyResult:
        """Validate that the query is strictly read-only and references valid schema."""
        if query.database_type == DatabaseType.POSTGRESQL:
            return self._check_postgres(str(query.raw_query), schema=schema)
        elif query.database_type == DatabaseType.MONGODB:
            return self._check_mongo(query.raw_query)
        return PolicyResult(allowed=False, reason=f"Unsupported database type: {query.database_type}")

    def _check_postgres(
        self,
        sql: str,
        schema: RelevantSchema | None = None,
    ) -> PolicyResult:
        if not sql or not sql.strip():
            return PolicyResult(allowed=False, reason="Empty SQL query.")

        # 1. Multi-statement check
        try:
            statements = sqlglot.parse(sql, read="postgres")
        except Exception as err:
            return PolicyResult(allowed=False, reason=f"Failed to parse SQL: {err}")

        # Filter out empty or whitespace statements
        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            return PolicyResult(
                allowed=False,
                reason=f"Prohibited: multiple SQL statements detected ({len(statements)} found).",
            )

        root = statements[0]

        # 2. Check for prohibited AST expression types
        for prohibited in PROHIBITED_SQL_EXPRESSIONS:
            if root.find(prohibited) is not None or isinstance(root, prohibited):
                return PolicyResult(
                    allowed=False,
                    reason=f"Prohibited SQL expression detected: {prohibited.__name__}.",
                )

        # 3. Must be a Select, Union, or CTE Select
        if not isinstance(root, (exp.Select, exp.Union)):
            select_expr = root.find(exp.Select)
            if select_expr is None:
                return PolicyResult(
                    allowed=False,
                    reason=f"Query does not evaluate to a read-only SELECT (got {type(root).__name__}).",
                )

        # 4. Deterministic schema reference check (if schema provided)
        if schema is not None:
            allowed_tables = schema.get_object_names()
            for table_exp in root.find_all(exp.Table):
                t_name = table_exp.name.lower()
                if t_name and t_name not in allowed_tables:
                    return PolicyResult(
                        allowed=False,
                        reason=f"Prohibited: table '{t_name}' does not exist in relevant schema {sorted(allowed_tables)}.",
                    )

        return PolicyResult(allowed=True, reason="Query verified read-only SELECT.")

    def _check_mongo(self, query: str | MongoQuery) -> PolicyResult:
        if not isinstance(query, MongoQuery):
            return PolicyResult(
                allowed=False,
                reason="MongoDB query must be a typed MongoQuery structure.",
            )

        if query.operation not in ("find", "aggregate"):
            return PolicyResult(
                allowed=False,
                reason=f"Prohibited MongoDB operation '{query.operation}'. Only 'find' and 'aggregate' are allowed.",
            )

        # Check for disallowed aggregate stages if operation is aggregate
        if query.operation == "aggregate" and query.pipeline:
            disallowed_stages = {"$out", "$merge", "$writeConcern", "$collMod", "$planCacheStats"}
            for stage in query.pipeline:
                for key in stage.keys():
                    if key.lower() in disallowed_stages:
                        return PolicyResult(
                            allowed=False,
                            reason=f"Prohibited MongoDB aggregation stage: '{key}'.",
                        )

        return PolicyResult(allowed=True, reason="MongoDB operation verified read-only.")
