"""Safe database query execution enforcing timeouts and row limits."""

from typing import Any
import psycopg
from pymongo import MongoClient

from backend.src.config import settings
from backend.src.schemas.pipeline import GeneratedQuery, MongoQuery
from backend.src.data.models.schema import DatabaseType


class QueryExecutor:
    """Executes validated and policy-approved database queries with strict bounds."""

    def __init__(
        self,
        postgres_dsn: str | None = None,
        mongo_uri: str | None = None,
        mongo_database: str | None = None,
        timeout_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.postgres_dsn = postgres_dsn or settings.postgres_dsn
        self.mongo_uri = mongo_uri or settings.mongodb_uri
        self.mongo_database = mongo_database or settings.mongodb_database
        self.timeout_seconds = timeout_seconds or settings.query_timeout_seconds
        self.max_rows = max_rows or settings.max_result_rows

    def execute(
        self,
        query: GeneratedQuery,
        connection: str | None = None,
        database: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Execute query safely and return (columns, rows).

        `connection` and `database` come from the target being queried, so a
        database added at runtime is executed against its own server rather
        than the one named in settings. Both fall back to settings when absent,
        which is what the built-in targets rely on. `database` applies to
        MongoDB only; PostgreSQL's is inside the DSN.
        """
        if query.database_type == DatabaseType.POSTGRESQL:
            return self._execute_postgres(str(query.raw_query), dsn=connection)
        elif query.database_type == DatabaseType.MONGODB:
            if not isinstance(query.raw_query, MongoQuery):
                raise ValueError("MongoDB query must be a MongoQuery model instance.")
            return self._execute_mongo(query.raw_query, uri=connection, database=database)
        raise ValueError(f"Unsupported database type: {query.database_type}")

    def _execute_postgres(
        self,
        sql: str,
        dsn: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        # An empty string means the caller HAS a target and that target's DSN is
        # unconfigured. Falling back to POSTGRES_DSN there runs the query against
        # a different database than the one the user selected, which does not
        # fail -- it answers from the wrong data, or returns no rows at all. Only
        # the absence of a target (None) may fall back.
        if dsn is not None and not dsn.strip():
            raise ValueError(
                "This database's connection string is not configured, so the query "
                "would have run against the default POSTGRES_DSN database instead. "
                "Set its DSN (for the FinOps target, FINOPS_DSN) in .env."
            )

        conn_str = dsn or self.postgres_dsn
        if not conn_str:
            raise ValueError("PostgreSQL connection string (POSTGRES_DSN) is not configured.")

        timeout_ms = int(self.timeout_seconds * 1000)
        with psycopg.connect(conn_str, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Set statement timeout for safe execution
                cur.execute(f"SET statement_timeout = {timeout_ms};")
                cur.execute(sql)

                if not cur.description:
                    return [], []

                columns = [desc[0] for desc in cur.description]
                rows_tuples = cur.fetchmany(self.max_rows)
                rows = [dict(zip(columns, row)) for row in rows_tuples]
                return columns, rows

    def _execute_mongo(
        self,
        query: MongoQuery,
        uri: str | None = None,
        database: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        # As above: an unconfigured target must not quietly become MONGODB_URI.
        if uri is not None and not uri.strip():
            raise ValueError(
                "This database's connection URI is not configured, so the query "
                "would have run against the default MONGODB_URI database instead. "
                "Set its URI in .env."
            )

        conn_uri = uri or self.mongo_uri
        if not conn_uri:
            raise ValueError("MongoDB connection URI (MONGODB_URI) is not configured.")

        db_name = database or self.mongo_database
        if not db_name:
            raise ValueError("MongoDB database name is not configured.")

        client: MongoClient[dict[str, Any]] = MongoClient(
            conn_uri,
            serverSelectionTimeoutMS=int(self.timeout_seconds * 1000),
        )
        try:
            db = client[db_name]
            collection = db[query.collection]

            limit = min(query.limit or self.max_rows, self.max_rows)

            raw_docs: list[dict[str, Any]] = []

            if query.operation == "find":
                find_filter = query.filter or {}
                find_kwargs: dict[str, Any] = {}
                if query.projection and "*" not in query.projection and "all" not in query.projection:
                    find_kwargs["projection"] = query.projection

                cursor = collection.find(find_filter, **find_kwargs)
                if query.sort:
                    cursor = cursor.sort(query.sort)
                cursor = cursor.limit(limit)
                raw_docs = list(cursor)

            elif query.operation == "aggregate":
                pipeline = list(query.pipeline or [])
                # Ensure pipeline has a limit stage capped at max_rows
                has_limit = any("$limit" in stage for stage in pipeline)
                if not has_limit:
                    pipeline.append({"$limit": limit})
                cursor = collection.aggregate(pipeline)
                raw_docs = list(cursor)

            # Discover columns
            column_set: set[str] = set()
            for doc in raw_docs:
                column_set.update(doc.keys())

            # Put _id first if present
            columns = sorted(column_set)
            if "_id" in columns:
                columns.remove("_id")
                columns.insert(0, "_id")

            return columns, raw_docs
        finally:
            client.close()
