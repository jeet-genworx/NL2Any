"""Builds the database adapter for a database type.

Lives beside the adapters rather than in the ingestion pipeline: choosing an
adapter and wiring it to its connection settings is a data-layer concern, and
anything needing metadata extraction can now ask for one without importing
ingestion.
"""

from backend.src.config import settings
from backend.src.data.clients.base import DatabaseAdapter
from backend.src.data.clients.mongo.adapter import MongoDBAdapter
from backend.src.data.clients.postgres.adapter import PostgreSQLAdapter
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import DatabaseTarget, as_target


def get_adapter(database: DatabaseTarget | DatabaseType) -> DatabaseAdapter:
    """Build the adapter for a database target, using its own connection string.

    Takes a target rather than an engine type so two PostgreSQL databases reach
    their own DSNs; a bare DatabaseType still resolves to the demo target.
    """
    target = as_target(database)
    if target.db_type == DatabaseType.POSTGRESQL:
        return PostgreSQLAdapter(dsn=target.connection)
    # The database within a MongoDB server comes from the target when it carries
    # one -- a user-supplied URI names it in the path -- and from settings only
    # for the built-in target, whose URI has no path.
    return MongoDBAdapter(
        uri=target.connection,
        database=target.mongo_database or settings.mongodb_database,
    )
