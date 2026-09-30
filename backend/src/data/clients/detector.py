"""Deterministic database type detection and identity parsing for connection strings.

Two jobs, both purely local -- no SLM, no network, no driver.

`detect_database_type` answers which engine a connection string names, which is
what picks the adapter, the generator and the validator.

`parse_connection` goes one step further and pulls out the pieces needed to
*identify* a database: its engine, the database name and the host. That is
everything required to name a target, label it in the picker, and give MongoDB
the database name its URI carries in the path.

What this module deliberately does not do is take a connection string apart in
order to reconnect from the pieces. The original string is always handed to
psycopg or pymongo untouched: they understand SRV lookups, `sslmode`,
`replicaSet`, percent-encoded credentials and libpq's keyword form, and a DSN
reassembled from parsed fragments would lose some of that. Parsing here is
best-effort and for identity only -- when it cannot make sense of a string it
returns empty fields rather than raising, and the driver remains the authority
on whether the string actually works.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from backend.src.data.models.schema import DatabaseType

_POSTGRES_SCHEMES = ("postgresql", "postgres")
_MONGO_SCHEMES = ("mongodb",)

# SQLAlchemy writes its URLs as `dialect+driver://`, so `postgresql+asyncpg://`
# and `postgresql+psycopg2://` are the forms an application's own config file
# usually holds -- and the forms someone pastes. The driver suffix names the
# Python library SQLAlchemy should load, which is not our concern: the database
# is the same one either way, so the suffix is dropped before the engine is
# decided. MongoDB's `mongodb+srv` only looks like this pattern; it is a real
# scheme meaning "resolve the hosts by SRV record", and stripping it would
# change which server is reached, so `_driver_scheme` never rewrites it.
_SQLALCHEMY_DRIVER = re.compile(r"^(postgresql|postgres)\+[a-z0-9_]+$", re.IGNORECASE)

# libpq also accepts "host=localhost dbname=app user=me", which has no scheme.
# psql and pgAdmin hand that form out freely, so it is worth recognising rather
# than rejecting as malformed.
_KEYWORD_PAIR = re.compile(r"(\w+)\s*=\s*('[^']*'|\S+)")


@dataclass(frozen=True)
class ConnectionInfo:
    """What could be read off a connection string, for identity purposes.

    `database` and `host` are empty strings when the string did not carry them
    or could not be parsed -- callers fall back to a generic label rather than
    treating that as an error.
    """

    db_type: DatabaseType
    database: str = ""
    host: str = ""
    port: str = ""


def _scheme_of(cleaned: str) -> str | None:
    """The URI scheme, lowercased, or None when the string has no `scheme://`."""
    match = re.match(r"^([a-zA-Z0-9\+\-]+)://", cleaned)
    return match.group(1).lower() if match else None


def _base_scheme(scheme: str) -> str:
    """The scheme with any SQLAlchemy driver suffix removed.

    `postgresql+asyncpg` -> `postgresql`, while `mongodb+srv` is returned as it
    is, because there the suffix is part of the scheme's meaning rather than a
    note about which Python library to import.
    """
    return scheme.split("+", 1)[0] if _SQLALCHEMY_DRIVER.match(scheme) else scheme


def driver_connection(connection_string: str) -> str:
    """The connection string in the form the database driver actually accepts.

    Almost always the string exactly as supplied -- psycopg and pymongo are the
    authorities on their own syntax, and this module does not try to rewrite
    what they already understand.

    The exception is a SQLAlchemy URL. `postgresql+asyncpg://...` is a perfectly
    normal thing to have in an application's config and to paste here, but libpq
    rejects that scheme outright, so the driver suffix is dropped. Nothing else
    about the string is touched, which does mean driver-specific query
    parameters are passed through as-is: asyncpg's `?ssl=require` is not
    translated into libpq's `?sslmode=require`, and a string relying on one will
    need that parameter adjusted by hand.
    """
    cleaned = str(connection_string).strip()
    scheme = _scheme_of(cleaned)
    if scheme is None:
        return cleaned

    base = _base_scheme(scheme)
    if base == scheme:
        return cleaned
    return f"{base}{cleaned[len(scheme):]}"


def _keyword_fields(cleaned: str) -> dict[str, str]:
    """Parse libpq's `key=value key=value` form into a dict, quotes stripped."""
    return {
        key.lower(): value.strip("'")
        for key, value in _KEYWORD_PAIR.findall(cleaned)
    }


def _looks_like_keyword_form(cleaned: str) -> bool:
    """Whether this is libpq keyword/value syntax rather than a URI."""
    fields = _keyword_fields(cleaned)
    return bool(fields) and ("host" in fields or "dbname" in fields or "hostaddr" in fields)


def detect_database_type(connection_string: str) -> DatabaseType:
    """Deterministically detect database type from a connection string.

    Supports:
        postgresql://...          -> DatabaseType.POSTGRESQL
        postgres://...            -> DatabaseType.POSTGRESQL
        postgresql+asyncpg://...  -> DatabaseType.POSTGRESQL (any SQLAlchemy driver)
        mongodb://...             -> DatabaseType.MONGODB
        mongodb+srv://..          -> DatabaseType.MONGODB
        host=... dbname=...       -> DatabaseType.POSTGRESQL (libpq keyword form)

    Does not use the SLM.
    """
    if not connection_string or not isinstance(connection_string, str):
        raise ValueError("Connection string must be a non-empty string.")

    cleaned = connection_string.strip()
    scheme = _scheme_of(cleaned)

    if scheme is None:
        # No scheme: the only form worth accepting is libpq's, which is
        # PostgreSQL by definition. Anything else is malformed.
        if _looks_like_keyword_form(cleaned):
            return DatabaseType.POSTGRESQL
        raise ValueError(
            f"Invalid connection string format: '{connection_string}'. Expected "
            "postgresql://, mongodb:// or libpq 'host=... dbname=...' syntax."
        )

    base = _base_scheme(scheme)
    if base in _POSTGRES_SCHEMES:
        return DatabaseType.POSTGRESQL
    if base in _MONGO_SCHEMES or scheme == "mongodb+srv":
        return DatabaseType.MONGODB
    raise ValueError(
        f"Unsupported database scheme: '{scheme}'. Only PostgreSQL and MongoDB "
        "are supported (postgresql://, postgres://, postgresql+<driver>://, "
        "mongodb://, mongodb+srv://)."
    )


def parse_connection(connection_string: str) -> ConnectionInfo:
    """Read engine, database name and host off a connection string.

    Raises ValueError for a string whose engine cannot be determined -- that is
    a genuine input error worth reporting. Everything beyond the engine is
    best-effort: a string this cannot fully parse still yields a ConnectionInfo,
    with empty fields where the detail was unavailable.

    The host is taken from the last `@` onwards, which is how both libpq and
    pymongo split it, so an unencoded `@` inside a password does not shift the
    host. An unencoded `/` in a password will still confuse it; that only costs
    a nicer label, never a connection.
    """
    db_type = detect_database_type(connection_string)
    cleaned = connection_string.strip()

    if _scheme_of(cleaned) is None:
        fields = _keyword_fields(cleaned)
        return ConnectionInfo(
            db_type=db_type,
            database=fields.get("dbname", "") or fields.get("database", ""),
            host=fields.get("host", "") or fields.get("hostaddr", ""),
            port=fields.get("port", ""),
        )

    try:
        parts = urlsplit(cleaned)
        host = parts.hostname or ""
        port = str(parts.port) if parts.port else ""
    except ValueError:
        # A malformed port or bracketed IPv6 host makes urlsplit raise. The
        # string may still be perfectly valid to the driver, so fall through
        # with what little is known rather than rejecting it here.
        return ConnectionInfo(db_type=db_type)

    # Both engines carry the database name as the first path segment;
    # MongoDB's may be absent, with `?authSource=` naming a different thing.
    database = unquote(parts.path.lstrip("/").split("/")[0]) if parts.path else ""

    return ConnectionInfo(db_type=db_type, database=database, host=host, port=port)


def identity_of(connection_string: str) -> str:
    """A canonical string naming *which database* a connection string reaches.

    Used to decide whether a newly supplied connection string refers to a
    database that has already been ingested. Credentials are deliberately
    excluded, so rotating a password does not orphan a database's schema and
    embeddings on disk.

    The trade-off is that two connection strings differing only by user map to
    one identity, and so share ingested artifacts. That is the right default
    here -- the artifacts describe the database, not the session -- but it does
    mean a user with narrower visible tables reuses the fuller schema that a
    previous user's ingestion produced.
    """
    info = parse_connection(connection_string)
    port = f":{info.port}" if info.port else ""
    return f"{info.db_type.value}://{info.host.lower()}{port}/{info.database}"


def redact(connection_string: str) -> str:
    """The connection string with any password replaced by `***`.

    For log lines and error messages, which should be able to name the string
    that failed without writing a credential into a file or a response body.
    """
    cleaned = str(connection_string).strip()

    if _scheme_of(cleaned) is None:
        return _KEYWORD_PAIR.sub(
            lambda m: f"{m.group(1)}=***" if m.group(1).lower() == "password" else m.group(0),
            cleaned,
        )

    scheme, separator, rest = cleaned.partition("://")
    if not separator:
        return cleaned

    netloc, slash, tail = rest.partition("/")
    credentials, at, host = netloc.rpartition("@")
    if not at:
        return cleaned

    user, colon, _password = credentials.partition(":")
    masked = f"{user}:***" if colon else user
    return f"{scheme}://{masked}@{host}{slash}{tail}"
