"""Filesystem primitives shared by every repository in this package.

Every ingestion artifact is a JSON or TOML file on disk, and every one of them
was previously written with its own copy of "mkdir the parent, dump, write with
an explicit encoding" and read with its own copy of "check it exists, raise a
FileNotFoundError that tells the user which command to run first". Those two
patterns live here once, so a repository module is left saying only *what* it
stores and in which shape.
"""

import json
import tomllib
from pathlib import Path
from typing import Any, Callable

import tomli_w

PathLike = Path | str

# Parsed artifacts, keyed by resolved path. Each entry remembers the file
# signature it was parsed from, so a file rewritten by a later ingestion run is
# re-read on the next access and a file that has not changed is not parsed again.
_cache: dict[str, tuple[tuple[int, int], Any]] = {}


def _signature(path: Path) -> tuple[int, int]:
    """Cheap change detector for a file: (modification time, size)."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def read_cached(
    path: PathLike,
    parser: Callable[[Path], Any],
    *,
    hint: str = "",
) -> Any:
    """Parse `path` with `parser`, reusing the previous result while unchanged.

    The returned object is shared between callers, so treat it as read-only and
    copy anything you intend to modify.
    """
    resolved = require(path, hint=hint)
    key = str(resolved.resolve())
    signature = _signature(resolved)

    entry = _cache.get(key)
    if entry is not None and entry[0] == signature:
        return entry[1]

    parsed = parser(resolved)
    _cache[key] = (signature, parsed)
    return parsed


def clear_cache() -> None:
    """Drop every cached artifact. Used by tests and after bulk rewrites."""
    _cache.clear()


def require(path: PathLike, *, hint: str = "") -> Path:
    """Return `path` as a Path, raising FileNotFoundError when it is missing.

    `hint` is the recovery advice appended to the message -- usually the CLI
    command that produces the missing file.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found at: {resolved}. {hint}".strip())
    return resolved


def write_text(path: PathLike, content: str) -> Path:
    """Write text to `path`, creating parent directories as needed.

    Drops any cached parse of that path, so a reader that asks for it next sees
    what was just written rather than the previous version.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    _cache.pop(str(resolved.resolve()), None)
    return resolved


def write_json(path: PathLike, payload: Any, *, indent: int | None = None) -> Path:
    """Serialize `payload` as JSON and write it to `path`.

    Passing `indent` also appends a trailing newline, so human-facing records
    (the descriptions file) stay diff-friendly while machine-facing bulk data
    (the embedding vectors) stays compact.
    """
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    return write_text(path, text + "\n" if indent is not None else text)


def write_toml(path: PathLike, document: dict[str, Any]) -> Path:
    """Serialize `document` as TOML and write it to `path`."""
    return write_text(path, tomli_w.dumps(document))


def read_text(path: PathLike, *, hint: str = "") -> str:
    """Read text from `path`, raising FileNotFoundError with `hint` if missing."""
    return require(path, hint=hint).read_text(encoding="utf-8")


def read_json(path: PathLike, *, hint: str = "") -> Any:
    """Parse JSON from `path`, raising FileNotFoundError with `hint` if missing."""
    return json.loads(read_text(path, hint=hint))


def read_toml(path: PathLike, *, hint: str = "") -> dict[str, Any]:
    """Parse TOML from `path`, raising FileNotFoundError with `hint` if missing."""
    return tomllib.loads(read_text(path, hint=hint))


def read_toml_cached(path: PathLike, *, hint: str = "") -> dict[str, Any]:
    """Parse TOML from `path`, reusing the previous parse while unchanged."""
    return read_cached(
        path,
        lambda resolved: tomllib.loads(resolved.read_text(encoding="utf-8")),
        hint=hint,
    )


def read_json_cached(path: PathLike, *, hint: str = "") -> Any:
    """Parse JSON from `path`, reusing the previous parse while unchanged."""
    return read_cached(
        path,
        lambda resolved: json.loads(resolved.read_text(encoding="utf-8")),
        hint=hint,
    )
