"""Persistence for ingestion artifacts.

One module per artifact -- schema TOML, graph/MST TOML, documentation TOML,
embeddings JSON -- over shared filesystem primitives in `files`, with every
canonical location resolved by `paths`. Business logic in `core/` builds and
consumes these artifacts; only this package knows how they are stored.
"""
