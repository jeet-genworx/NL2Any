"""Standalone entrypoint for extracting and writing the canonical schema TOML."""

from query_processing.cli import init_schema_cli

if __name__ == "__main__":
    init_schema_cli()
