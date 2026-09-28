"""Application entry point for NL2AnyQuery."""

import uvicorn
from backend.src.api.rest.app import app

__all__ = ["app", "run"]


def run() -> None:
    """Run the FastAPI application with uvicorn."""
    print("Starting NL2AnyQuery FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.src.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
