from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the natural language query endpoint."""

    database: str = Field(
        default="postgres",
        description="Database target: 'postgres' or 'mongo'",
    )
    question: str = Field(
        ...,
        description="Natural language question to query",
    )
    provider: Literal["koboldcpp", "huggingface", "gemini"] = Field(
        default="koboldcpp",
        description="Selected LLM provider: 'koboldcpp', 'huggingface', or 'gemini'",
    )
    jargons: list[str] = Field(
        default_factory=list,
        description="Optional list of domain-specific jargon terms for spelling check",
    )
