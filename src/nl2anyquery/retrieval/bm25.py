"""In-memory BM25 schema object retriever."""

from pathlib import Path
import re
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from nl2anyquery.core.config import settings
from nl2anyquery.models.schema import DatabaseSchema, SchemaObject, SchemaObjectKind
from nl2anyquery.schema.toml_store import load_schema_file


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    # Split on whitespace and non-alphanumeric characters, keeping words and numbers
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    # Expand snake_case into sub-tokens as well to improve lexical matching
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        if "_" in tok:
            for sub in tok.split("_"):
                if sub:
                    expanded.append(sub)
    return expanded


def _build_document_text(obj: SchemaObject) -> str:
    """Build searchable document text from SchemaObject."""
    kind_label = "TABLE" if obj.kind == SchemaObjectKind.TABLE else "COLLECTION"
    field_label = "COLUMNS" if obj.kind == SchemaObjectKind.TABLE else "FIELDS"

    field_paths = obj.all_field_paths()
    fields_str = " ".join(field_paths)

    field_desc_parts = []
    for path in field_paths:
        f = obj.get_field(path)
        if f and f.description:
            field_desc_parts.append(f"{path} = {f.description}")
    field_desc_str = " ".join(field_desc_parts)

    return (
        f"{kind_label}: {obj.name} "
        f"DESCRIPTION: {obj.description} "
        f"{field_label}: {fields_str} "
        f"{field_label} DESCRIPTIONS: {field_desc_str}"
    )


class RetrievalResult(BaseModel):
    """Result of BM25 retrieval for a schema object."""

    object_name: str
    score: float
    schema_object: SchemaObject


class BM25Retriever:
    """In-memory BM25 index built from DatabaseSchema."""

    def __init__(self, schema: DatabaseSchema) -> None:
        self.schema = schema
        self.objects = schema.objects

        # Build corpus
        self.corpus_texts = [_build_document_text(obj) for obj in self.objects]
        self.tokenized_corpus = [_tokenize(doc) for doc in self.corpus_texts]

        # Initialize BM25 model
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    @classmethod
    def from_toml_file(cls, file_path: Path | str) -> "BM25Retriever":
        """Instantiate BM25Retriever directly from a TOML schema file."""
        schema = load_schema_file(file_path)
        return cls(schema)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve schema objects matching the query, ordered by BM25 score."""
        if not self.bm25 or not self.objects:
            return []

        k = top_k if top_k is not None else settings.bm25_top_k
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            # Return top k default objects if query has no tokens
            return [
                RetrievalResult(
                    object_name=obj.name,
                    score=0.0,
                    schema_object=obj,
                )
                for obj in self.objects[:k]
            ]

        scores = self.bm25.get_scores(tokenized_query)

        # Pair objects with their scores
        scored_objects = [
            (self.objects[i], float(scores[i]))
            for i in range(len(self.objects))
        ]

        # Sort descending by score
        scored_objects.sort(key=lambda pair: pair[1], reverse=True)

        results = [
            RetrievalResult(
                object_name=obj.name,
                score=score,
                schema_object=obj,
            )
            for obj, score in scored_objects[:k]
        ]
        return results
