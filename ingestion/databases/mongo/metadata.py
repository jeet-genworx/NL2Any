"""Deterministic MongoDB schema and metadata extraction."""

from datetime import datetime, timezone
from typing import Any
from bson import ObjectId
from pymongo.database import Database

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)

SAFE_CATEGORICAL_KEYWORDS = {
    "status",
    "category",
    "priority",
    "city",
    "state",
    "country",
    "role",
    "department",
    "type",
    "kind",
    "gender",
}

SENSITIVE_KEYWORDS = {
    "password",
    "secret",
    "token",
    "key",
    "auth",
    "hash",
    "credential",
    "salt",
    "email",
    "phone",
    "ssn",
    "address",
    "card",
    "cvv",
    "birth",
    "salary",
}


def _infer_type_name(val: Any) -> str:
    """Map python/BSON values to standard type names."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, datetime):
        return "datetime"
    if isinstance(val, ObjectId):
        return "objectId"
    if isinstance(val, dict):
        return "object"
    if isinstance(val, list):
        if not val:
            return "array"
        elem_types = {type(elem) for elem in val}
        if len(elem_types) == 1:
            first = val[0]
            if isinstance(first, dict):
                return "array[object]"
            elif isinstance(first, (int, float)):
                return "array[number]"
            elif isinstance(first, str):
                return "array[string]"
        return "array"
    return type(val).__name__


class MongoDBMetadataExtractor:
    """Extracts collections, nested field paths, observed types, and safe sample values from MongoDB."""

    def __init__(self, db: Database[dict[str, Any]], sample_limit: int = 20) -> None:
        self.db = db
        self.sample_limit = sample_limit

    def _is_safe_for_sampling(self, field_name: str) -> bool:
        lower = field_name.lower()
        if any(sens in lower for sens in SENSITIVE_KEYWORDS):
            return False
        return any(safe in lower for safe in SAFE_CATEGORICAL_KEYWORDS)

    def _extract_fields_from_docs(self, docs: list[dict[str, Any]]) -> list[Field]:
        """Infer field structure and nested paths from sampled documents."""
        # Key -> (observed_types: set, sample_values: list, nested_docs: list)
        fields_map: dict[str, dict[str, Any]] = {}

        for doc in docs:
            self._traverse_doc(doc, fields_map)

        fields: list[Field] = []
        for name, info in fields_map.items():
            types_str = " | ".join(sorted(info["types"])) if info["types"] else "unknown"
            nested_fields = []
            if info["nested_docs"]:
                nested_fields = self._extract_fields_from_docs(info["nested_docs"])

            fields.append(
                Field(
                    name=name,
                    type=types_str,
                    nullable=info["nullable"],
                    sample_values=list(info["sample_values"])[:4],
                    nested=nested_fields,
                )
            )

        return fields

    def _traverse_doc(
        self,
        doc: dict[str, Any],
        fields_map: dict[str, dict[str, Any]],
    ) -> None:
        for k, v in doc.items():
            if k not in fields_map:
                fields_map[k] = {
                    "types": set(),
                    "nullable": False,
                    "sample_values": set(),
                    "nested_docs": [],
                }

            if v is None:
                fields_map[k]["nullable"] = True
                continue

            t_name = _infer_type_name(v)
            fields_map[k]["types"].add(t_name)

            if isinstance(v, dict):
                fields_map[k]["nested_docs"].append(v)
            elif isinstance(v, list):
                dict_items = [elem for elem in v if isinstance(elem, dict)]
                if dict_items:
                    fields_map[k]["nested_docs"].extend(dict_items)
            elif self._is_safe_for_sampling(k) and isinstance(v, (str, int, float, bool)):
                if len(fields_map[k]["sample_values"]) < 4:
                    fields_map[k]["sample_values"].add(str(v))

    def extract_schema(self) -> DatabaseSchema:
        """Extract authoritative MongoDB schema metadata."""
        collection_names = [
            c for c in self.db.list_collection_names()
            if not c.startswith("system.")
        ]
        collection_names.sort()

        schema_objects: list[SchemaObject] = []

        for c_name in collection_names:
            coll = self.db[c_name]
            sample_docs = list(coll.find().limit(self.sample_limit))
            fields = self._extract_fields_from_docs(sample_docs)

            schema_objects.append(
                SchemaObject(
                    name=c_name,
                    kind=SchemaObjectKind.COLLECTION,
                    description="",
                    fields=fields,
                )
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        schema = DatabaseSchema(
            database_type=DatabaseType.MONGODB,
            database_name=self.db.name,
            schema_version="1.0",
            last_updated=now_iso,
            objects=schema_objects,
            relationships=[],  # No relationships hallucinated in MongoDB
        )
        schema.validate_consistency()
        return schema
