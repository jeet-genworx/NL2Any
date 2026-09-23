"""Normalized schema data models."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field as PydanticField


class DatabaseType(str, Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class SchemaObjectKind(str, Enum):
    """Kind of database schema object."""

    TABLE = "table"
    COLLECTION = "collection"


class Field(BaseModel):
    """Database-independent representation of a column or document field."""

    name: str
    type: str
    description: str = ""
    nullable: bool = True
    sample_values: list[Any] = PydanticField(default_factory=list)
    nested: list["Field"] = PydanticField(default_factory=list)

    def all_field_names(self, prefix: str = "") -> list[str]:
        """Return all field names/paths including nested ones in dotted notation."""
        current = f"{prefix}.{self.name}" if prefix else self.name
        names = [current]
        for child in self.nested:
            names.extend(child.all_field_names(prefix=current))
        return names


class Relationship(BaseModel):
    """Generic relationship between two schema objects."""

    from_object: str
    from_field: str
    to_object: str
    to_field: str
    relationship_type: str = "many_to_one"


class SchemaObject(BaseModel):
    """Database-independent representation of a table or collection."""

    name: str
    kind: SchemaObjectKind = SchemaObjectKind.TABLE
    description: str = ""
    fields: list[Field] = PydanticField(default_factory=list)

    def get_field(self, field_name: str) -> Field | None:
        """Find a field by name or dotted path."""
        parts = field_name.split(".")
        current_fields = self.fields
        target: Field | None = None
        for part in parts:
            found = next((f for f in current_fields if f.name == part), None)
            if not found:
                return None
            target = found
            current_fields = found.nested
        return target

    def all_field_paths(self) -> list[str]:
        """List all field paths including nested fields."""
        paths: list[str] = []
        for field in self.fields:
            paths.extend(field.all_field_names())
        return paths


class DatabaseSchema(BaseModel):
    """Normalized, database-independent schema model."""

    database_type: DatabaseType
    database_name: str
    schema_version: str = "1.0"
    last_updated: str | None = None
    description_generated_at: str | None = None
    objects: list[SchemaObject] = PydanticField(default_factory=list)
    relationships: list[Relationship] = PydanticField(default_factory=list)

    def get_object(self, name: str) -> SchemaObject | None:
        """Find a schema object by name."""
        return next((obj for obj in self.objects if obj.name.lower() == name.lower()), None)

    def validate_consistency(self) -> None:
        """Validate internal consistency of the schema before serialization.

        Ensures:
        1. All object names are unique.
        2. All field names within an object are unique.
        3. All relationships reference existing objects and fields.
        """
        # 1. Unique object names
        seen_objects = set()
        for obj in self.objects:
            obj_lower = obj.name.lower()
            if obj_lower in seen_objects:
                raise ValueError(f"Schema consistency error: duplicate object name '{obj.name}'")
            seen_objects.add(obj_lower)

            # 2. Unique direct field names
            seen_fields = set()
            for field in obj.fields:
                if field.name in seen_fields:
                    raise ValueError(
                        f"Schema consistency error: duplicate field '{field.name}' in object '{obj.name}'"
                    )
                seen_fields.add(field.name)

        # 3. Relationships reference existing objects and fields
        for rel in self.relationships:
            from_obj = self.get_object(rel.from_object)
            if not from_obj:
                raise ValueError(
                    f"Schema consistency error: relationship source object '{rel.from_object}' does not exist"
                )

            # Check from_field exists in from_object (either exact or dotted)
            from_field = from_obj.get_field(rel.from_field)
            if not from_field and rel.from_field not in from_obj.all_field_paths():
                raise ValueError(
                    f"Schema consistency error: relationship source field '{rel.from_field}' does not exist in '{rel.from_object}'"
                )

            to_obj = self.get_object(rel.to_object)
            if not to_obj:
                raise ValueError(
                    f"Schema consistency error: relationship target object '{rel.to_object}' does not exist"
                )

            to_field = to_obj.get_field(rel.to_field)
            if not to_field and rel.to_field not in to_obj.all_field_paths():
                raise ValueError(
                    f"Schema consistency error: relationship target field '{rel.to_field}' does not exist in '{rel.to_object}'"
                )
