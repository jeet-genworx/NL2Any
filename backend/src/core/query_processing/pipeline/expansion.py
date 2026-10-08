"""Deterministic materialization of a selected table set into a RelevantSchema."""

from backend.src.schemas.pipeline import RelevantSchema
from backend.src.data.models.schema import DatabaseSchema, Relationship


class SchemaExpander:
    """Turns a set of object names into the schema slice the planner is given."""

    def materialize(
        self,
        selected_object_names: list[str],
        relationships: list[Relationship],
        schema: DatabaseSchema,
    ) -> RelevantSchema:
        """Build the slice from exactly the selected objects and relationships.

        No table is added and none is dropped: the table selector already chose
        from the full breadth-first neighborhood, bridge tables included, so
        inventing further members here would silently overrule it. What this adds
        is the body of each table -- columns, types, descriptions, sample values --
        which the selector decided about by name and description alone and which
        the planner needs to pick fields.

        Relationships are filtered to the selected set rather than trusted, so a
        slice can never carry an edge to a table that is not in it.
        """
        if not selected_object_names:
            return RelevantSchema(objects=[], relationships=[])

        selected = {name.lower() for name in selected_object_names}
        objects = [obj for obj in schema.objects if obj.name.lower() in selected]
        kept: list[Relationship] = []
        for rel in relationships:
            if rel.from_object.lower() in selected and rel.to_object.lower() in selected:
                if rel not in kept:
                    kept.append(rel)
        return RelevantSchema(objects=objects, relationships=kept)
