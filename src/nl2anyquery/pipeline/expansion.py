"""Deterministic relationship and schema expansion in Python."""

from nl2anyquery.models.pipeline import RelevantSchema
from nl2anyquery.models.schema import DatabaseSchema, DatabaseType, Relationship, SchemaObject


class SchemaExpander:
    """Expands selected objects with necessary relational links and returns RelevantSchema."""

    def expand(
        self,
        selected_object_names: list[str],
        schema: DatabaseSchema,
    ) -> RelevantSchema:
        """Deterministically expand selected objects with foreign keys and bridge tables."""
        if not selected_object_names:
            return RelevantSchema(objects=[], relationships=[])

        selected_set = {name.lower() for name in selected_object_names}
        expanded_names = set(selected_set)

        relevant_relationships: list[Relationship] = []

        if schema.database_type == DatabaseType.POSTGRESQL:
            # 1. Direct relationships between already selected objects
            for rel in schema.relationships:
                from_obj = rel.from_object.lower()
                to_obj = rel.to_object.lower()
                if from_obj in selected_set and to_obj in selected_set:
                    relevant_relationships.append(rel)

            # 2. Check for 1-hop bridge tables connecting selected objects
            # Example: orders and products selected -> order_items connects both
            for candidate in schema.objects:
                cand_lower = candidate.name.lower()
                if cand_lower in expanded_names:
                    continue

                # Find relationships where candidate links to selected objects
                outgoing = [
                    r for r in schema.relationships
                    if r.from_object.lower() == cand_lower and r.to_object.lower() in selected_set
                ]
                # If this candidate links to 2 or more of the selected objects, it's a join bridge table!
                target_objects = {r.to_object.lower() for r in outgoing}
                if len(target_objects) >= 2:
                    expanded_names.add(cand_lower)
                    relevant_relationships.extend(outgoing)

            # Re-check for any newly connected relationships among expanded names
            for rel in schema.relationships:
                from_obj = rel.from_object.lower()
                to_obj = rel.to_object.lower()
                if from_obj in expanded_names and to_obj in expanded_names:
                    if rel not in relevant_relationships:
                        relevant_relationships.append(rel)

        # Collect the actual SchemaObject instances
        relevant_objects: list[SchemaObject] = []
        for obj in schema.objects:
            if obj.name.lower() in expanded_names:
                relevant_objects.append(obj)

        return RelevantSchema(
            objects=relevant_objects,
            relationships=relevant_relationships,
        )
