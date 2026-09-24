"""Schema profiler that enriches table and column descriptions via SLM."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field as PydanticField

from query_processing.core.text_utils import extract_json_block
from query_processing.models.schema import (
    DatabaseSchema,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path("query_processing/prompts/schema_description.txt")
MAX_OBJECTS_PER_CALL = 5


class DescribedObject(BaseModel):
    name: str
    description: str = ""
    fields: dict[str, str] = PydanticField(default_factory=dict)


class SchemaDescriptionResponse(BaseModel):
    objects: list[DescribedObject] = PydanticField(default_factory=list)


def _format_fields_for_prompt(fields: list[Field], prefix: str = "") -> str:
    lines: list[str] = []
    for f in fields:
        field_path = f"{prefix}.{f.name}" if prefix else f.name
        samples_str = f" (samples: {', '.join(map(str, f.sample_values[:3]))})" if f.sample_values else ""
        lines.append(f"  - {field_path} ({f.type}){samples_str}")
        if f.nested:
            lines.append(_format_fields_for_prompt(f.nested, prefix=field_path))
    return "\n".join(lines)


def _format_object_context(obj: SchemaObject) -> str:
    header = f"Object: {obj.name} ({obj.kind.value})"
    fields_text = _format_fields_for_prompt(obj.fields)
    return f"{header}\nFields:\n{fields_text}\n"


class SchemaProfiler:
    """Enriches DatabaseSchema with descriptions using an SLM in bounded batches."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_PROMPT_PATH)
        if not self.prompt_path.exists():
            # Try absolute from cwd
            self.prompt_path = Path.cwd() / self.prompt_path

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Schema description prompt template not found at {self.prompt_path}")

    def _needs_enrichment(self, obj: SchemaObject, refresh: bool) -> bool:
        if refresh:
            return True
        if not obj.description:
            return True
        return any(not f.description for f in obj.fields)

    async def enrich_schema(
        self,
        schema: DatabaseSchema,
        *,
        refresh: bool = False,
    ) -> DatabaseSchema:
        """Enrich schema object and field descriptions.

        Only modifies description fields; never modifies authoritative schema metadata.
        Batches at most 5 objects per SLM call.
        """
        template = self._load_prompt_template()

        # Identify objects needing enrichment
        target_objects = [
            obj for obj in schema.objects if self._needs_enrichment(obj, refresh)
        ]

        if not target_objects:
            logger.info("All schema objects already have descriptions; skipping SLM enrichment.")
            return schema

        # Split into batches of at most 5
        batches = [
            target_objects[i : i + MAX_OBJECTS_PER_CALL]
            for i in range(0, len(target_objects), MAX_OBJECTS_PER_CALL)
        ]

        kind_singular = (
            "table" if schema.database_type.value == "postgresql" else "collection"
        )
        kind_plural = (
            "tables" if schema.database_type.value == "postgresql" else "collections"
        )

        for batch in batches:
            objects_context = "\n".join(_format_object_context(obj) for obj in batch)
            prompt = template.format(
                database_type=schema.database_type.value,
                database_name=schema.database_name,
                kind_singular=kind_singular,
                kind_plural=kind_plural,
                objects_context=objects_context,
            )

            try:
                raw_response = await self.provider.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
                json_data = extract_json_block(raw_response)
                parsed = SchemaDescriptionResponse.model_validate(json_data)

                # Merge descriptions into schema (AUTHORITATIVE METADATA IS NEVER OVERWRITTEN)
                for item in parsed.objects:
                    obj = schema.get_object(item.name)
                    if not obj:
                        continue

                    if item.description and (refresh or not obj.description):
                        obj.description = item.description.strip()

                    for field_name, field_desc in item.fields.items():
                        field = obj.get_field(field_name)
                        if field and field_desc and (refresh or not field.description):
                            field.description = field_desc.strip()

            except Exception as err:
                logger.warning(
                    "Schema description SLM call failed for batch [%s]: %s. "
                    "Preserving deterministic schema without destroying metadata.",
                    ", ".join(o.name for o in batch),
                    err,
                )

        schema.description_generated_at = datetime.now(timezone.utc).isoformat()
        return schema
