"""Join path selection: deciding how the selected tables connect.

The table selector answers "which tables is this question about". It does not
answer "how do those tables join", and when the answer is "through two tables
nobody selected" the generator is left to invent a join condition. This stage
closes that gap: every real foreign-key path between the selected tables is
enumerated deterministically, and the model's only job is to pick which of
those real paths the question means.

Structure comes from the graph, semantics from the model, and the final say
from Python -- a chosen path id that was not offered is discarded, and a choice
that connects nothing falls back to the deterministic shortest paths.
"""

import logging
from pathlib import Path

from backend.src.config import settings
from backend.src.utils.text_utils import extract_json_block
from backend.src.schemas.pipeline import (
    JoinPathCandidate,
    JoinPathResolution,
    QuestionAnalysis,
)
from backend.src.data.models.schema import DatabaseSchema
from backend.src.core.query_processing.pipeline.join_graph import (
    JoinGraph,
    build_join_path_candidates,
)
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

_DESCRIPTION_LIMIT = 200


def _get_path_selector_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/join_path_selection.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "join_path_selection.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_PATH_SELECTOR_PROMPT = _get_path_selector_prompt_path()


def _truncate(text: str) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= _DESCRIPTION_LIMIT else clean[:_DESCRIPTION_LIMIT].rstrip() + "..."


def format_candidates(
    candidates: list[JoinPathCandidate],
    schema: DatabaseSchema,
) -> str:
    """Render candidates for the prompt, with the descriptions that disambiguate them.

    The intermediate tables' descriptions are the whole point: `orders` versus
    `subscriptions` is not a structural difference, so without them the model
    has nothing to choose on but path length.
    """
    lines: list[str] = []
    for candidate in candidates:
        if candidate.connected:
            arrow = " -> ".join(candidate.tables)
            lines.append(f"[{candidate.path_id}] {arrow}  ({candidate.hops} hops)")
            for edge in candidate.edges:
                lines.append(
                    f"    join: {edge.from_object}.{edge.from_field}"
                    f" = {edge.to_object}.{edge.to_field}  ({edge.relationship_type})"
                )
            # Only the intermediate tables need explaining; the endpoints were
            # already chosen and described by the table selector.
            for table in candidate.tables[1:-1]:
                obj = schema.get_object(table)
                if obj and obj.description:
                    lines.append(f"    via {table}: {_truncate(obj.description)}")
        else:
            lines.append(f"[{candidate.path_id}] {candidate.tables[0]}  (STANDALONE - {candidate.note})")
            obj = schema.get_object(candidate.tables[0])
            if obj and obj.description:
                lines.append(f"    {candidate.tables[0]}: {_truncate(obj.description)}")
    return "\n".join(lines)


def _one_path_per_pair(chosen: list[JoinPathCandidate]) -> list[JoinPathCandidate]:
    """Keep a single path per endpoint pair, preferring the shortest.

    Two tables need one route between them, not six. Asked to connect
    discrepancies to customers, a model will happily return every candidate it
    was shown and justify each one, and each redundant path drags its
    intermediate tables into the schema slice -- which inflates the planner and
    generator prompts and hands them columns the question never mentioned. The
    choice of WHICH route is the model's; the fact that one is enough is not.

    Standalone candidates are kept as-is: each covers a different object, so
    there is nothing to deduplicate.
    """
    best: dict[tuple[str, ...], JoinPathCandidate] = {}
    standalone: list[JoinPathCandidate] = []
    for candidate in chosen:
        if not candidate.connected:
            standalone.append(candidate)
            continue
        key = tuple(sorted(name.lower() for name in candidate.endpoints))
        incumbent = best.get(key)
        if incumbent is None or (candidate.hops, candidate.tables) < (
            incumbent.hops,
            incumbent.tables,
        ):
            best[key] = candidate
    # Preserve the model's ordering for the paths that survive.
    kept = list(best.values()) + standalone
    order = {c.path_id: i for i, c in enumerate(chosen)}
    return sorted(kept, key=lambda c: order[c.path_id])


class JoinPathSelector:
    """Resolves how selected schema objects connect, choosing among real FK paths."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
        max_hops: int | None = None,
        max_paths_per_pair: int | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_PATH_SELECTOR_PROMPT)
        self.max_hops = max_hops if max_hops is not None else settings.join_path_max_hops
        self.max_paths_per_pair = (
            max_paths_per_pair
            if max_paths_per_pair is not None
            else settings.join_path_max_paths_per_pair
        )
        self.max_candidates = (
            max_candidates if max_candidates is not None else settings.join_path_max_candidates
        )

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Join path selection prompt not found at {self.prompt_path}")

    async def resolve(
        self,
        question_analysis: QuestionAnalysis,
        selected_objects: list[str],
        schema: DatabaseSchema,
    ) -> JoinPathResolution:
        """Choose the join paths connecting the selected objects.

        Returns a resolution whose `resolved_objects` is what the expander should
        materialize: the selected objects the question still needs, plus every
        intermediate table the chosen paths walk through.
        """
        selected = list(
            dict.fromkeys(name for name in (selected_objects or []) if name and name.strip())
        )

        # Nothing to connect: a single object is already a complete schema slice.
        if len(selected) < 2:
            return JoinPathResolution(
                selected_objects=selected,
                resolved_objects=selected,
                reason="Fewer than two objects selected; no join path required.",
            )

        graph = JoinGraph(schema)
        candidates = build_join_path_candidates(
            selected_object_names=selected,
            schema=schema,
            max_hops=self.max_hops,
            max_paths_per_pair=self.max_paths_per_pair,
            max_candidates=self.max_candidates,
            graph=graph,
        )

        connected = [c for c in candidates if c.connected]

        # With no path between any pair there is nothing to choose: asking the
        # model to pick from standalone-only candidates could only drop tables
        # for no reason. This is also the MongoDB case, whose schema has no
        # foreign keys at all.
        if not connected:
            return self._build_resolution(
                selected=selected,
                candidates=candidates,
                chosen=candidates,
                slm_invoked=False,
                fallback_used=False,
                reason=(
                    "No foreign-key path exists between any of the selected objects; "
                    "keeping all of them unchanged."
                ),
            )

        chosen, slm_invoked, fallback_used, reason = await self._choose(
            question_analysis=question_analysis,
            candidates=candidates,
            schema=schema,
        )

        return self._build_resolution(
            selected=selected,
            candidates=candidates,
            chosen=chosen,
            slm_invoked=slm_invoked,
            fallback_used=fallback_used,
            reason=reason,
        )

    async def _choose(
        self,
        question_analysis: QuestionAnalysis,
        candidates: list[JoinPathCandidate],
        schema: DatabaseSchema,
    ) -> tuple[list[JoinPathCandidate], bool, bool, str]:
        """Ask the model which paths fit, falling back to shortest paths on failure."""
        template = self._load_prompt_template()
        prompt = template.format(
            question=question_analysis.question,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            selected_objects=", ".join(
                dict.fromkeys(name for c in candidates for name in c.endpoints)
            ),
            candidates_context=format_candidates(candidates, schema),
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=settings.join_path_selector_max_tokens,
            )
            data = extract_json_block(raw_response)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

            by_id = {c.path_id: c for c in candidates}
            raw_ids = data.get("chosen_paths") or []
            if not isinstance(raw_ids, list):
                raise ValueError("'chosen_paths' must be a list of path ids.")

            chosen: list[JoinPathCandidate] = []
            for item in raw_ids:
                path_id = str(item).strip()
                candidate = by_id.get(path_id)
                if candidate is None:
                    logger.warning("Discarded hallucinated/non-candidate join path id: %r", item)
                    continue
                if candidate not in chosen:
                    chosen.append(candidate)

            if not chosen:
                raise ValueError("Model chose no valid join paths.")

            chosen = _one_path_per_pair(chosen)
            reason = str(data.get("reason") or "").strip()
            return chosen, True, False, reason

        except Exception as err:
            logger.warning(
                "Join path selection SLM call failed: %s. Falling back to shortest paths.", err
            )
            return (
                self._deterministic_choice(candidates),
                True,
                True,
                f"Fallback to shortest join paths due to error: {err}",
            )

    def _deterministic_choice(
        self,
        candidates: list[JoinPathCandidate],
    ) -> list[JoinPathCandidate]:
        """Shortest path per pair of endpoints, plus every standalone object.

        The conservative choice: it connects everything it can and drops nothing,
        which is the right behaviour when the model's judgement is unavailable.
        """
        best: dict[tuple[str, ...], JoinPathCandidate] = {}
        standalone: list[JoinPathCandidate] = []
        for candidate in candidates:
            if not candidate.connected:
                standalone.append(candidate)
                continue
            key = tuple(sorted(name.lower() for name in candidate.endpoints))
            incumbent = best.get(key)
            if incumbent is None or (candidate.hops, candidate.tables) < (
                incumbent.hops,
                incumbent.tables,
            ):
                best[key] = candidate
        return list(best.values()) + standalone

    def _build_resolution(
        self,
        selected: list[str],
        candidates: list[JoinPathCandidate],
        chosen: list[JoinPathCandidate],
        slm_invoked: bool,
        fallback_used: bool,
        reason: str,
    ) -> JoinPathResolution:
        """Derive the final object set and the trace fields from the chosen paths."""
        covered: dict[str, str] = {}
        for candidate in chosen:
            for table in candidate.tables:
                covered.setdefault(table.lower(), table)

        # Selected objects keep their original order so the slice stays readable;
        # intermediate tables follow, sorted for determinism.
        selected_lower = {name.lower() for name in selected}
        resolved = [covered[name.lower()] for name in selected if name.lower() in covered]
        connectors = sorted(
            name for key, name in covered.items() if key not in selected_lower
        )
        resolved.extend(connectors)

        dropped = [name for name in selected if name.lower() not in covered]

        # An object no chosen edge touches cannot be joined to the rest, which
        # the planner needs to know about rather than discover by generating a
        # query that cannot run.
        joined: set[str] = set()
        for candidate in chosen:
            for edge in candidate.edges:
                joined.add(edge.from_object.lower())
                joined.add(edge.to_object.lower())
        unjoinable = (
            [name for name in resolved if name.lower() not in joined] if len(resolved) > 1 else []
        )

        return JoinPathResolution(
            selected_objects=selected,
            candidates=candidates,
            chosen_path_ids=[c.path_id for c in chosen],
            resolved_objects=resolved,
            connector_objects=connectors,
            dropped_objects=dropped,
            unjoinable_objects=unjoinable,
            slm_invoked=slm_invoked,
            fallback_used=fallback_used,
            reason=reason,
        )
