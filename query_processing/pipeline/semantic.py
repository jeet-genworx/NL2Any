"""Semantic analysis stage extracting subjective and objective components."""

import logging
from pathlib import Path
from query_processing.core.text_utils import extract_json_block
from query_processing.models.pipeline import SemanticAnalysisResult
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_PROMPT = Path("query_processing/prompts/semantic_analysis.txt")


class SemanticAnalyzer:
    """Extracts subjective concepts and objective constraints from the question."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_SEMANTIC_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Semantic analysis prompt file not found at {self.prompt_path}")

    async def analyze(self, question: str) -> SemanticAnalysisResult:
        """Analyze natural language query into subjective entities and analytical objectives."""
        template = self._load_prompt_template()
        prompt = template.format(question=question)

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=800,
            )
            data = extract_json_block(raw_response)
            return SemanticAnalysisResult.model_validate(data)
        except Exception as err:
            logger.warning("Semantic analysis SLM failed: %s. Returning empty fallback.", err)
            return SemanticAnalysisResult()
