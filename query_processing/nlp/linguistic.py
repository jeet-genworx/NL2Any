"""Deterministic linguistic analyzer using spaCy."""

from typing import Any
from pydantic import BaseModel, Field
import spacy


class LinguisticEntity(BaseModel):
    """Named entity extracted by spaCy."""

    text: str
    label: str


class LinguisticAnalysis(BaseModel):
    """Linguistic extraction result containing nouns, verbs, and entities."""

    nouns: list[str] = Field(default_factory=list)
    verbs: list[str] = Field(default_factory=list)
    entities: list[LinguisticEntity] = Field(default_factory=list)


class LinguisticAnalyzer:
    """Extracts nouns, verbs, and entities from user questions without SLM."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Any = None

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' is not installed. "
                    f"Please install it using: uv run python -m spacy download {self.model_name}"
                )
        return self._nlp

    def analyze(self, text: str) -> LinguisticAnalysis:
        """Deterministically extract nouns, verbs, and entities from input text."""
        if not text or not text.strip():
            return LinguisticAnalysis()

        nlp = self._get_nlp()
        doc = nlp(text)

        # Extract nouns and noun chunks
        noun_set: set[str] = set()
        # Noun chunks give meaningful phrases like 'support tickets', 'total amount'
        for chunk in doc.noun_chunks:
            chunk_clean = chunk.text.strip().lower()
            if chunk_clean:
                noun_set.add(chunk_clean)

        # Individual noun tokens
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                noun_set.add(token.text.lower())

        # Extract verbs
        verb_set: set[str] = set()
        for token in doc:
            if token.pos_ == "VERB":
                verb_set.add(token.lemma_.lower())
                verb_set.add(token.text.lower())

        # Extract named entities
        entities: list[LinguisticEntity] = []
        for ent in doc.ents:
            entities.append(
                LinguisticEntity(
                    text=ent.text,
                    label=ent.label_,
                )
            )

        return LinguisticAnalysis(
            nouns=sorted(noun_set),
            verbs=sorted(verb_set),
            entities=entities,
        )
