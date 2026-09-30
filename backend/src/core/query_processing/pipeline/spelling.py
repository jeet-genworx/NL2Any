"""Spelling checker stage using SymSpell with request-scoped custom jargon."""

import importlib.resources
import logging
from pathlib import Path
import re
from typing import Any

from symspellpy import SymSpell, Verbosity
from symspellpy.editdistance import DistanceAlgorithm, EditDistance

from backend.src.schemas.pipeline import SpellingCorrectionItem, SpellingCorrectionResult

logger = logging.getLogger(__name__)


class SpellingChecker:
    """Performs SymSpell-based spelling correction on natural language queries."""

    def __init__(
        self,
        max_dictionary_edit_distance: int = 2,
        prefix_length: int = 7,
        dictionary_path: str | Path | None = None,
    ) -> None:
        self.max_dictionary_edit_distance = max_dictionary_edit_distance
        self.prefix_length = prefix_length
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance,
            prefix_length=prefix_length,
        )

        if dictionary_path:
            self.sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        else:
            default_dict = (
                importlib.resources.files("symspellpy")
                / "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(default_dict, term_index=0, count_index=1)

        self._comparer = EditDistance(DistanceAlgorithm.DAMERAU_OSA)

    def check(
        self,
        question: str,
        jargons: list[str] | None = None,
    ) -> SpellingCorrectionResult:
        """Correct misspellings in the question using SymSpell and user-provided jargon.

        Args:
            question: Raw user question string.
            jargons: Optional list of request-specific domain terms / jargon words.

        Returns:
            SpellingCorrectionResult containing original, corrected, and correction items.
        """
        if not question or not question.strip():
            return SpellingCorrectionResult(
                original_question=question,
                corrected_question=question,
                corrections=[],
                jargons_applied=[],
            )

        jargons_clean = [j.strip() for j in (jargons or []) if j and j.strip()]
        jargons_map = {j.lower(): j for j in jargons_clean}

        # Tokenize by word boundaries, keeping whitespace and punctuation
        tokens = re.split(r"(\b\w+\b)", question)
        result: list[str] = []
        corrections: list[SpellingCorrectionItem] = []

        for tok in tokens:
            if not tok:
                continue

            tok_lower = tok.lower()

            # 1. Exact match in user-provided jargon: preserve user casing
            if tok_lower in jargons_map:
                result.append(jargons_map[tok_lower])
                continue

            # 2. Identifiers containing underscores or digits: preserve as technical tokens
            if "_" in tok or any(c.isdigit() for c in tok):
                result.append(tok)
                continue

            # 3. Alphabetical words: check for typos
            if tok.isalpha():
                # First check if this token is a typo of any user-provided jargon term
                jargon_matched: str | None = None
                for j_low, j_orig in jargons_map.items():
                    d = self._comparer.compare(tok_lower, j_low, self.max_dictionary_edit_distance)
                    if 0 <= d <= self.max_dictionary_edit_distance:
                        jargon_matched = j_orig
                        break

                if jargon_matched:
                    if jargon_matched != tok:
                        corrections.append(
                            SpellingCorrectionItem(original=tok, corrected=jargon_matched)
                        )
                    result.append(jargon_matched)
                    continue

                # Query SymSpell standard vocabulary
                suggs = self.sym_spell.lookup(
                    tok,
                    Verbosity.TOP,
                    max_edit_distance=self.max_dictionary_edit_distance,
                    include_unknown=True,
                    transfer_casing=True,
                )
                if suggs and suggs[0].distance > 0 and suggs[0].term != tok:
                    corrected_term = suggs[0].term
                    corrections.append(
                        SpellingCorrectionItem(original=tok, corrected=corrected_term)
                    )
                    result.append(corrected_term)
                else:
                    result.append(tok)
            else:
                result.append(tok)

        corrected_text = "".join(result)
        return SpellingCorrectionResult(
            original_question=question,
            corrected_question=corrected_text,
            corrections=corrections,
            jargons_applied=jargons_clean,
        )
