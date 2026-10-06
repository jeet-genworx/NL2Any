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


# English stopwords to exempt from spell checking and jargon fuzzy matching
ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "an", "and", "another", "any", "anyhow",
    "anyone", "anything", "anyway", "anywhere", "are", "aren't", "around", "as", "at",
    "be", "became", "because", "become", "becomes", "becoming", "been", "before",
    "beforehand", "behind", "being", "below", "beside", "besides", "between",
    "beyond", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do",
    "does", "doesn't", "doing", "done", "don't", "down", "during", "each", "either", "else",
    "elsewhere", "enough", "even", "ever", "every", "everyone", "everything",
    "everywhere", "except", "few", "for", "former", "formerly", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "hence", "her", "here",
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
    "himself", "his", "how", "however", "i", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "just", "least", "less", "many", "may", "me", "meanwhile",
    "might", "more", "moreover", "most", "mostly", "much", "must", "my",
    "myself", "neither", "never", "nevertheless", "next", "no", "nobody",
    "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off",
    "often", "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "own", "per",
    "perhaps", "quite", "rather", "same", "several", "she", "should", "shouldn't", "since",
    "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "this", "those",
    "though", "through", "throughout", "thru", "thus", "to", "together",
    "too", "toward", "towards", "under", "unless", "until", "up", "upon",
    "us", "using", "various", "versus", "very", "via", "vs", "was", "wasn't", "we",
    "well", "were", "weren't", "what", "whatever", "when", "whence", "whenever",
    "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
    "wherever", "whether", "which", "while", "whither", "who", "whoever",
    "whole", "whom", "whose", "why", "will", "won't", "with", "within", "without",
    "would", "wouldn't", "yet", "you", "your", "yours", "yourself", "yourselves",
})


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
        # Sort jargons by length descending so longer compound/hyphenated phrases match first
        jargons_sorted = sorted(jargons_clean, key=len, reverse=True)
        jargons_map = {j.lower(): j for j in jargons_clean}

        corrections: list[SpellingCorrectionItem] = []
        placeholders: dict[str, str] = {}
        current_text = question

        # 1. Exact or case-insensitive match for each user-provided jargon (including hyphenated/compound terms)
        for idx, j_term in enumerate(jargons_sorted):
            escaped = re.escape(j_term)
            lead = r"(?<![\w-])" if re.match(r"[\w-]", j_term[0]) else ""
            trail = r"(?![\w-])" if re.match(r"[\w-]", j_term[-1]) else ""
            pattern = lead + escaped + trail
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            if matches:
                ph = f"__JARGON_PH_{idx}__"
                placeholders[ph] = j_term
                for m in reversed(matches):
                    orig_span = m.group(0)
                    if orig_span != j_term:
                        corrections.append(
                            SpellingCorrectionItem(original=orig_span, corrected=j_term)
                        )
                    current_text = current_text[:m.start()] + ph + current_text[m.end():]

        # Tokenize by word boundaries or jargon placeholders, keeping whitespace and punctuation
        tokens = re.split(r"(\b\w+\b|__JARGON_PH_\d+__)", current_text)
        result: list[str] = []

        for tok in tokens:
            if not tok:
                continue

            # Restore matched jargon placeholder
            if tok in placeholders:
                result.append(placeholders[tok])
                continue

            tok_lower = tok.lower()

            # Exact match in user-provided jargon: preserve user casing
            if tok_lower in jargons_map:
                result.append(jargons_map[tok_lower])
                continue

            # Identifiers containing underscores, digits, or hyphens: preserve as technical tokens
            if "_" in tok or "-" in tok or any(c.isdigit() for c in tok):
                result.append(tok)
                continue

            # Substep: Stop words - exempt from spell checking and jargon typo matching
            if tok_lower in ENGLISH_STOPWORDS:
                result.append(tok)
                continue

            # Alphabetical words: check for typos
            if tok.isalpha():
                # First check if this token is a typo of any user-provided jargon term
                jargon_matched: str | None = None
                for j_low, j_orig in jargons_map.items():
                    # Scale maximum allowed edit distance based on jargon term length:
                    # - 1-2 chars (e.g. 'AB'): exact match only (0 edits) to prevent false positives with short words
                    # - 3-4 chars: at most 1 edit
                    # - > 4 chars: up to max_dictionary_edit_distance (default 2)
                    if len(j_low) <= 2:
                        max_j_dist = 0
                    elif len(j_low) <= 4:
                        max_j_dist = 1
                    else:
                        max_j_dist = self.max_dictionary_edit_distance

                    if abs(len(tok_lower) - len(j_low)) > max_j_dist:
                        continue

                    d = self._comparer.compare(tok_lower, j_low, max_j_dist)
                    if 0 <= d <= max_j_dist:
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
                # Only treat as correction if SymSpell found a valid dictionary match within max_edit_distance
                # and the letters actually differ (never treat an unknown word lowercased by SymSpell as a correction)
                if (
                    suggs
                    and 0 < suggs[0].distance <= self.max_dictionary_edit_distance
                    and suggs[0].term.lower() != tok_lower
                ):
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
