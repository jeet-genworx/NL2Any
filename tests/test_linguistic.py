"""Tests for spaCy deterministic linguistic analysis."""

from query_processing.nlp.linguistic import LinguisticAnalyzer


def test_linguistic_analyzer_extraction():
    analyzer = LinguisticAnalyzer()
    text = "Show all urgent support tickets filed by customers in Chicago"
    analysis = analyzer.analyze(text)

    # Nouns / noun chunks should capture domain concepts
    assert any("ticket" in n for n in analysis.nouns)
    assert any("customer" in n for n in analysis.nouns)

    # Verbs should capture 'show' or 'file'
    assert any(v in ("show", "file") for v in analysis.verbs)

    # Entities should capture Chicago as GPE
    entity_texts = [e.text for e in analysis.entities]
    assert "Chicago" in entity_texts
    chicago_ent = next(e for e in analysis.entities if e.text == "Chicago")
    assert chicago_ent.label in ("GPE", "LOC")


def test_linguistic_analyzer_empty_input():
    analyzer = LinguisticAnalyzer()
    analysis = analyzer.analyze("")
    assert analysis.nouns == []
    assert analysis.verbs == []
    assert analysis.entities == []
