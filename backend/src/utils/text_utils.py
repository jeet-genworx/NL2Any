"""Model response and text manipulation utilities."""

import json
import re
from typing import Any


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output.

    Handles both closed blocks `<think>...</think>` and unclosed `<think>...` blocks
    arising from generation length limits.
    """
    if not text:
        return ""

    # Strip closed think tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # In case there's an unclosed <think> tag at the beginning or elsewhere
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def strip_ansi_escapes(text: str) -> str:
    """Remove ANSI terminal escape codes (colors, underline, reset, etc.) from text."""
    if not text:
        return ""
    # Standard ANSI escape codes (\x1b[...m, \x1b[...K, etc.)
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    # OSC sequences
    cleaned = re.sub(r"\x1b\][^\x1b]*\x1b\\", "", cleaned)
    # Leftover raw bracket escape remnants like [4m or [0m
    cleaned = re.sub(r"\[[0-9;]+m", "", cleaned)
    return cleaned


def clean_unreadable_characters(text: str) -> str:
    """Strip ANSI escape sequences and non-printable control characters except standard whitespace."""
    if not text:
        return ""
    cleaned = strip_ansi_escapes(text)
    # Strip non-printable control chars, keeping \t (0x09), \n (0x0A), \r (0x0D)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
    return cleaned.strip()


def extract_json_block(text: str) -> dict[str, Any] | list[Any]:
    """Extract and parse a JSON object or array from a model response.

    Strips any <think> tags, extracts JSON from code fences or raw JSON substrings,
    and returns parsed Python data.
    """
    cleaned = strip_think_tags(text)

    # 1. Try markdown code block ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Try outermost { ... }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Try outermost [ ... ]
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidate = cleaned[bracket_start : bracket_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4. Fallback: try parsing whole text directly
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as err:
        preview = cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
        raise ValueError(
            f"Failed to extract valid JSON from model response: {err}. Response preview: {preview!r}"
        ) from err
