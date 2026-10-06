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


def repair_json(text: str) -> str:
    """Repair common SLM JSON defects:
    - Removes trailing commas before } and ] (string-literal aware)
    - Normalizes single-quoted keys and string literals to double quotes
    - Strips JavaScript/C++ single-line (//) and multi-line (/* */) comments
    - Replaces Python constants (True, False, None) outside string literals
    - Quotes unquoted keys ({ key: value })
    - Auto-closes unclosed brackets and braces if the response was truncated
    """
    if not text:
        return ""

    result: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    quote_char: str | None = None

    while i < n:
        c = text[i]

        if in_string:
            if escape:
                escape = False
                result.append(c)
            elif c == "\\":
                escape = True
                result.append(c)
            elif c == quote_char:
                in_string = False
                result.append('"')
            elif quote_char == "'" and c == '"':
                result.append('\\"')
            else:
                result.append(c)
            i += 1
            continue

        # Outside string: normalize opening quotes
        if c in ('"', "'"):
            in_string = True
            quote_char = c
            result.append('"')
            i += 1
            continue

        # Python boolean/None constants outside strings
        if text[i : i + 4] == "True" and (i + 4 == n or not (text[i + 4].isalnum() or text[i + 4] == "_")):
            result.append("true")
            i += 4
            continue
        if text[i : i + 5] == "False" and (i + 5 == n or not (text[i + 5].isalnum() or text[i + 5] == "_")):
            result.append("false")
            i += 5
            continue
        if text[i : i + 4] == "None" and (i + 4 == n or not (text[i + 4].isalnum() or text[i + 4] == "_")):
            result.append("null")
            i += 4
            continue

        # Single-line comment //
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue

        # Multi-line comment /* */
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        # Trailing comma before } or ]
        if c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ("}", "]"):
                i += 1
                continue

        result.append(c)
        i += 1

    cleaned = "".join(result)

    # Fix unquoted keys outside strings: { key: ... } or , key: ...
    cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)

    # Balance unclosed braces/brackets if generation was truncated
    stack: list[str] = []
    in_str = False
    esc = False
    for c in cleaned:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack and stack[-1] == c:
                stack.pop()

    if in_str:
        cleaned += '"'
    while stack:
        cleaned += stack.pop()

    return cleaned


def extract_json_block(text: str) -> dict[str, Any] | list[Any]:
    """Extract and parse a JSON object or array from a model response.

    Strips any <think> tags, extracts JSON from code fences or raw JSON substrings,
    repairs common SLM formatting quirks (trailing commas, comments, single quotes),
    and returns parsed Python data.
    """
    cleaned = clean_unreadable_characters(strip_think_tags(text))
    candidates: list[str] = []

    # 1. Try markdown code block ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", cleaned, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # 2. Try outermost { ... }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(cleaned[brace_start : brace_end + 1])
    elif brace_start != -1:
        # Truncated JSON starting with {
        candidates.append(cleaned[brace_start:])

    # 3. Try outermost [ ... ]
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidates.append(cleaned[bracket_start : bracket_end + 1])
    elif bracket_start != -1:
        # Truncated JSON starting with [
        candidates.append(cleaned[bracket_start:])

    # 4. Fallback: whole cleaned text
    candidates.append(cleaned.strip())

    last_err: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        # Pass 1: standard fast parse
        try:
            return json.loads(candidate, strict=False)
        except (json.JSONDecodeError, TypeError) as err:
            last_err = err

        # Pass 2: repair syntax defects (trailing commas, quotes, comments, unclosed braces)
        try:
            repaired = repair_json(candidate)
            return json.loads(repaired, strict=False)
        except (json.JSONDecodeError, TypeError) as err:
            last_err = err

    preview = cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
    raise ValueError(
        f"Failed to extract valid JSON from model response: {last_err}. Response preview: {preview!r}"
    ) from last_err

