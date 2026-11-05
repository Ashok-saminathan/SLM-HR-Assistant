"""Output validation: parse and check the model's raw text before returning it.

A response is VALID only if it is a JSON object with exactly the expected fields and
types, a non-empty answer, and (when answerable) a source that is a known policy id
that was actually present in the prompt context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_ANSWER_CHARS = 600


@dataclass
class ParsedOutput:
    valid: bool
    answer: str = ""
    source: str | None = None
    answerable: bool = False
    errors: list[str] = field(default_factory=list)
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "answer": self.answer,
            "source": self.source,
            "answerable": self.answerable,
            "errors": self.errors,
        }


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} block (models sometimes add prose or code fences)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_output(raw: str, context_policy_id: str | None = None, known_ids: set[str] | None = None) -> ParsedOutput:
    out = ParsedOutput(valid=False, raw=raw)
    blob = _extract_json_object(raw or "")
    if blob is None:
        out.errors.append("no_json_object")
        return out
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        out.errors.append("invalid_json")
        return out
    if not isinstance(obj, dict):
        out.errors.append("not_an_object")
        return out

    required = {"answer", "source", "answerable"}
    missing = required - set(obj)
    if missing:
        out.errors.append("missing_fields:" + ",".join(sorted(missing)))
        return out

    answer, source, answerable = obj["answer"], obj["source"], obj["answerable"]
    if not isinstance(answer, str) or not answer.strip():
        out.errors.append("empty_or_non_string_answer")
    elif len(answer) > MAX_ANSWER_CHARS:
        out.errors.append("answer_too_long")
    if not isinstance(answerable, bool):
        out.errors.append("answerable_not_bool")
    if source is not None and not isinstance(source, str):
        out.errors.append("source_not_string_or_null")
    if isinstance(answerable, bool):
        if answerable and source is None:
            out.errors.append("answerable_without_source")
        if answerable and isinstance(source, str):
            if known_ids is not None and source not in known_ids:
                out.errors.append("unknown_source_id")
            elif context_policy_id is not None and source != context_policy_id:
                out.errors.append("source_not_in_context")

    out.answer = answer.strip() if isinstance(answer, str) else ""
    out.source = source if isinstance(source, str) else None
    out.answerable = answerable if isinstance(answerable, bool) else False
    out.valid = not out.errors
    return out
