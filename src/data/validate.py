"""Cleaning + validation rules for raw QA records (pure functions, unit tested)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

MIN_Q, MAX_Q = 8, 300
MIN_A, MAX_A = 5, 600


def normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def question_key(q: str) -> str:
    """Key used for de-duplication: case / punctuation / spacing insensitive."""
    return re.sub(r"[^a-z0-9 ]", "", normalize_text(q).lower()).strip()


def validate_record(rec: dict[str, Any], known_policy_ids: set[str]) -> str | None:
    """Return None if the record is valid, otherwise a short reason string."""
    for field_name in ("question", "answer", "answerable"):
        if field_name not in rec:
            return f"missing_{field_name}"
    q, a = normalize_text(rec["question"]), normalize_text(rec["answer"])
    if not q:
        return "empty_question"
    if not a:
        return "empty_answer"
    if not (MIN_Q <= len(q) <= MAX_Q):
        return "question_length"
    if not (MIN_A <= len(a) <= MAX_A):
        return "answer_length"
    if not isinstance(rec["answerable"], bool):
        return "answerable_not_bool"
    if rec["answerable"]:
        pid = rec.get("policy_id")
        if pid not in known_policy_ids:
            return "unknown_policy_id"
        facts = rec.get("key_facts") or []
        if not facts:
            return "no_key_facts"
        if not all(f.lower() in a.lower() for f in facts):
            return "key_fact_not_in_answer"
    else:
        if rec.get("policy_id") is not None:
            return "unanswerable_with_policy_id"
    return None


def clean_records(raw: list[dict[str, Any]], known_policy_ids: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    """Normalise, validate and de-duplicate. Returns (clean_records, drop_reason_counts)."""
    dropped: Counter = Counter()
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    for rec in raw:
        reason = validate_record(rec, known_policy_ids)
        if reason:
            dropped[reason] += 1
            continue
        rec = dict(rec)
        rec["question"] = normalize_text(rec["question"])
        rec["answer"] = normalize_text(rec["answer"])
        key = question_key(rec["question"])
        if key in seen:
            dropped["duplicate_question"] += 1
            continue
        seen.add(key)
        clean.append(rec)
    return clean, dropped
