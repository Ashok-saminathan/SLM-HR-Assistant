"""Prompt construction. Single source of truth so train / eval / API stay identical."""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are an HR policy assistant for Nimbus Works. "
    "Answer the employee's question using ONLY the policy excerpt provided. "
    "If the excerpt does not contain the answer, say so and set answerable to false. "
    'Reply with one JSON object and nothing else: '
    '{"answer": <short answer>, "source": <policy id or null>, "answerable": <true or false>}.'
)


def build_user_message(policy_id: str, policy_text: str, question: str) -> str:
    return f"Policy excerpt [id: {policy_id}]:\n{policy_text}\n\nQuestion: {question}"


def build_messages(policy_id: str, policy_text: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(policy_id, policy_text, question)},
    ]


def build_target(answer: str, source: str | None, answerable: bool) -> str:
    """The exact assistant string the model is trained to produce."""
    return json.dumps(
        {"answer": answer, "source": source, "answerable": answerable}, ensure_ascii=False
    )
