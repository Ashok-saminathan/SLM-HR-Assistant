"""Application service: validation -> retrieval -> prompt -> SLM -> output validation -> logging.

Kept free of FastAPI and torch imports so it can be unit-tested with a fake engine.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.data.policy_kb import POLICIES, REFUSAL_ANSWER
from src.model.output_validation import validate_output
from src.model.prompts import build_messages
from src.retrieval.retriever import BM25Retriever


class Engine(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> Any:  # returns object with .text
        ...


@dataclass
class AskResult:
    request_id: str
    answer: str
    answerable: bool
    source: str | None
    retrieved_policy_id: str
    degraded: bool
    latency_ms: float


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: Counter = Counter()
        self.latencies_ms: deque = deque(maxlen=1000)

    def record(self, latency_ms: float, **flags: bool) -> None:
        with self._lock:
            self.counts["requests"] += 1
            for k, v in flags.items():
                if v:
                    self.counts[k] += 1
            self.latencies_ms.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = sorted(self.latencies_ms)
            p = lambda q: round(lat[min(len(lat) - 1, int(q * (len(lat) - 1)))], 1) if lat else None
            return {**self.counts, "latency_ms_p50": p(0.5), "latency_ms_p95": p(0.95)}


class AssistantService:
    def __init__(self, engine: Engine, log_path: str | None = None, top_k: int = 1):
        self.engine = engine
        self.retriever = BM25Retriever(POLICIES)
        self.top_k = top_k
        self.known_ids = {p["id"] for p in POLICIES}
        self.metrics = Metrics()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()

    def ask(self, question: str) -> AskResult:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        hit = self.retriever.search(question, top_k=self.top_k)[0]
        messages = build_messages(hit.policy_id, hit.text, question)
        raw = self.engine.generate(messages).text
        parsed = validate_output(raw, context_policy_id=hit.policy_id, known_ids=self.known_ids)

        if parsed.valid:
            answer, answerable, source, degraded = parsed.answer, parsed.answerable, parsed.source, False
        else:  # safe fallback: never return unvalidated model text to the user
            answer, answerable, source, degraded = REFUSAL_ANSWER, False, None, True
        latency_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record(latency_ms, degraded=degraded, refused=not answerable)
        self._log({"request_id": rid, "question": question[:200], "retrieved": hit.policy_id,
                   "retrieval_score": round(hit.score, 3), "valid": parsed.valid, "errors": parsed.errors,
                   "answerable": answerable, "latency_ms": round(latency_ms, 1)})
        return AskResult(rid, answer, answerable, source, hit.policy_id, degraded, round(latency_ms, 1))

    def _log(self, record: dict[str, Any]) -> None:
        if not self.log_path:
            return
        with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
