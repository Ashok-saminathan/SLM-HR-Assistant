"""Task metrics. Pure Python so they can be unit tested without a model."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from src.model.output_validation import ParsedOutput


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9.,%:]+", s.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def fact_recall(text: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 1.0
    low = text.lower()
    return sum(f.lower() in low for f in key_facts) / len(key_facts)


def score_example(case: dict[str, Any], parsed: ParsedOutput, raw_text: str, context_policy_id: str) -> dict[str, Any]:
    """Score one prediction against one held-out case.

    case keys: answerable, gold_policy_id, key_facts, gold_answer
    """
    gold_answerable = bool(case["answerable"])
    retrieval_hit = (context_policy_id == case["gold_policy_id"]) if gold_answerable else None
    row: dict[str, Any] = {
        "valid_json": parsed.valid,
        "retrieval_hit": retrieval_hit,
        "answerable_correct": parsed.valid and parsed.answerable == gold_answerable,
        "fact_recall_raw": fact_recall(raw_text, case["key_facts"]) if gold_answerable else None,
        "token_f1": token_f1(parsed.answer or raw_text, case["gold_answer"]) if gold_answerable else None,
    }
    if gold_answerable:
        row["source_correct"] = bool(parsed.valid and parsed.source == case["gold_policy_id"])
        row["task_success"] = bool(
            parsed.valid and parsed.answerable and row["source_correct"]
            and fact_recall(parsed.answer, case["key_facts"]) == 1.0
        )
    else:
        row["source_correct"] = None
        row["task_success"] = bool(parsed.valid and not parsed.answerable)  # correct refusal
    return row


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 4)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows = per-example dicts: score_example() output + latency_s, new_tokens, answerable_gold."""
    ans = [r for r in rows if r["answerable_gold"]]
    oos = [r for r in rows if not r["answerable_gold"]]
    ans_hit = [r for r in ans if r["retrieval_hit"]]
    lat = [r["latency_s"] for r in rows]
    toks = [r["new_tokens"] for r in rows]
    return {
        "n_cases": len(rows), "n_answerable": len(ans), "n_out_of_scope": len(oos),
        # response validity
        "valid_json_rate": _mean([float(r["valid_json"]) for r in rows]),
        # end-to-end task quality (retrieval + model)
        "task_success_rate": _mean([float(r["task_success"]) for r in rows]),
        "answerable_accuracy": _mean([float(r["answerable_correct"]) for r in rows]),
        "fact_recall_raw_mean": _mean([r["fact_recall_raw"] for r in ans]),
        "token_f1_mean": _mean([r["token_f1"] for r in ans]),
        "source_accuracy": _mean([float(r["source_correct"]) for r in ans]),
        # model-only quality (retrieval removed from the picture)
        "retrieval_hit_rate": _mean([float(r["retrieval_hit"]) for r in ans]),
        "task_success_given_retrieval_hit": _mean([float(r["task_success"]) for r in ans_hit]),
        "correct_refusal_rate_oos": _mean([float(r["task_success"]) for r in oos]),
        "false_refusal_rate_given_hit": _mean([float(r["valid_json"] and not r["pred_answerable"]) for r in ans_hit]),
        # speed
        "latency_mean_s": _mean(lat), "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
        "new_tokens_mean": _mean(toks),
        "tokens_per_s_mean": _mean([r["tokens_per_s"] for r in rows]),
    }


def failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        if r["task_success"]:
            continue
        if not r["valid_json"]:
            c["invalid_output_format"] += 1
        elif r["answerable_gold"] and r["retrieval_hit"] is False:
            c["retrieval_miss"] += 1
        elif r["answerable_gold"] and not r["pred_answerable"]:
            c["false_refusal"] += 1
        elif not r["answerable_gold"] and r["pred_answerable"]:
            c["hallucinated_answer_for_out_of_scope"] += 1
        else:
            c["wrong_or_incomplete_answer"] += 1
    return dict(c)


def summarize_latency(values: list[float]) -> dict[str, float | None]:
    return {"mean": round(statistics.fmean(values), 4) if values else None,
            "p50": _pct(values, 0.5), "p95": _pct(values, 0.95)}
