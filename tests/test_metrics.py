from src.evaluation.metrics import aggregate, fact_recall, failure_breakdown, score_example, token_f1
from src.model.output_validation import ParsedOutput

CASE = {"answerable": True, "gold_policy_id": "leave-annual", "key_facts": ["18"], "gold_answer": "You get 18 days."}
OOS = {"answerable": False, "gold_policy_id": None, "key_facts": [], "gold_answer": "refuse"}


def parsed(answer="You get 18 days.", source="leave-annual", answerable=True, valid=True):
    return ParsedOutput(valid=valid, answer=answer, source=source, answerable=answerable)


def test_token_f1_and_recall():
    assert token_f1("18 days", "18 days") == 1.0
    assert token_f1("", "x") == 0.0
    assert fact_recall("It is 18 days", ["18", "days"]) == 1.0
    assert fact_recall("nothing", ["18"]) == 0.0


def test_score_success_and_failure_modes():
    assert score_example(CASE, parsed(), "raw", "leave-annual")["task_success"]
    assert not score_example(CASE, parsed(answer="20 days"), "raw", "leave-annual")["task_success"]
    assert not score_example(CASE, parsed(source="leave-sick"), "raw", "leave-annual")["task_success"]
    assert not score_example(CASE, parsed(valid=False), "raw", "leave-annual")["task_success"]
    assert score_example(OOS, parsed(source=None, answerable=False), "raw", "leave-sick")["task_success"]
    assert not score_example(OOS, parsed(), "raw", "leave-sick")["task_success"]


def _row(case, p, ctx):
    s = score_example(case, p, "raw", ctx)
    s.update(answerable_gold=case["answerable"], pred_answerable=p.answerable, latency_s=1.0, new_tokens=10, tokens_per_s=10.0)
    return s


def test_aggregate_and_failures():
    rows = [_row(CASE, parsed(), "leave-annual"),
            _row(CASE, parsed(source=None, answerable=False), "leave-annual"),   # false refusal
            _row(CASE, parsed(source="leave-sick"), "leave-sick"),                # retrieval miss -> wrong source
            _row(OOS, parsed(), "leave-sick")]                                     # hallucinated answer
    m = aggregate(rows)
    assert m["n_cases"] == 4 and m["task_success_rate"] == 0.25
    assert m["retrieval_hit_rate"] == round(2 / 3, 4)
    assert m["latency_p50_s"] == 1.0
    f = failure_breakdown(rows)
    assert f["false_refusal"] == 1 and f["retrieval_miss"] == 1 and f["hallucinated_answer_for_out_of_scope"] == 1
