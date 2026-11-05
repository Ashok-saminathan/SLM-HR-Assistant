import json

from src.data.build_dataset import make_raw_export, split_list, split_policy_ids
from src.data.policy_kb import POLICIES
from src.data.validate import clean_records, question_key, validate_record

IDS = {p["id"] for p in POLICIES}


def test_kb_key_facts_are_in_answers():
    for p in POLICIES:
        for q, a, facts in p["qa"]:
            assert all(f.lower() in a.lower() for f in facts), (p["id"], q)


def test_kb_answers_are_supported_by_policy_text():
    """Every number-like key fact in an answer must also appear in the policy excerpt (no invented facts)."""
    for p in POLICIES:
        text = p["text"].lower().replace("-", " ")
        for q, a, facts in p["qa"]:
            for f in facts:
                if any(ch.isdigit() for ch in f):
                    assert f.lower().replace("-", " ") in text, (p["id"], f)


def test_clean_removes_each_noise_type():
    clean, dropped = clean_records(make_raw_export(42), IDS)
    for reason in ("duplicate_question", "empty_answer", "missing_question", "unknown_policy_id",
                   "question_length", "key_fact_not_in_answer"):
        assert dropped[reason] > 0, reason
    assert all(validate_record(r, IDS) is None for r in clean)
    keys = [question_key(r["question"]) for r in clean]
    assert len(keys) == len(set(keys))


def test_validate_record_rules():
    ok = {"question": "How many days?", "answer": "18 days", "answerable": True, "policy_id": "leave-annual", "key_facts": ["18"]}
    assert validate_record(ok, IDS) is None
    assert validate_record({**ok, "answer": " "}, IDS) == "empty_answer"
    assert validate_record({**ok, "policy_id": "nope"}, IDS) == "unknown_policy_id"
    assert validate_record({**ok, "answerable": False}, IDS) == "unanswerable_with_policy_id"


def test_question_key_normalisation():
    assert question_key("  HOW many days??  ") == question_key("how many days?")


def test_policy_split_is_disjoint_and_complete():
    ids = sorted(IDS)
    tr, va, te = split_policy_ids(ids, 42)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert set(tr) | set(va) | set(te) == IDS
    assert len(va) >= 1 and len(te) >= 1


def test_split_is_deterministic():
    assert split_policy_ids(sorted(IDS), 42) == split_policy_ids(sorted(IDS), 42)
    assert split_list(list(range(20)), 1) == split_list(list(range(20)), 1)


def test_processed_files_have_no_leakage():
    """Run after `python -m src.data.build_dataset`: test questions/policies must not appear in train/val."""
    from src.data.build_dataset import PROCESSED_DIR, process
    from src.utils import PROJECT_ROOT, load_config, read_jsonl

    process(load_config(PROJECT_ROOT / "configs/data.yaml"))  # idempotent rebuild from data/raw
    train = read_jsonl(PROCESSED_DIR / "train.jsonl")
    val = read_jsonl(PROCESSED_DIR / "validation.jsonl")
    test = read_jsonl(PROCESSED_DIR / "test.jsonl")
    seen_q = {r["messages"][1]["content"].split("Question: ")[-1] for r in train + val}
    assert not (seen_q & {r["question"] for r in test})
    stats = json.loads((PROCESSED_DIR / "dataset_stats.json").read_text())
    assert not (set(stats["policy_split"]["test"]) & set(stats["policy_split"]["train"]))
    for r in train:  # supervised target must be valid JSON with the 3 fields
        tgt = json.loads(r["messages"][-1]["content"])
        assert set(tgt) == {"answer", "source", "answerable"}
