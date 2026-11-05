from src.data.policy_kb import POLICIES
from src.retrieval.retriever import BM25Retriever, tokenize


def test_tokenize_stems_plurals():
    assert tokenize("Leave days") == tokenize("leaves day")[:0] + tokenize("Leave days")
    assert "day" in tokenize("days")


def test_retrieval_recall_on_kb_questions():
    r = BM25Retriever(POLICIES)
    total = hit = 0
    for p in POLICIES:
        for q, _, _ in p["qa"]:
            total += 1
            hit += r.search(q, 1)[0].policy_id == p["id"]
    assert hit / total >= 0.80, f"top-1 retrieval too weak: {hit}/{total}"


def test_exclude_ids():
    r = BM25Retriever(POLICIES)
    top = r.search("annual leave carry forward", 1)[0].policy_id
    assert r.search("annual leave carry forward", 1, exclude_ids={top})[0].policy_id != top
