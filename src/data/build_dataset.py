"""Dataset pipeline:  raw export  ->  clean/validate/dedupe  ->  grouped split  ->  chat JSONL.

Usage:
    python -m src.data.build_dataset            # runs both stages
    python -m src.data.build_dataset --stage raw
    python -m src.data.build_dataset --stage process

Split strategy: GROUPED BY POLICY. All questions about one policy go to the same split,
so the test set contains policies the model never saw during training (no leakage via
paraphrased questions about the same facts). Out-of-scope questions are split 60/20/20.

Training / validation files contain ready-to-train chat `messages`. The test file only
stores the question, the gold policy id and key facts; the evaluation script builds the
prompt context at run time (retrieved or oracle) so retrieval can be measured separately.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from src.data.policy_kb import OUT_OF_SCOPE_QUESTIONS, POLICIES, REFUSAL_ANSWER
from src.data.validate import clean_records
from src.model.prompts import build_messages, build_target
from src.retrieval.retriever import BM25Retriever
from src.utils import PROJECT_ROOT, load_config, read_jsonl, write_json, write_jsonl

RAW_PATH = PROJECT_ROOT / "data/raw/hr_qa_raw.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
POLICY_PATH = PROJECT_ROOT / "data/processed/policies.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports/dataset_report.md"


# ----------------------------------------------------------------------------- raw
def make_raw_export(seed: int = 42) -> list[dict]:
    """Simulate a messy raw export: authored QA pairs plus realistic noise.

    The noise (duplicates, empty answers, unknown ids, wrong facts ...) is injected
    deterministically so the cleaning step has something real to remove and report.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    n = 0
    for p in POLICIES:
        for q, a, facts in p["qa"]:
            n += 1
            rows.append({"id": f"qa-{n:04d}", "policy_id": p["id"], "question": q, "answer": a,
                         "key_facts": facts, "answerable": True})
    for q in OUT_OF_SCOPE_QUESTIONS:
        n += 1
        rows.append({"id": f"qa-{n:04d}", "policy_id": None, "question": q, "answer": REFUSAL_ANSWER,
                     "key_facts": [], "answerable": False})

    good = list(rows)
    noise: list[dict] = []
    for rec in rng.sample([r for r in good if r["answerable"]], 8):  # near-duplicates
        d = dict(rec); d["id"] += "-dup"
        d["question"] = "  " + rec["question"].upper().rstrip("?") + " ??  "
        noise.append(d)
    for rec in rng.sample(good, 3):  # empty answer
        d = dict(rec); d["id"] += "-noans"; d["question"] += " (again)"; d["answer"] = "   "
        noise.append(d)
    for rec in rng.sample(good, 3):  # missing question field
        d = {k: v for k, v in rec.items() if k != "question"}; d["id"] += "-noq"
        noise.append(d)
    for rec in rng.sample([r for r in good if r["answerable"]], 3):  # unknown policy id
        d = dict(rec); d["id"] += "-badpid"; d["policy_id"] = "leave-unknown"; d["question"] += " Please explain."
        noise.append(d)
    for rec in rng.sample(good, 3):  # question too short
        d = dict(rec); d["id"] += "-short"; d["question"] = "leave?"
        noise.append(d)
    for rec in rng.sample([r for r in good if r["answerable"]], 3):  # answer lacks the key fact
        d = dict(rec); d["id"] += "-wrongfact"; d["question"] += " Tell me."; d["answer"] = "Please check with your manager."
        noise.append(d)
    rows = good + noise
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------------- process
def split_policy_ids(ids: list[str], seed: int, ratios=(0.7, 0.15, 0.15)):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    return ids[n_val + n_test:], ids[:n_val], ids[n_val: n_val + n_test]


def split_list(items: list, seed: int, ratios=(0.6, 0.2, 0.2)):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val, n_test = round(n * ratios[1]), round(n * ratios[2])
    return items[n_val + n_test:], items[:n_val], items[n_val: n_val + n_test]


def to_chat_example(rec: dict, retriever: BM25Retriever, policy_by_id: dict, rng: random.Random,
                    wrong_context_rate: float, allow_wrong_context: bool) -> dict:
    """Build one supervised example. Three kinds:
       answerable   : gold policy text as context  -> grounded answer
       wrong-context: answerable Q but a DIFFERENT policy as context -> refusal (teaches grounding)
       out-of-scope : retrieved context            -> refusal
    """
    if rec["answerable"]:
        use_wrong = allow_wrong_context and rng.random() < wrong_context_rate
        if use_wrong:
            hit = retriever.search(rec["question"], top_k=1, exclude_ids={rec["policy_id"]})[0]
            kind, ctx_id, ctx_text = "wrong_context", hit.policy_id, hit.text
            target = build_target(REFUSAL_ANSWER, None, False)
        else:
            p = policy_by_id[rec["policy_id"]]
            kind, ctx_id, ctx_text = "answerable", p["id"], p["text"]
            target = build_target(rec["answer"], p["id"], True)
    else:
        hit = retriever.search(rec["question"], top_k=1)[0]
        kind, ctx_id, ctx_text = "out_of_scope", hit.policy_id, hit.text
        target = build_target(REFUSAL_ANSWER, None, False)
    msgs = build_messages(ctx_id, ctx_text, rec["question"])
    msgs.append({"role": "assistant", "content": target})
    return {"id": rec["id"], "kind": kind, "context_policy_id": ctx_id, "messages": msgs}


def process(cfg: dict) -> dict:
    seed = cfg["seed"]
    raw = read_jsonl(RAW_PATH)
    policy_by_id = {p["id"]: p for p in POLICIES}
    clean, dropped = clean_records(raw, set(policy_by_id))

    ans = [r for r in clean if r["answerable"]]
    oos = [r for r in clean if not r["answerable"]]
    tr_ids, va_ids, te_ids = split_policy_ids(sorted({r["policy_id"] for r in ans}), seed)
    oos_tr, oos_va, oos_te = split_list(sorted(oos, key=lambda r: r["id"]), seed)

    def pick(ids):
        return [r for r in ans if r["policy_id"] in ids]

    splits = {"train": pick(tr_ids) + oos_tr, "validation": pick(va_ids) + oos_va, "test": pick(te_ids) + oos_te}

    # Retriever for building contexts. Training-time distractors may come from any policy.
    retriever = BM25Retriever(POLICIES)
    rng = random.Random(seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(POLICY_PATH, POLICIES_AS_ROWS())

    stats = {"raw_records": len(raw), "clean_records": len(clean), "dropped": dict(dropped), "splits": {}}
    for name, recs in splits.items():
        recs = sorted(recs, key=lambda r: r["id"])
        if name == "test":
            rows = [{"id": r["id"], "question": r["question"], "gold_policy_id": r["policy_id"],
                     "gold_answer": r["answer"], "key_facts": r["key_facts"], "answerable": r["answerable"]}
                    for r in recs]
        else:
            rows = [to_chat_example(r, retriever, policy_by_id, rng, cfg["wrong_context_rate"], True) for r in recs]
        write_jsonl(PROCESSED_DIR / f"{name}.jsonl", rows)
        kinds = Counter(r["kind"] if "kind" in r else ("answerable" if r["answerable"] else "out_of_scope") for r in rows)
        stats["splits"][name] = {"n": len(rows), "kinds": dict(kinds)}
    stats["policy_split"] = {"train": sorted(tr_ids), "validation": sorted(va_ids), "test": sorted(te_ids)}
    write_json(PROCESSED_DIR / "dataset_stats.json", stats)
    write_report(stats)
    return stats


def POLICIES_AS_ROWS():
    return [{"id": p["id"], "title": p["title"], "text": p["text"]} for p in POLICIES]


def write_report(stats: dict) -> None:
    lines = [
        "# Dataset report (auto-generated by `src/data/build_dataset.py`)", "",
        "## Source",
        "Synthetic HR policy knowledge base for the fictional company *Nimbus Works* "
        "(`src/data/policy_kb.py`). Authored for this project; no real or personal data.", "",
        "## Cleaning",
        f"- Raw records: **{stats['raw_records']}**",
        f"- Clean records kept: **{stats['clean_records']}**",
        "- Records dropped, by reason:",
    ]
    for k, v in sorted(stats["dropped"].items()):
        lines.append(f"  - `{k}`: {v}")
    lines += ["", "## Splits (grouped by policy - no policy appears in two splits)", "",
              "| Split | Examples | Kinds |", "|---|---|---|"]
    for name, s in stats["splits"].items():
        lines.append(f"| {name} | {s['n']} | {s['kinds']} |")
    lines += ["", "## Policies per split"]
    for name, ids in stats["policy_split"].items():
        lines.append(f"- **{name}** ({len(ids)}): {', '.join(ids)}")
    lines += [
        "", "## Example kinds",
        "- `answerable`: gold policy excerpt in the prompt, grounded JSON answer.",
        "- `wrong_context`: a real question but a *different* policy excerpt; target is a refusal (teaches grounding).",
        "- `out_of_scope`: question not covered by any policy; target is a refusal.",
        "", "## Format",
        "Chat JSONL (`messages` = system / user / assistant). Assistant target is a JSON object "
        '`{"answer", "source", "answerable"}`. The test split stores raw fields only; prompts are built at '
        "evaluation time so the retrieval step can be measured separately.", "",
        "## Known limitations",
        "- Small (a few hundred examples) and synthetic; answers are templated, so wording diversity is limited.",
        "- Fine-tuning here teaches *format, grounding and refusal behaviour*; the facts themselves come from retrieval.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["raw", "process", "all"], default="all")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/data.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("raw", "all"):
        n = write_jsonl(RAW_PATH, make_raw_export(cfg["seed"]))
        print(f"raw export written: {n} records -> {RAW_PATH}")
    if args.stage in ("process", "all"):
        stats = process(cfg)
        print("clean:", stats["clean_records"], "dropped:", stats["dropped"])
        for k, v in stats["splits"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
