"""Run the held-out test set through one model variant and write metrics + predictions.

The SAME test cases and the SAME retrieval/prompt code are used for every variant, so
baseline and fine-tuned numbers are directly comparable.

Usage:
    python -m src.evaluation.evaluate --variant base      --name baseline
    python -m src.evaluation.evaluate --variant ft_merged --name finetuned
    python -m src.evaluation.evaluate --variant ft_merged_nf4 --name finetuned_nf4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.policy_kb import POLICIES
from src.evaluation.metrics import aggregate, failure_breakdown, score_example
from src.model.inference import SLMEngine, peak_memory_mb, reset_peak_memory
from src.model.output_validation import validate_output
from src.model.prompts import build_messages
from src.retrieval.retriever import BM25Retriever
from src.utils import PROJECT_ROOT, capture_environment, load_config, read_jsonl, write_json, write_jsonl


def prepare_prompts(cases: list[dict], mode: str, top_k: int = 1):
    """Build (messages, context_policy_id) for each case. mode: retrieved | oracle."""
    retriever = BM25Retriever(POLICIES)
    by_id = {p["id"]: p for p in POLICIES}
    out = []
    for c in cases:
        if mode == "oracle" and c["answerable"]:
            pid, text = c["gold_policy_id"], by_id[c["gold_policy_id"]]["text"]
        else:
            hit = retriever.search(c["question"], top_k=top_k)[0]
            pid, text = hit.policy_id, hit.text
        out.append((build_messages(pid, text, c["question"]), pid))
    return out


def run_evaluation(variant: str, mcfg: dict[str, Any], name: str, split: str = "test",
                   mode: str | None = None, batch_size: int | None = None) -> dict[str, Any]:
    from src.optimization.variants import load_variant  # imported lazily: needs torch

    mode = mode or mcfg["eval"]["mode"]
    batch_size = batch_size or mcfg["eval"]["batch_size"]
    cases = read_jsonl(PROJECT_ROOT / f"data/processed/{split}.jsonl")
    prompts = prepare_prompts(cases, mode, mcfg["retrieval"]["top_k"])

    reset_peak_memory()
    model, tok, load_s = load_variant(variant, mcfg)
    engine = SLMEngine(model, tok, mcfg["generation"])
    engine.generate(prompts[0][0])  # warm-up (CUDA kernels / caches), excluded from timing
    reset_peak_memory()

    known = {p["id"] for p in POLICIES}
    rows, preds = [], []
    for i in range(0, len(cases), batch_size):
        chunk = list(zip(cases[i:i + batch_size], prompts[i:i + batch_size]))
        results = engine.generate_batch([p[0] for _, p in chunk])
        for (case, (_, ctx_id)), res in zip(chunk, results):
            parsed = validate_output(res.text, context_policy_id=ctx_id, known_ids=known)
            s = score_example(case, parsed, res.text, ctx_id)
            s.update(answerable_gold=bool(case["answerable"]), pred_answerable=parsed.answerable,
                     latency_s=res.latency_s, new_tokens=res.new_tokens, tokens_per_s=res.tokens_per_s)
            rows.append(s)
            preds.append({"id": case["id"], "question": case["question"], "context_policy_id": ctx_id,
                          "raw_output": res.text, "parsed": parsed.as_dict(), "score": s})

    try:
        footprint = round(model.get_memory_footprint() / 2**20, 1)
    except Exception:
        footprint = None
    metrics = {
        "name": name, "variant": variant, "split": split, "mode": mode, "batch_size": batch_size,
        "model": mcfg["model"], "generation": mcfg["generation"],
        "metrics": aggregate(rows), "failures": failure_breakdown(rows),
        "resources": {"load_time_s": round(load_s, 2), "model_footprint_mb": footprint, **peak_memory_mb()},
        "environment": capture_environment(),
    }
    out_dir = PROJECT_ROOT / mcfg["paths"]["experiments_dir"] / "eval" / name
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", preds)
    return metrics


def main() -> None:
    from src.optimization.variants import VARIANTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/model.yaml"))
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--name", required=True, help="output folder name under experiments/eval/")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--mode", choices=["retrieved", "oracle"])
    ap.add_argument("--batch-size", type=int)
    a = ap.parse_args()
    m = run_evaluation(a.variant, load_config(a.config), a.name, a.split, a.mode, a.batch_size)
    for k, v in m["metrics"].items():
        print(f"{k:38s} {v}")
    print("failures:", m["failures"])
    print("resources:", m["resources"])


if __name__ == "__main__":
    main()
