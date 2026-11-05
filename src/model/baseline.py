"""Stage 03 - baseline: run the UNMODIFIED instruction-tuned SLM on representative prompts.

Prompts come from the VALIDATION split (never the test split). Saves raw outputs, the
generation settings, latency and an automatic summary of observed limitations.

Usage:  python -m src.model.baseline [--config configs/model.yaml] [--limit 12]
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.data.policy_kb import POLICIES
from src.model.inference import SLMEngine, peak_memory_mb, reset_peak_memory
from src.model.output_validation import validate_output
from src.utils import PROJECT_ROOT, capture_environment, load_config, read_jsonl, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/model.yaml"))
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()

    from src.optimization.variants import load_variant

    mcfg = load_config(a.config)
    rows = read_jsonl(PROJECT_ROOT / "data/processed/validation.jsonl")[: a.limit]
    reset_peak_memory()
    model, tok, load_s = load_variant("base", mcfg)
    engine = SLMEngine(model, tok, mcfg["generation"])
    engine.generate(rows[0]["messages"][:-1])  # warm-up

    known = {p["id"] for p in POLICIES}
    outputs, errs = [], Counter()
    for r in rows:
        res = engine.generate(r["messages"][:-1])
        parsed = validate_output(res.text, r["context_policy_id"], known)
        for e in parsed.errors:
            errs[e.split(":")[0]] += 1
        outputs.append({"id": r["id"], "kind": r["kind"], "prompt": r["messages"][:-1], "expected": r["messages"][-1]["content"],
                        "output": res.text, "valid": parsed.valid, "errors": parsed.errors,
                        "latency_s": round(res.latency_s, 3), "new_tokens": res.new_tokens})

    lat = [o["latency_s"] for o in outputs]
    summary = {
        "model": mcfg["model"], "generation_settings": mcfg["generation"], "n_prompts": len(outputs),
        "valid_json_rate": round(sum(o["valid"] for o in outputs) / len(outputs), 3),
        "latency_mean_s": round(sum(lat) / len(lat), 3), "load_time_s": round(load_s, 2),
        "error_counts": dict(errs), "resources": peak_memory_mb(),
        "observed_limitations_auto": [f"{k}: {v}/{len(outputs)} prompts" for k, v in errs.most_common()],
        "observed_limitations_manual": "TODO: read experiments/baseline/outputs.jsonl and note qualitative issues "
                                       "(wrong facts, ignoring the excerpt, verbose answers, no refusal ...).",
        "environment": capture_environment(),
    }
    out = PROJECT_ROOT / mcfg["paths"]["experiments_dir"] / "baseline"
    write_jsonl(out / "outputs.jsonl", outputs)
    write_json(out / "summary.json", summary)
    print(summary["valid_json_rate"], "valid JSON;", summary["error_counts"])


if __name__ == "__main__":
    main()
