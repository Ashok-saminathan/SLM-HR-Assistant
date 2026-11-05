"""Aggregate experiments/eval/*/metrics.json into reports/benchmark_report.md.

Usage:  python -m src.evaluation.report [--config configs/model.yaml]
The report is generated ONLY from measured files - nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import PROJECT_ROOT, load_config

COLUMNS = [
    ("valid_json_rate", "Valid JSON"),
    ("task_success_rate", "Task success"),
    ("task_success_given_retrieval_hit", "Success given retrieval hit"),
    ("correct_refusal_rate_oos", "Correct refusal (OOS)"),
    ("false_refusal_rate_given_hit", "False refusal"),
    ("fact_recall_raw_mean", "Fact recall (raw)"),
    ("token_f1_mean", "Token F1"),
    ("latency_p50_s", "p50 latency (s)"),
    ("latency_p95_s", "p95 latency (s)"),
    ("tokens_per_s_mean", "Tokens/s"),
]


def build_report(eval_dir: Path) -> str:
    runs = []
    for p in sorted(eval_dir.glob("*/metrics.json")):
        runs.append(json.loads(p.read_text()))
    if not runs:
        return "# Benchmark report\n\nNo evaluation runs found in `experiments/eval/`. Run `src/evaluation/evaluate.py` first.\n"
    lines = ["# Benchmark report (auto-generated from `experiments/eval/*/metrics.json`)", ""]
    r0 = runs[0]
    env = r0["environment"]
    lines += [f"- Base model: `{r0['model']['name']}`  |  split: `{r0['split']}`  |  GPU: {env.get('gpu', 'none (CPU)')}",
              f"- transformers {env.get('transformers')}, torch {env.get('torch')}, peft {env.get('peft')}", "",
              "## Quality and speed (same held-out test cases for every run)", "",
              "| Run | Variant | Mode | Batch | " + " | ".join(c[1] for c in COLUMNS) + " |",
              "|---|---|---|---|" + "---|" * len(COLUMNS)]
    for r in runs:
        m = r["metrics"]
        lines.append(f"| {r['name']} | {r['variant']} | {r['mode']} | {r['batch_size']} | "
                     + " | ".join(str(m.get(k)) for k, _ in COLUMNS) + " |")
    lines += ["", "## Resource usage", "", "| Run | Load time (s) | Model footprint (MB) | GPU peak (MB) | CPU RSS (MB) |", "|---|---|---|---|---|"]
    for r in runs:
        res = r["resources"]
        lines.append(f"| {r['name']} | {res['load_time_s']} | {res['model_footprint_mb']} | {res['gpu_peak_mb']} | {res['cpu_rss_mb']} |")
    lines += ["", "## Failure analysis (counts of failed test cases)", ""]
    for r in runs:
        lines.append(f"- **{r['name']}**: {r['failures'] or 'no failures'}")
    lines += ["", "## How to read this", "",
              "- *Task success* is end-to-end (retrieval + model). *Success given retrieval hit* isolates the model.",
              "- Differences between `baseline` and `finetuned` come from fine-tuning only; differences between "
              "`oracle` and `retrieved` mode come from retrieval only; quantized/merged variants measure the optimization trade-off.",
              f"- The test set is small ({r0['metrics']['n_cases']} cases): treat differences of a few points as noise."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/model.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    text = build_report(PROJECT_ROOT / cfg["paths"]["experiments_dir"] / "eval")
    out = PROJECT_ROOT / "reports/benchmark_report.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
