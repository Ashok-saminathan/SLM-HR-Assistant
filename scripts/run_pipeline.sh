#!/usr/bin/env bash
# End-to-end reproduction of the main workflow (needs a GPU for sensible speed; ~15-30 min on a Colab T4).
# Usage: bash scripts/run_pipeline.sh [configs/model.yaml]
set -euo pipefail
CFG="${1:-configs/model.yaml}"

python -m pytest -q tests                                        # 0. unit tests (no model needed)
python -m src.data.build_dataset                                 # 1. raw -> clean -> split -> chat JSONL + dataset report
python -m src.model.baseline --config "$CFG"                     # 2. baseline outputs on validation prompts
python -m src.evaluation.evaluate --config "$CFG" --variant base --name baseline            # 3. baseline on the TEST set
python -m src.model.train --model-config "$CFG"                  # 4. LoRA fine-tuning (validation split only)
python -m src.evaluation.evaluate --config "$CFG" --variant ft_lora --name finetuned_lora   # 5. fine-tuned on the SAME test set
python -m src.optimization.benchmark --config "$CFG" --variants ft_merged ft_merged_nf4 --batch-sizes 1 4   # 6. optimization
python -m src.evaluation.evaluate --config "$CFG" --variant ft_merged --name finetuned_oracle --mode oracle  # 7. isolate retrieval vs model
python -m src.evaluation.report --config "$CFG"                  # 8. reports/benchmark_report.md from measured files
echo "Done. Start the API with: SLM_VARIANT=ft_merged uvicorn src.api.main:app --port 8000"
