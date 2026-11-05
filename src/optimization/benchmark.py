"""Stage 05 - optimization experiment: run several serving variants on the SAME test cases.

Measures speed, memory and output quality for every variant, so the trade-off is explicit.

Usage:
    python -m src.optimization.benchmark --variants ft_lora ft_merged ft_merged_nf4 --batch-sizes 1 4
Outputs go to experiments/eval/<variant>_bs<N>/ ; then run `python -m src.evaluation.report`.
"""
from __future__ import annotations

import argparse
import gc

from src.evaluation.evaluate import run_evaluation
from src.optimization.variants import VARIANTS
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/model.yaml"))
    ap.add_argument("--variants", nargs="+", default=["ft_lora", "ft_merged", "ft_merged_nf4"], choices=sorted(VARIANTS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    a = ap.parse_args()
    mcfg = load_config(a.config)
    for v in a.variants:
        for bs in a.batch_sizes:
            name = f"{v}_bs{bs}"
            print(f"=== {name}")
            try:
                m = run_evaluation(v, mcfg, name, batch_size=bs)
                print({k: m["metrics"][k] for k in ("valid_json_rate", "task_success_rate", "latency_p50_s")}, m["resources"])
            except Exception as e:  # e.g. nf4 on a CPU-only machine: report and continue
                print(f"SKIPPED {name}: {type(e).__name__}: {e}")
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
