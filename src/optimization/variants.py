"""Model variants used for the optimization experiment.

Each variant is a way of serving the same task:
  base                 baseline instruction-tuned SLM, no adaptation
  ft_lora              base + LoRA adapter attached (extra adapter matmuls at inference)
  ft_merged            adapter merged into base weights (no adapter overhead)
  ft_merged_nf4        merged model quantized to 4-bit NF4 with bitsandbytes (CUDA only)
  ft_merged_int8_cpu   merged model with PyTorch dynamic int8 quantization (CPU only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.model.loader import load_model, load_tokenizer, merge_adapter
from src.utils import PROJECT_ROOT

VARIANTS: dict[str, dict[str, Any]] = {
    "base": {"adapter": False, "merge": False, "quantization": None},
    "base_nf4": {"adapter": False, "merge": False, "quantization": "nf4"},
    "ft_lora": {"adapter": True, "merge": False, "quantization": None},
    "ft_merged": {"adapter": True, "merge": True, "quantization": None},
    "ft_merged_nf4": {"adapter": True, "merge": True, "quantization": "nf4"},
    "ft_merged_int8_cpu": {"adapter": True, "merge": True, "quantization": "dynamic_int8"},
}


def load_variant(name: str, mcfg: dict[str, Any]):
    """Return (model, tokenizer, load_seconds) for a named variant."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    spec = VARIANTS[name]
    base, rev = mcfg["model"]["name"], mcfg["model"]["revision"]
    adapter_dir = str(PROJECT_ROOT / mcfg["paths"]["adapter_dir"])
    merged_dir = PROJECT_ROOT / mcfg["paths"]["merged_dir"]

    if spec["adapter"] and not Path(adapter_dir).exists():
        raise FileNotFoundError(f"adapter not found at {adapter_dir}. Run training first (src/model/train.py).")

    if spec["merge"]:
        if not (merged_dir / "config.json").exists():
            merge_adapter(base, adapter_dir, str(merged_dir), rev)
        model, secs = load_model(str(merged_dir), None, spec["quantization"])
        tok = load_tokenizer(str(merged_dir))
    else:
        model, secs = load_model(base, rev, spec["quantization"], adapter_dir if spec["adapter"] else None)
        tok = load_tokenizer(base, rev)
    return model, tok, secs
