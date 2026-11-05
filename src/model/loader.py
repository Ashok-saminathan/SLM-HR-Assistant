"""Model + tokenizer loading (Hugging Face Transformers, optional PEFT adapter / quantization)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _torch():
    import torch
    return torch


def default_dtype():
    torch = _torch()
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_tokenizer(model_name_or_path: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    return tok


def bnb_nf4_config():
    from transformers import BitsAndBytesConfig

    torch = _torch()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model(
    model_name_or_path: str,
    revision: str | None = None,
    quantization: str | None = None,   # None | "nf4" | "dynamic_int8"
    adapter_dir: str | None = None,    # attach a LoRA adapter (kept separate, not merged)
) -> tuple[Any, float]:
    """Load a causal LM. Returns (model, load_seconds)."""
    from transformers import AutoModelForCausalLM

    torch = _torch()
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"revision": revision}
    if quantization == "nf4":
        if not torch.cuda.is_available():
            raise RuntimeError("nf4 quantization (bitsandbytes) needs a CUDA GPU")
        kwargs["quantization_config"] = bnb_nf4_config()
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float32 if quantization == "dynamic_int8" else default_dtype()

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    if quantization == "dynamic_int8":
        # CPU-only post-training dynamic quantization of Linear layers to int8.
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif quantization != "nf4" and torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, time.perf_counter() - t0


def merge_adapter(base_name: str, adapter_dir: str, out_dir: str, revision: str | None = None) -> str:
    """Merge LoRA weights into the base model and save a standalone fp16/fp32 checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": default_dtype()}
    if _torch().cuda.is_available():
        kwargs["device_map"] = {"": 0}  # merge on GPU: fp16 matmuls on CPU are slow
    base = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out)
    load_tokenizer(base_name, revision).save_pretrained(out)
    return str(out)
