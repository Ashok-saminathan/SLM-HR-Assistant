"""SLM inference: single / batched generation with latency and memory measurement."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenResult:
    text: str
    latency_s: float       # wall time of the generate() call (for batches: per-batch time / batch size)
    new_tokens: int
    tokens_per_s: float


def peak_memory_mb() -> dict[str, float | None]:
    """Peak GPU memory allocated by torch (if CUDA) and current process RSS."""
    out: dict[str, float | None] = {"gpu_peak_mb": None, "cpu_rss_mb": None}
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    try:
        import psutil

        out["cpu_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:
        pass
    return out


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class SLMEngine:
    def __init__(self, model: Any, tokenizer: Any, generation_cfg: dict[str, Any]):
        self.model = model
        self.tok = tokenizer
        self.gen_cfg = dict(generation_cfg)

    def _device(self):
        return next(self.model.parameters()).device

    def _render(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_batch(self, batch_messages: list[list[dict[str, str]]]) -> list[GenResult]:
        import torch

        prompts = [self._render(m) for m in batch_messages]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self._device())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.gen_cfg.get("max_new_tokens", 160),
                do_sample=self.gen_cfg.get("do_sample", False),
                repetition_penalty=self.gen_cfg.get("repetition_penalty", 1.0),
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        results = []
        for row in gen:
            n_new = int((row != self.tok.pad_token_id).sum().item())
            text = self.tok.decode(row, skip_special_tokens=True).strip()
            per_item = elapsed / len(batch_messages)
            results.append(GenResult(text, per_item, n_new, n_new / per_item if per_item > 0 else 0.0))
        return results

    def generate(self, messages: list[dict[str, str]]) -> GenResult:
        return self.generate_batch([messages])[0]
