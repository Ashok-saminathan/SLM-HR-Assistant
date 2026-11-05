# Model selection (Stage 01)

**Task:** answer employee HR-policy questions from a short policy excerpt, replying with a small JSON object.
**Why an SLM:** the task is narrow, the context is short (one policy excerpt, ~100 tokens), HR text should not leave the
company network, and requests are frequent - so a small, self-hosted model with low latency and no per-token API cost is a better fit
than a hosted large model. Fine-tuning a 1-2B model with LoRA is feasible on a single free-tier GPU.

## Candidates

Figures are from the public model cards as I know them; **re-check size / licence / context on the model card on the day you run the
experiment and fix the exact revision hash in `configs/model.yaml`.**

| Model | Params | Context | Licence | fp16 weights (approx.) | Fits QLoRA on 16 GB T4? | Notes |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | ~1.5B | 32K | Apache-2.0 | ~3 GB | Yes (and plain LoRA) | Strong instruction following + JSON output for its size; permissive licence |
| Qwen2.5-0.5B-Instruct | ~0.5B | 32K | Apache-2.0 | ~1 GB | Yes, also CPU-trainable | Fallback for CPU-only demo (`configs/model_0.5b.yaml`); weaker reasoning |
| Llama-3.2-1B-Instruct | ~1.2B | 128K | Llama 3.2 Community Licence (gated, accept terms; attribution / usage conditions) | ~2.5 GB | Yes | Good quality; gated download and custom licence add friction |
| SmolLM2-1.7B-Instruct | ~1.7B | 8K | Apache-2.0 | ~3.4 GB | Yes | Open and permissive; shorter context, weaker multilingual |
| Gemma-2-2B-it | ~2.6B | 8K | Gemma Terms of Use (gated) | ~5 GB | Yes (tighter) | Good quality, larger; custom terms |
| Phi-3.5-mini-instruct | ~3.8B | 128K | MIT | ~7.6 GB | Only with QLoRA, little headroom | Highest capability here but slowest; heavier for deployment |

## Decision: Qwen2.5-1.5B-Instruct

1. **Licence** - Apache-2.0, ungated: reviewers can reproduce without accepting terms or tokens.
2. **Size vs. hardware** - ~3 GB in fp16: LoRA and QLoRA both fit a free Colab T4; the 4-bit version is ~1 GB for cheap serving.
3. **Capability** - instruction-tuned, good at structured (JSON) output, which the validation layer depends on.
4. **Context** - 32K is far more than the ~300-token prompts used here (room to add more retrieved excerpts later).
5. **Fallback** - same family at 0.5B shares the tokenizer / chat template, so the whole pipeline runs on CPU by changing one config file.

Trade-offs accepted: not the most capable candidate (Phi-3.5-mini is larger and stronger); chosen for the best capability-per-GB with a permissive licence.
The choice is a hypothesis: the baseline and fine-tuned evaluations (`reports/benchmark_report.md`) show whether 1.5B is enough.
