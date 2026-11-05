# Technical report - HR Policy Assistant with a Small Language Model

All numbers below come from one run on a Colab **Tesla T4** (transformers 5.22.0, torch 2.11.0+cu130, peft 0.21.2, bitsandbytes 0.50.2,
Python 3.13), seed 42. Raw evidence: `experiments/` (metrics, per-case predictions, training log); the full tables are in
`reports/benchmark_report.md` (auto-generated from those files).

## 1. Problem and success criteria
Employees ask HR-policy questions; the assistant answers from a policy excerpt (JSON `{answer, source, answerable}`) or refuses.
Targets were fixed before measuring:

| Criterion | Target | Result (`ft_merged`, retrieved) | Met? |
|---|---|---|---|
| Valid JSON | >= 95% | 100% | yes |
| Task success given correct retrieval | >= 85% | 71.4% (10/14) | **no** |
| Correct refusal of out-of-scope questions | >= 80% | 100% (5/5) | yes |
| p95 latency, batch 1, T4 | <= 3 s | 1.67 s | yes |

## 2. Why an SLM and which one
Narrow task, short context, in-house data, latency/cost -> self-hosted SLM. Six candidates were compared in `reports/model_selection.md`;
**Qwen2.5-1.5B-Instruct** was chosen (Apache-2.0, fits LoRA/QLoRA on a T4, good JSON instruction following). Fallback: 0.5B on CPU.

## 3. Data
Synthetic knowledge base of 34 policies / 108 authored QA pairs + 25 out-of-scope questions (`src/data/policy_kb.py`).
A raw export with injected noise was cleaned by explicit rules (`src/data/validate.py`): **156 raw -> 133 clean** (8 duplicates,
3 each of empty answer / missing question / unknown policy id / too-short question / answer missing its key fact).
Split grouped by policy (24 / 5 / 5 policies) so test policies are unseen: train 92, validation 21, test 20 (15 answerable + 5 out-of-scope).
Train/validation examples are of three kinds: answerable (gold excerpt), wrong-context (different excerpt -> refusal) and out-of-scope (-> refusal).

## 4. Pipeline and the three levers
Retrieval (BM25: supplies facts) / prompting (system prompt + JSON contract) / fine-tuning (LoRA: format, grounding, refusal) are separate
modules and are evaluated separately: retrieved vs oracle mode isolates retrieval; baseline vs fine-tuned with the *same* prompt isolates fine-tuning.
BM25 top-1 retrieval found the right policy for 14/15 answerable test questions (hit rate 93%); the miss was "When do I receive my laptop?"
(retrieved `exit-process` instead of `joining-formalities`).

## 5. Baseline (M4)
Unmodified Qwen2.5-1.5B-Instruct, greedy decoding, `max_new_tokens=160`, same prompt, same retrieval.
- On 12 validation prompts: only 4/12 (33%) passed validation; 8/12 failed with `empty_or_non_string_answer`.
- On the 20 test cases: valid-JSON **45%**, task success **40%**, correct refusal of out-of-scope **0/5**.
- Observed limitation: the model *does* output JSON and often knows the right content, but ignores the schema types - it returns
  `{"answer": 12, ...}`, `{"answer": false, ...}` or `{"answer": null, ...}` instead of a sentence, and sometimes wraps output in a
  ```` ```json ```` fence. Lenient fact recall on the raw text is 73%, so the baseline's problem is format and refusal behaviour, not knowledge.
- Note: validation is deliberately strict (answer must be a non-empty string); this strictness is what a downstream client would need.

## 6. Adaptation (M5)
LoRA r=16, alpha=32, dropout 0.05 on all attention + MLP projections; lr 2e-4 cosine, 3 epochs, effective batch 16, loss on answer tokens only.
- Trainable parameters 18.46 M of 1.562 B (**1.18%**); training time **55 s**; peak GPU memory 4.9 GB; fp16 base model (plain LoRA, not QLoRA).
- Mean train loss 0.271; validation loss 0.186 -> 0.104 -> 0.101 over the 3 epochs (`experiments/training/log_history.json`).
- Why LoRA: ~1% trainable parameters, small adapter, trains in about a minute on a free GPU; a full fine-tune is unnecessary for a format/behaviour change.

## 7. Optimization (M6)
Same 20 test cases, retrieved mode, T4:

| Variant | Batch | p50 latency / request | p95 | Tokens/s | Model memory | Task success |
|---|---|---|---|---|---|---|
| base (baseline) | 1 | 0.86 s | 1.59 s | 25.1 | 2944 MB | 40% |
| LoRA attached (unmerged) | 1 | 2.71 s | 3.20 s | 13.0 | 3015 MB | 75% |
| **Merged adapter (fp16)** | 1 | **1.32 s** | 1.67 s | 26.7 | 2944 MB | 75% |
| Merged adapter (fp16) | 4 | 0.36 s | 0.44 s | 96.1 | 2944 MB | 75% |
| Merged + 4-bit NF4 | 1 | 1.90 s | 2.41 s | 18.6 | **1070 MB** | 90% |
| Merged + 4-bit NF4 | 4 | 0.46 s | 0.58 s | 71.4 | 1070 MB | 90% |

Findings (each is measured, `experiments/eval/*/metrics.json`):
- **Merging the adapter** halves latency (2.71 s -> 1.32 s) with *identical* outputs/metrics and no extra memory; the unmerged adapter even misses the 3 s p95 target (3.20 s).
- **Batching (4)** cuts per-request latency from 1.32 s to 0.36 s (~3.7x throughput) with identical quality.
- **4-bit NF4** cuts model memory by **64%** (2944 -> 1070 MB) but is ~43% *slower* per request at batch 1 on a T4 (de-quantisation overhead). Use it when memory, not latency, is the constraint.
- NF4 scored *higher* task success (90% vs 75%) but this is 3 cases out of 20 and should be read as noise, not as "quantization improves quality": it happened to refuse one fewer borderline question.

## 8. Evaluation (M7)
| Run | Valid JSON | Task success | Success given retrieval hit | Correct refusal (OOS) | False refusal | Fact recall | Token F1 |
|---|---|---|---|---|---|---|---|
| Baseline | 45% | 40% | 57.1% | 0% | 0% | 73.3% | 0.356 |
| Fine-tuned (merged, retrieved) | **100%** | **75%** | 71.4% | **100%** | 28.6% | 66.7% | 0.556 |
| Fine-tuned (merged, **oracle** context) | 100% | 80% | 73.3% | 100% | 26.7% | 73.3% | 0.596 |

- Fine-tuning fixed **format** (45% -> 100% valid), **grounding** and **refusal** (0/5 -> 5/5 out-of-scope correctly refused). Task success rose from 40% to 75%.
- Retrieval explains only **1 of the 5** failed cases (75% retrieved vs 80% oracle); the dominant remaining error is the model, not retrieval.
- Fact recall did not improve (73% -> 67%): the fine-tuned model sometimes refuses instead of answering.

**Failure analysis (fine-tuned, 5 failures out of 20):**
- **4 false refusals** - the model said "I could not find this in the excerpt" although the excerpt contained the answer. Three of the four are
  yes/no questions whose answer is *implied* by the excerpt (e.g. "Can I attach casual leave to my annual leave?" - excerpt says it *cannot* be combined;
  "Can I take marriage leave more than once?" - "available once"; "Can I apply for a transfer in my 6th month?" - "after 12 months"); the fourth was
  a list question ("What documents must I submit when joining?"). Likely cause: 30% of training examples are refusals and the tiny training set has few
  implicit yes/no examples, so the model learnt to be over-cautious when the answer is not stated verbatim.
- **1 retrieval miss** ("When do I receive my laptop?" -> wrong excerpt `exit-process`; the model then refused, which is the *correct* behaviour for the context it saw).
- Baseline failures were 11 x invalid output format + 1 retrieval miss (no refusal behaviour at all).

I did **not** retrain to fix these test-set failures (the test set must not be used for tuning). They are reported as the known limitation below.

## 9. Deployment (M8)
FastAPI (`/ask`, `/health`, `/metrics`), Pydantic request validation (422 on bad input), output validation with safe fallback
(`degraded: true`), structured request log, 500 handler that does not leak internals. Unit-tested with a fake engine
(`tests/test_service_and_api.py`, 29 tests pass). **Live demo to capture before submission:** run the API with `SLM_VARIANT=ft_merged` and show
(1) a normal question, (2) `"hi"` -> 422, (3) an out-of-scope question -> refusal; screenshot or paste the responses here: ______

## 10. Limitations and next steps
- Tiny synthetic data and a 20-case test set: one case = 5 points, so differences of a few points are noise (also true for the NF4 result).
- Over-refusal on implicit / yes-no answers (4/14 with correct context). Next: add more yes/no and list-style training examples, lower `wrong_context_rate`
  (0.30 -> 0.15), tune on the **validation** split only, then re-run the identical test evaluation once.
- BM25 top-1 retrieval misses ~7%: add hybrid (BM25 + embeddings), top-2 context, and a "no relevant policy" score threshold.
- No multi-turn, no auth / rate limiting; model revision is not pinned (`configs/model.yaml: revision: null`) - pin a commit hash.
- Next: GGUF / llama.cpp or vLLM serving, load testing, a larger and real (anonymised) FAQ dataset.

## 11. Compliance checklist
- [x] >= 3 SLMs compared (6)  - [x] baseline saved before tuning (`experiments/baseline/`)  - [x] test set never used for tuning (train reads train + validation only)
- [x] hardware, library versions, settings recorded in `experiments/*/metrics.json`, `run_summary.json`  - [ ] pin the model revision hash
- [x] no secrets / private data / weights in the repo (`artifacts/` and checkpoints are git-ignored)  - [x] README reproduces the workflow
