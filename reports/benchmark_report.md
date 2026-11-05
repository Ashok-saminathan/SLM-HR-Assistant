# Benchmark report (auto-generated from `experiments/eval/*/metrics.json`)

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`  |  split: `test`  |  GPU: Tesla T4
- transformers 5.22.0, torch 2.11.0+cu130, peft 0.21.2

## Quality and speed (same held-out test cases for every run)

| Run | Variant | Mode | Batch | Valid JSON | Task success | Success given retrieval hit | Correct refusal (OOS) | False refusal | Fact recall (raw) | Token F1 | p50 latency (s) | p95 latency (s) | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | base | retrieved | 1 | 0.45 | 0.4 | 0.5714 | 0.0 | 0.0 | 0.7333 | 0.3557 | 0.8564 | 1.5856 | 25.0788 |
| finetuned_lora | ft_lora | retrieved | 1 | 1.0 | 0.75 | 0.7143 | 1.0 | 0.2857 | 0.6667 | 0.556 | 2.7057 | 3.2037 | 12.9841 |
| finetuned_oracle | ft_merged | oracle | 1 | 1.0 | 0.8 | 0.7333 | 1.0 | 0.2667 | 0.7333 | 0.596 | 1.3017 | 1.5417 | 26.9213 |
| ft_merged_bs1 | ft_merged | retrieved | 1 | 1.0 | 0.75 | 0.7143 | 1.0 | 0.2857 | 0.6667 | 0.556 | 1.3233 | 1.6662 | 26.6748 |
| ft_merged_bs4 | ft_merged | retrieved | 4 | 1.0 | 0.75 | 0.7143 | 1.0 | 0.2857 | 0.6667 | 0.556 | 0.3552 | 0.436 | 96.12 |
| ft_merged_nf4_bs1 | ft_merged_nf4 | retrieved | 1 | 1.0 | 0.9 | 0.9286 | 1.0 | 0.0714 | 0.8667 | 0.5919 | 1.8957 | 2.4096 | 18.6064 |
| ft_merged_nf4_bs4 | ft_merged_nf4 | retrieved | 4 | 1.0 | 0.9 | 0.9286 | 1.0 | 0.0714 | 0.8667 | 0.5919 | 0.4648 | 0.5808 | 71.3834 |

## Resource usage

| Run | Load time (s) | Model footprint (MB) | GPU peak (MB) | CPU RSS (MB) |
|---|---|---|---|---|
| baseline | 6.5 | 2944.4 | 2969.4 | 1547.1 |
| finetuned_lora | 8.0 | 3014.8 | 3049.9 | 1627.0 |
| finetuned_oracle | 3.49 | 2944.4 | 2969.4 | 1516.1 |
| ft_merged_bs1 | 10.06 | 2944.4 | 2968.5 | 1617.6 |
| ft_merged_bs4 | 3.39 | 2944.4 | 3014.7 | 1678.5 |
| ft_merged_nf4_bs1 | 1.31 | 1070.2 | 1149.3 | 1740.0 |
| ft_merged_nf4_bs4 | 1.31 | 1070.2 | 1186.6 | 1744.9 |

## Failure analysis (counts of failed test cases)

- **baseline**: {'invalid_output_format': 11, 'retrieval_miss': 1}
- **finetuned_lora**: {'false_refusal': 4, 'retrieval_miss': 1}
- **finetuned_oracle**: {'false_refusal': 4}
- **ft_merged_bs1**: {'false_refusal': 4, 'retrieval_miss': 1}
- **ft_merged_bs4**: {'false_refusal': 4, 'retrieval_miss': 1}
- **ft_merged_nf4_bs1**: {'false_refusal': 1, 'retrieval_miss': 1}
- **ft_merged_nf4_bs4**: {'false_refusal': 1, 'retrieval_miss': 1}

## How to read this

- *Task success* is end-to-end (retrieval + model). *Success given retrieval hit* isolates the model.
- Differences between `baseline` and `finetuned` come from fine-tuning only; differences between `oracle` and `retrieved` mode come from retrieval only; quantized/merged variants measure the optimization trade-off.
- The test set is small (20 cases): treat differences of a few points as noise.
