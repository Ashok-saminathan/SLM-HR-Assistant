"""LoRA / QLoRA fine-tuning with PEFT + the Hugging Face Trainer.

Only the train and validation splits are read. The test split is never touched here.
Loss is computed on the assistant answer tokens only (prompt tokens are masked with -100).

Usage:
    python -m src.model.train --model-config configs/model.yaml --train-config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.model.loader import bnb_nf4_config, default_dtype, load_tokenizer
from src.utils import PROJECT_ROOT, capture_environment, load_config, read_jsonl, set_seed, write_json


class ChatSFTDataset:
    """Tokenises chat examples; labels = -100 for the prompt, token ids for the assistant reply."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.items = []
        skipped = 0
        for r in rows:
            msgs = r["messages"]
            prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            target = msgs[-1]["content"] + tokenizer.eos_token
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            ids = p_ids + t_ids
            if len(ids) > max_len:  # never truncate the answer silently - skip and count
                skipped += 1
                continue
            self.items.append({"input_ids": ids, "labels": [-100] * len(p_ids) + t_ids})
        self.skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default=str(PROJECT_ROOT / "configs/model.yaml"))
    ap.add_argument("--train-config", default=str(PROJECT_ROOT / "configs/train.yaml"))
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data/processed"))
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    mcfg, tcfg = load_config(args.model_config), load_config(args.train_config)
    set_seed(tcfg["seed"])
    name, rev = mcfg["model"]["name"], mcfg["model"]["revision"]
    t = tcfg["training"]
    use_cuda = torch.cuda.is_available()
    qlora = tcfg["method"] == "qlora"
    if qlora and not use_cuda:
        raise SystemExit("QLoRA needs a CUDA GPU + bitsandbytes. Use method: lora on CPU.")

    tok = load_tokenizer(name, rev)
    tok.padding_side = "right"  # training pads on the right
    kwargs = {"revision": rev}
    if qlora:
        kwargs.update(quantization_config=bnb_nf4_config(), device_map={"": 0})
    else:
        kwargs["torch_dtype"] = default_dtype()
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.config.use_cache = False
    if qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t["gradient_checkpointing"])
    elif t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lc = tcfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], task_type="CAUSAL_LM", bias="none"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_ds = ChatSFTDataset(read_jsonl(Path(args.data_dir) / "train.jsonl"), tok, t["max_seq_len"])
    val_ds = ChatSFTDataset(read_jsonl(Path(args.data_dir) / "validation.jsonl"), tok, t["max_seq_len"])
    print(f"train={len(train_ds)} val={len(val_ds)} (skipped too long: {train_ds.skipped}/{val_ds.skipped})")

    adapter_dir = PROJECT_ROOT / mcfg["paths"]["adapter_dir"]
    exp_dir = PROJECT_ROOT / mcfg["paths"]["experiments_dir"] / "training"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # `warmup_ratio` was removed in newer transformers; an explicit step count works in every version.
    steps_per_epoch = max(1, -(-len(train_ds) // (t["per_device_batch_size"] * t["grad_accum_steps"])))
    warmup_steps = max(1, int(t["warmup_ratio"] * steps_per_epoch * t["epochs"]))

    args_tr = TrainingArguments(
        output_dir=str(exp_dir / "checkpoints"),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_steps=warmup_steps,
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_cuda,
        optim="paged_adamw_8bit" if qlora else "adamw_torch",
        report_to="none",
        seed=tcfg["seed"],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - t0
    final_eval = trainer.evaluate()

    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    write_json(exp_dir / "log_history.json", trainer.state.log_history)
    write_json(exp_dir / "run_summary.json", {
        "base_model": name, "revision": rev, "method": tcfg["method"],
        "lora": lc, "training": t, "seed": tcfg["seed"],
        "trainable_params": trainable, "total_params": total,
        "train_examples": len(train_ds), "val_examples": len(val_ds),
        "train_runtime_s": round(wall, 1), "train_loss": result.training_loss,
        "final_eval": final_eval,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else None,
        "adapter_dir": str(adapter_dir),
        "environment": capture_environment(),
    })
    print("adapter saved to", adapter_dir)
    print(json.dumps(final_eval, indent=2))


if __name__ == "__main__":
    main()
