from src.evaluation.evaluate import prepare_prompts
from src.model.prompts import SYSTEM_PROMPT, build_messages, build_target
from src.model.train import ChatSFTDataset
from src.utils import PROJECT_ROOT, read_jsonl


class FakeTokenizer:
    """Character-level tokenizer: enough to verify label masking without downloading a model."""
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_labels_mask_prompt_and_keep_answer():
    row = {"messages": build_messages("leave-annual", "text", "How many days?")
           + [{"role": "assistant", "content": build_target("18 days", "leave-annual", True)}]}
    ds = ChatSFTDataset([row], FakeTokenizer(), max_len=10_000)
    item = ds[0]
    n_prompt = sum(1 for x in item["labels"] if x == -100)
    assert n_prompt > 0 and len(item["labels"]) == len(item["input_ids"])
    answer_ids = [x for x in item["labels"] if x != -100]
    assert "".join(chr(i) for i in answer_ids).endswith("<eos>")
    assert "18 days" in "".join(chr(i) for i in answer_ids)


def test_too_long_examples_are_skipped_not_truncated():
    row = {"messages": build_messages("a", "t" * 100, "q?") + [{"role": "assistant", "content": "{}"}]}
    ds = ChatSFTDataset([row], FakeTokenizer(), max_len=50)
    assert len(ds) == 0 and ds.skipped == 1


def test_eval_prompts_modes():
    cases = read_jsonl(PROJECT_ROOT / "data/processed/test.jsonl")
    oracle = prepare_prompts(cases, "oracle")
    for case, (msgs, ctx) in zip(cases, oracle):
        if case["answerable"]:
            assert ctx == case["gold_policy_id"]
        assert msgs[0]["content"] == SYSTEM_PROMPT and case["question"] in msgs[1]["content"]
    retrieved = prepare_prompts(cases, "retrieved")
    assert len(retrieved) == len(cases)
