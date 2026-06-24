"""
Dataset loading and preprocessing for MALoRA training.

Datasets:
    - Magicoder-OSS-Instruct-75K  (40% of mix)
    - Evol-Instruct-Code-80k      (35% of mix)
    - CodeXGLUE + MdEval debug    (25% of mix)

Key improvements over original:
    - interleave_datasets instead of concat → balanced sampling
    - DEV_MODE subset for fast iteration
    - context 512 (mean seq len is 430 anyway, 4x attention speedup)
    - proper overflow filtering
"""

import os
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


from training_config import TrainingConfig

conf = TrainingConfig()
MODEL_NAME      = conf.MODEL_ID
CONTEXT_LENGTH  = conf.CONTEXT_LENGTH
FILTER_LONG     = True

# ── DEV MODE ─────────────────────────────────────────────────────────────────
# Set True for fast iteration (10K samples, ~30 min run)
# Set False for real training (full dataset, ~25-35 hrs)
DEV_MODE = True
DEV_TRAIN_SIZE = 10_000
DEV_EVAL_SIZE  = 500
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_messages(messages, tokenizer):
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        full_text,
        truncation=not FILTER_LONG,
        max_length=CONTEXT_LENGTH,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    tokenized["overflow_flag"] = len(input_ids) > CONTEXT_LENGTH

    if FILTER_LONG and tokenized["overflow_flag"]:
        input_ids = input_ids[:CONTEXT_LENGTH]
        tokenized["input_ids"] = input_ids
        if "attention_mask" in tokenized:
            tokenized["attention_mask"] = tokenized["attention_mask"][:CONTEXT_LENGTH]

    # loss masking — only train on assistant responses
    assistant_token_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    labels = [-100] * len(input_ids)
    seq_len = len(assistant_token_ids)

    for i in range(len(input_ids) - seq_len + 1):
        if input_ids[i : i + seq_len] == assistant_token_ids:
            labels[i + seq_len :] = input_ids[i + seq_len :]
            break
    else:
        labels = input_ids.copy()

    tokenized["labels"] = labels
    return tokenized


class Loader:
    def __init__(self):
        pass

    def prune_and_filter(self, dataset, processing_fn, remove_cols):
        processed = dataset.map(processing_fn, remove_columns=remove_cols)

        if FILTER_LONG:
            before = len(processed)
            processed = processed.filter(lambda x: not x["overflow_flag"])
            dropped = before - len(processed)
            print(f"    Filtered {dropped} long samples ({dropped/before:.1%})")

        if "overflow_flag" in processed.column_names:
            processed = processed.remove_columns(["overflow_flag"])

        return processed

    def get_magicoder(self, tokenizer):
        print("\nLoading Magicoder...")
        raw   = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
        split = raw["train"].train_test_split(test_size=0.02, seed=42)

        def fmt(example):
            return tokenize_messages([
                {"role": "user",      "content": example["problem"].strip()},
                {"role": "assistant", "content": example["solution"].strip()},
            ], tokenizer)

        cols = split["train"].column_names
        train = self.prune_and_filter(split["train"], fmt, cols)
        eval_ = self.prune_and_filter(split["test"],  fmt, cols)
        print(f"  Magicoder: {len(train)} train | {len(eval_)} eval")
        return train, eval_

    def get_evolinstruct(self, tokenizer):
        print("\nLoading Evol-Instruct...")
        raw   = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1")
        split = raw["train"].train_test_split(test_size=0.02, seed=42)

        def fmt(example):
            return tokenize_messages([
                {"role": "user",      "content": example["instruction"].strip()},
                {"role": "assistant", "content": example["output"].strip()},
            ], tokenizer)

        cols = split["train"].column_names
        train = self.prune_and_filter(split["train"], fmt, cols)
        eval_ = self.prune_and_filter(split["test"],  fmt, cols)
        print(f"  Evol: {len(train)} train | {len(eval_)} eval")
        return train, eval_

    def get_debugging(self, tokenizer):
        print("\nLoading debugging datasets...")
        codexglue = load_dataset("google/code_x_glue_cc_code_refinement", "medium")
        mdeval    = load_dataset("m-a-p/MdEval-Instruct")

        def fmt_cxg(example):
            return tokenize_messages([
                {"role": "user",      "content": f"Fix the bug in this code:\n\n{example['buggy'].strip()}"},
                {"role": "assistant", "content": example["fixed"].strip()},
            ], tokenizer)

        def fmt_md(example):
            return tokenize_messages(example["messages"], tokenizer)

        cxg_cols = codexglue["train"].column_names
        md_cols  = mdeval["train"].column_names

        cxg_train = self.prune_and_filter(codexglue["train"],      fmt_cxg, cxg_cols)
        cxg_eval  = self.prune_and_filter(codexglue["validation"], fmt_cxg, cxg_cols)

        md_split  = mdeval["train"].train_test_split(test_size=0.10, seed=42)
        md_train  = self.prune_and_filter(md_split["train"], fmt_md, md_cols)
        md_eval   = self.prune_and_filter(md_split["test"],  fmt_md, md_cols)

        train = concatenate_datasets([cxg_train, md_train])
        eval_ = concatenate_datasets([cxg_eval,  md_eval])
        print(f"  Debug: {len(train)} train | {len(eval_)} eval")
        return train, eval_


def make_dataset():
    loader = Loader()

    train_magic, eval_magic = loader.get_magicoder(tokenizer)
    train_evol,  eval_evol  = loader.get_evolinstruct(tokenizer)
    train_debug, eval_debug = loader.get_debugging(tokenizer)

    # cast all to same features
    target_features = train_magic.features
    train_evol  = train_evol.cast(target_features)
    train_debug = train_debug.cast(target_features)
    eval_evol   = eval_evol.cast(target_features)
    eval_debug  = eval_debug.cast(target_features)

    # FIXED: interleave instead of concatenate → balanced sampling
    train_dataset = interleave_datasets(
        [train_magic, train_evol, train_debug],
        probabilities=[0.40, 0.35, 0.25],
        seed=42,
        stopping_strategy="first_exhausted",
    )
    eval_dataset = interleave_datasets(
        [eval_magic, eval_evol, eval_debug],
        probabilities=[0.40, 0.35, 0.25],
        seed=42,
        stopping_strategy="first_exhausted",
    )

    # DEV MODE: subset for fast iteration
    if DEV_MODE:
        print(f"\nDEV MODE: subsetting to {DEV_TRAIN_SIZE} train, {DEV_EVAL_SIZE} eval")
        train_dataset = train_dataset.select(range(min(DEV_TRAIN_SIZE, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(DEV_EVAL_SIZE,   len(eval_dataset))))

    print(f"\n=== Dataset Summary ===")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Eval:  {len(eval_dataset)} samples")

    # validate
    sample = train_dataset[0]
    assert "input_ids"      in sample
    assert "attention_mask" in sample
    assert "labels"         in sample
    assert len(sample["input_ids"]) == len(sample["labels"])
    print("Dataset validation passed.")

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/eval",  exist_ok=True)
    train_dataset.save_to_disk("data/train")
    eval_dataset.save_to_disk("data/eval")
    print("Saved datasets to disk.")


if __name__ == "__main__":
    make_dataset()
