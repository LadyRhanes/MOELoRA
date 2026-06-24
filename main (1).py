# ── modular.py ────────────────────────────────────────────────────────────────

import os
from datasets import Dataset, interleave_datasets
from transformers import AutoTokenizer

from config import TrainingConfig
from config import DATASET_CONFIGS
from config import TOTAL_SAMPLES, EVAL_FRACTION
from loaders.codecontestsloader import CodeContestsLoader
from loaders.codealpacaloader   import CodeAlpacaLoader
from loaders.python_instruct    import PythonInstructionsLoader
from loaders.code_search_net import CodeSearchNetLoader
from loaders.codefeedback import CodeFeedbackLoader
from loaders.leetcode_loader import LeetCodeLoader
from loaders.magic_coder_oss import MagicoderOSSLoader
from loaders.sql import SqlCreateContextLoaderBi


from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("WARNING: HF_TOKEN not set — gated datasets will fail")

# ── tokenizer ─────────────────────────────────────────────────────────────────

conf           = TrainingConfig()
MODEL_NAME     = conf.MODEL_ID
CONTEXT_LENGTH = conf.CONTEXT_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── loaders ───────────────────────────────────────────────────────────────────

LOADER_ARGS = dict(tokenizer=tokenizer, context_length=CONTEXT_LENGTH)

LOADERS = {
    "code_contests":       CodeContestsLoader(**LOADER_ARGS),
    "codealpaca":          CodeAlpacaLoader(**LOADER_ARGS),
    "python_instructions": PythonInstructionsLoader(**LOADER_ARGS),
    "magicoder":           MagicoderOSSLoader(**LOADER_ARGS),
    "code_feedback":       CodeFeedbackLoader(**LOADER_ARGS),
    "leetcode":            LeetCodeLoader(**LOADER_ARGS),
    "code_search_net":     CodeSearchNetLoader(**LOADER_ARGS),
    "sql_context":         SqlCreateContextLoaderBi(**LOADER_ARGS),
}
# ── main ──────────────────────────────────────────────────────────────────────

def make_dataset():
    total_samples = TOTAL_SAMPLES   
    eval_fraction = EVAL_FRACTION
    all_train = []
    all_eval  = []

    for cfg in DATASET_CONFIGS:
        total_alloc = int(TOTAL_SAMPLES * cfg.weight)
        eval_alloc  = max(1, int(total_alloc * EVAL_FRACTION))
        train_alloc = total_alloc - eval_alloc

        print(
            f"\n[{cfg.name}] "
            f"total={total_alloc} | train={train_alloc} | eval={eval_alloc}"
        )

        if cfg.name not in LOADERS:
            raise KeyError(f"Missing loader: {cfg.name}")

        raw = LOADERS[cfg.name].collect(train_alloc + eval_alloc)

        n = len(raw["input_ids"])

        if n == 0:
            raise RuntimeError(f"{cfg.name} produced zero samples")

        # shuffle BEFORE train/eval split
        shuffled = Dataset.from_dict(raw).shuffle(seed=42)

        raw = {
            "input_ids": shuffled["input_ids"],
            "attention_mask": shuffled["attention_mask"],
            "labels": shuffled["labels"],
        }

        actual_eval  = min(eval_alloc, n)
        actual_train = n - actual_eval

        print(
            f"[{cfg.name}] "
            f"collected={n} "
            f"train={actual_train} "
            f"eval={actual_eval}"
        )

        all_train.append(
            Dataset.from_dict(
                {k: v[:actual_train] for k, v in raw.items()}
            )
        )

        all_eval.append(
            Dataset.from_dict(
                {k: v[actual_train:] for k, v in raw.items()}
            )
        )
    target_features = all_train[0].features
    all_train = [ds.cast(target_features) for ds in all_train]
    all_eval  = [ds.cast(target_features) for ds in all_eval]

    weights = [cfg.weight for cfg in DATASET_CONFIGS]

    train_dataset = interleave_datasets(
        all_train,
        probabilities=weights,
        seed=42,
        stopping_strategy="first_exhausted",
    ).shuffle(seed=42)

    eval_dataset = interleave_datasets(
        all_eval,
        probabilities=weights,
        seed=42,
        stopping_strategy="first_exhausted",
    ).shuffle(seed=42)

    print(f"\n=== Dataset Summary ===")
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── validation ────────────────────────────────────────────────────────────
    sample = train_dataset[0]
    assert "input_ids"      in sample
    assert "attention_mask" in sample
    assert "labels"         in sample
    assert len(sample["input_ids"]) == len(sample["labels"]), \
        f"Length mismatch: input_ids={len(sample['input_ids'])} labels={len(sample['labels'])}"
    assert any(x != -100 for x in sample["labels"]), \
        "All labels are -100 — label masking is broken"
    print("Validation passed.")

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/eval",  exist_ok=True)
    train_dataset.save_to_disk("data/train")
    eval_dataset.save_to_disk("data/eval")
    print("Saved to data/train and data/eval")

    return train_dataset, eval_dataset


if __name__ == "__main__":
    make_dataset()