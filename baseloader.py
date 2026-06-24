from datasets import load_dataset
from collections import defaultdict


class BaseDatasetLoader:
    """
    Base class for all dataset loaders.
    Subclasses must implement:
      - HF_ID        : str
      - SUBSET       : str | None
      - SPLIT        : str
      - _format(example) -> list[dict] | None
    """

    HF_ID  = None
    SUBSET = None
    SPLIT  = "train"

    def __init__(self, tokenizer, context_length, seed=42):
        self.tokenizer          = tokenizer
        self.context_length     = context_length
        self.seed               = seed
        self._assistant_tok_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def collect(self, n_samples):
        """
        Stream, format, tokenize and return exactly n_samples valid examples.
        Returns {input_ids, attention_mask, labels}.
        Supports _format() returning:
        - list[dict]         — single message list (one sample)
        - list[list[dict]]   — multiple message lists (multi-sample per raw example)
        - None               — skip example
        """
        print(f"  [{self.__class__.__name__}] Streaming — need {n_samples} samples...")

        ds = (
            load_dataset(
                self.HF_ID, self.SUBSET,
                split=self.SPLIT, streaming=True, trust_remote_code=True,
            )
            if self.SUBSET else
            load_dataset(
                self.HF_ID,
                split=self.SPLIT, streaming=True, trust_remote_code=True,
            )
        )

        collected = {"input_ids": [], "attention_mask": [], "labels": []}

        # ── per-reason skip counters ──────────────────────────────────
        raw_streamed      = 0
        format_rejected   = 0
        context_overflow  = 0
        all_labels_masked = 0
        # ─────────────────────────────────────────────────────────────

        for example in ds:
            if len(collected["input_ids"]) >= n_samples:
                break

            raw_streamed += 1

            # 1. format
            raw = self._format(example)
            if raw is None:
                format_rejected += 1
                continue

            # ── multi-sample detection ────────────────────────────────
            # if first element is itself a list, treat as multiple samples
            samples = raw if isinstance(raw[0], list) else [raw]
            # ─────────────────────────────────────────────────────────

            for messages in samples:
                if len(collected["input_ids"]) >= n_samples:
                    break

                # 2. tokenize + mask
                tokenized = self._tokenize_and_mask(messages)

                # 3. overflow check
                if tokenized["overflow_flag"]:
                    context_overflow += 1
                    continue

                # 4. label sanity check
                if all(l == -100 for l in tokenized["labels"]):
                    all_labels_masked += 1
                    continue

                collected["input_ids"].append(tokenized["input_ids"])
                collected["attention_mask"].append(tokenized["attention_mask"])
                collected["labels"].append(tokenized["labels"])

        # ── diagnostics ───────────────────────────────────────────────
        samples_collected = len(collected["input_ids"])
        total_skipped     = format_rejected + context_overflow + all_labels_masked
        overflow_rate     = (context_overflow / raw_streamed * 100) if raw_streamed else 0.0

        print(f"\n  [{self.__class__.__name__}] diagnostics")
        print(f"    raw_streamed      : {raw_streamed}")
        print(f"    samples_collected : {samples_collected}")
        print(f"    total_skipped     : {total_skipped}")
        print(f"      format_rejected  : {format_rejected}")
        print(f"      context_overflow : {context_overflow}")
        print(f"      all_labels_masked: {all_labels_masked}")
        print(f"    overflow_rate     : {overflow_rate:.2f}%")

        if samples_collected < n_samples:
            print(
                f"\n  [{self.__class__.__name__}] WARNING: "
                f"exhausted before {n_samples} — got {samples_collected}"
            )
        # ─────────────────────────────────────────────────────────────

        return collected

    def _format(self, example):
        raise NotImplementedError

    def _tokenize_and_mask(self, messages):
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        tokenized = self.tokenizer(
            full_text, truncation=True, max_length=self.context_length, padding=False,
        )

        input_ids = tokenized["input_ids"]
        tokenized["overflow_flag"] = len(input_ids) >= self.context_length

        labels = [-100] * len(input_ids)
        seq_len = len(self._assistant_tok_ids)
        for i in range(len(input_ids) - seq_len + 1):
            if input_ids[i : i + seq_len] == self._assistant_tok_ids:
                labels[i + seq_len:] = input_ids[i + seq_len:]
                break

        # Clamp out-of-range label token IDs to -100.
        # Use len(tokenizer), NOT tokenizer.vocab_size -- Qwen2.5 special tokens
        # (e.g. eos_token_id=151645) sit ABOVE vocab_size (151643) but are still
        # valid, legitimate tokens the model needs to predict (e.g. to learn
        # when to stop generating). Using vocab_size as the bound incorrectly
        # treats EOS and other special tokens as invalid and masks them out.
        vocab_size = len(self.tokenizer)
        labels = [l if l == -100 or 0 <= l < vocab_size else -100 for l in labels]

        tokenized["labels"] = labels
        return tokenized