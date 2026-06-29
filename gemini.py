
import os
import signal
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("WANDB_MODE", "online")

from dotenv import load_dotenv
load_dotenv()

import math
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from accelerate import Accelerator

from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel
from training_config import TrainingConfig
from main import make_dataset

# ── Config ────────────────────────────────────────────────────────────────────
conf = TrainingConfig()

MODEL_ID          = conf.MODEL_ID
OUTPUT_DIR        = conf.OUTPUT_DIR
LOGDIR            = conf.LOGDIR
RUN_NAME          = conf.RUN_NAME
PROJECT_NAME      = conf.PROJECT_NAME
CONTEXT_LENGTH    = conf.CONTEXT_LENGTH
TRAIN_BATCH       = conf.TRAIN_BATCH
EVAL_BATCH        = conf.EVAL_BATCH
GRAD_ACCUM        = conf.GRAD_ACCUM
LR                = conf.LR
NUM_EPOCHS        = conf.NUM_EPOCHS
MAX_STEPS         = conf.MAX_STEPS
EVAL_STEPS        = conf.EVAL_STEPS
LOGGING_STEPS     = conf.LOGGING_STEPS
NUM_CHECKPOINT_LIMIT = conf.NUM_CHECKPOINT_LIMIT
SEED              = conf.SEED
NUM_EXPERTS       = conf.NUM_EXPERTS
NUM_EXPERTS_PER_TOK = conf.NUM_EXPERTS_PER_TOK
EXPERTS_RANK      = conf.EXPERTS_RANK
ATTENTION_RANK    = conf.ATTENTION_RANK
EXPERTS_SCALE     = conf.EXPERTS_SCALE
ROUTER_AUX_COEF   = conf.ROUTER_AUX_COEF
QUANTIZE          = conf.QUANTIZE
USE_8BIT_ADAM     = conf.USE_8BIT_ADAM
RESUME_FROM       = conf.RESUME_FROM


# ── QLoRA config ──────────────────────────────────────────────────────────────
def get_qlora_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── Auto-save on crash/interrupt ──────────────────────────────────────────────
_trainer_ref = None

def _emergency_save(signum, frame):
    """Save checkpoint if training is interrupted (SIGTERM, SIGINT, SIGKILL)."""
    global _trainer_ref
    print(f"\n[SIGNAL {signum}] Caught interrupt — saving emergency checkpoint...")
    if _trainer_ref is not None:
        try:
            emergency_dir = os.path.join(OUTPUT_DIR, "emergency_checkpoint")
            _trainer_ref.save_model(emergency_dir)
            print(f"Emergency checkpoint saved to {emergency_dir}")
        except Exception as e:
            print(f"Emergency save failed: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, _emergency_save)
signal.signal(signal.SIGINT,  _emergency_save)


# ── Custom MoE Trainer ────────────────────────────────────────────────────────
class MoETrainer(Trainer):
    def __init__(self, tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer=tokenizer
        self._eval_aux_losses: list[float] = []
        self._routing_entropy: float = 0.0
        self._expert_usage: list[float] = []
        self._current_train_aux_loss: float = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── SAFETY: Clamp labels to model's actual vocab size ──────────────────
        labels = inputs.get("labels")
        if labels is not None:
            vocab_size = model.base_model.lm_head.weight.shape[0]   # e.g., 151936 for this model
            # Keep -100 as ignore; clamp any other out-of-range value to -100
            invalid_mask = (labels >= vocab_size) | (labels < 0)
            labels = torch.where(
                invalid_mask,
                torch.tensor(-100, dtype=labels.dtype, device=labels.device),
                labels
            )
            inputs["labels"] = labels
        # ─────────────────────────────────────────────────────────────────────────

        inputs["output_router_logits"] = True

       
        outputs = model(**inputs)

        loss = outputs.loss

        # ── aux loss tracking ─────────────────────────────────────────────────
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            aux_val = outputs.aux_loss.detach().item()
            if model.training:
                self._current_train_aux_loss = aux_val
            else:
                self._eval_aux_losses.append(aux_val)

        # ── expert routing telemetry ──────────────────────────────────────────
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None and model.training:
            all_entropies  = []
            all_usage      = []

            for layer_logits in outputs.router_logits:
                # layer_logits: [batch*seq, num_experts]
                probs = torch.softmax(layer_logits.float(), dim=-1)

                # routing entropy — high = balanced, low = collapse
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                all_entropies.append(entropy.item())

                # per-expert usage fraction
                usage = probs.mean(dim=0)
                all_usage.append(usage.detach().cpu())

            self._routing_entropy = sum(all_entropies) / len(all_entropies)
            self._expert_usage = torch.stack(all_usage).mean(dim=0).tolist()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time=None) -> None:
        # training metrics
        if hasattr(self, "_current_train_aux_loss"):
            logs["train_aux_loss"] = self._current_train_aux_loss

        if self._routing_entropy:
            logs["routing_entropy"] = round(self._routing_entropy, 4)

        if self._expert_usage:
            for i, usage in enumerate(self._expert_usage):
                logs[f"expert_{i}_pct"] = round(usage * 100, 2)

        # train perplexity
        if "loss" in logs:
            try:
                logs["train_perplexity"] = math.exp(logs["loss"])
            except OverflowError:
                logs["train_perplexity"] = float("inf")

        super().log(logs, start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_aux_losses.clear()
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self._eval_aux_losses:
            metrics[f"{metric_key_prefix}_aux_loss"] = (
                sum(self._eval_aux_losses) / len(self._eval_aux_losses)
            )

        if f"{metric_key_prefix}_loss" in metrics:
            try:
                metrics[f"{metric_key_prefix}_perplexity"] = math.exp(
                    metrics[f"{metric_key_prefix}_loss"]
                )
            except OverflowError:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        return metrics

    def _remove_unused_columns(self, dataset, description=None):
        # keep all columns — our custom collator handles filtering
        return dataset


# ── WandB Callback ────────────────────────────────────────────────────────────
class WandbConfigCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "model_id":       MODEL_ID,
                    "lr":             LR,
                    "num_epochs":     NUM_EPOCHS,
                    "num_experts":    NUM_EXPERTS,
                    "experts_rank":   EXPERTS_RANK,
                    "attention_rank": ATTENTION_RANK,
                    "context_length": CONTEXT_LENGTH,
                    "train_batch":    TRAIN_BATCH,
                    "grad_accum":     GRAD_ACCUM,
                }, allow_val_change=True)
        except ImportError:
            pass


# ── Checkpoint resume helper ──────────────────────────────────────────────────
def resolve_checkpoint(resume_from: str | None, accelerator: Accelerator) -> str | None:
    if not resume_from:
        return None

    resume_from = os.path.normpath(os.path.expanduser(resume_from))

    if "checkpoint-" in os.path.basename(resume_from):
        pass  # direct checkpoint path
    else:
        checkpoints = sorted(
            [x for x in os.listdir(resume_from) if x.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {resume_from}")
        resume_from = os.path.join(resume_from, checkpoints[-1])

    if accelerator.is_main_process:
        print(f"Resuming from: {resume_from}")

    return resume_from


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _trainer_ref

    accelerator = Accelerator()

    resume_from = resolve_checkpoint(RESUME_FROM, accelerator)

    # WandB setup
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_ENTITY"]  = "godofwar_1007-indian-institute-of-technology-indore"

    if accelerator.is_main_process:
        import wandb
        wandb.init(project=PROJECT_NAME, name=RUN_NAME, resume="allow", entity="lady-rhanes")

    set_seed(SEED)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Use eos_token_id as pad_token_id (standard practice for causal LMs
    # without a dedicated pad token). eos_token_id (151645) is a valid token
    # within len(tokenizer) (151665) -- it only exceeds the narrower
    # tokenizer.vocab_size (151643), which excludes special tokens but is
    # NOT the correct bound to validate against. The earlier "fix" avoiding
    # EOS as pad was based on this same vocab_size/len(tokenizer) confusion.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side    = "right"   # required for flash attention
    tokenizer.truncation_side = "left"

    # ── Datasets ──────────────────────────────────────────────────────────────
    with accelerator.main_process_first():
        if os.path.exists("data/train") and os.path.exists("data/eval"):
            if accelerator.is_main_process:
                print("Loading datasets from disk...")
            train_dataset = load_from_disk("data/train")
            eval_dataset  = load_from_disk("data/eval")
        else:
            if accelerator.is_main_process:
                print("Datasets not found — running multidata.py...")
            make_dataset()
            train_dataset = load_from_disk("data/train")
            eval_dataset  = load_from_disk("data/eval")

    # keep only tensor columns
    tensor_cols   = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.select_columns([c for c in tensor_cols if c in train_dataset.column_names])
    eval_dataset  = eval_dataset.select_columns([c for c in tensor_cols  if c in eval_dataset.column_names])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,   # FIX: pad labels with -100 not eos_token_id
    )

    # ── Base model ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading base model...")
        print(f"  QUANTIZE = {QUANTIZE}  ({'4-bit QLoRA' if QUANTIZE else 'full bf16 (Option 1: no quantization)'})")

    bnb_config = get_qlora_bnb_config() if QUANTIZE else None

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index} if QUANTIZE else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # ENABLED: major speedup on L4
    )

    # OPTION 1: when not quantizing, bnb's device_map isn't used, so the model
    # loads on CPU by default — explicitly move it to GPU ourselves.
    if not QUANTIZE:
        base_model = base_model.to(accelerator.device)

    # input_require_grads is needed for LoRA gradient flow through a frozen
    # base model regardless of whether that base is quantized or full bf16 —
    # gradient checkpointing still needs this hook either way.
    base_model.enable_input_require_grads()

    base_model.config.use_cache = False

    # ── MoE-LoRA wrap ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Wrapping with MALoRA...")

    moe_config = LoraMoeConfig.from_pretrained(MODEL_ID)    
    moe_config.experts_rank         = EXPERTS_RANK
    moe_config.attention_rank       = ATTENTION_RANK   # NEW: attention LoRA rank
    moe_config.experts_scale        = EXPERTS_SCALE
    moe_config.num_experts_per_tok  = NUM_EXPERTS_PER_TOK
    moe_config.num_local_experts    = NUM_EXPERTS
    moe_config.output_router_logits = True
    moe_config.router_aux_loss_coef = ROUTER_AUX_COEF
    moe_config.use_attention_lora   = True             # NEW: enable attention LoRA

    moe_model = LoraMoeModel(base_model, moe_config)
    moe_model.make_experts_trainable()
    moe_model.base_model.gradient_checkpointing_enable()
    moe_model.base_model.config.use_cache = False

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in moe_model.parameters())
        print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

        # dtype summary
        active_dtypes   = {}
        inactive_dtypes = {}
        for p in moe_model.parameters():
            key = str(p.dtype)
            if p.requires_grad:
                active_dtypes[key]   = active_dtypes.get(key, 0) + 1
            else:
                inactive_dtypes[key] = inactive_dtypes.get(key, 0) + 1
        print("Active dtypes:",   active_dtypes)
        print("Inactive dtypes:", inactive_dtypes)

        # OPTION 1 sanity check: confirm actual VRAM footprint post-wrap
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved  = torch.cuda.memory_reserved() / 1e9
            total     = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM after model wrap: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        optim="adamw_bnb_8bit" if USE_8BIT_ADAM else "adamw_torch",
        lr_scheduler_type="cosine_with_restarts",   # FIXED: better than plain cosine
        warmup_ratio=0.1,                           # FIXED: was 0.05, more warmup for MoE
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_dir=LOGDIR,
        run_name=RUN_NAME,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,                          # FIXED: was 1.0, MoE needs tighter clipping
        seed=SEED,
        save_total_limit=NUM_CHECKPOINT_LIMIT,
        save_strategy="steps",
        save_steps=EVAL_STEPS,                      # saves every eval → always have recent checkpoint
        report_to="wandb",
        remove_unused_columns=False,                # FIXED: was True in gemini, False is correct
        load_best_model_at_end=False,                # NEW: auto-loads best checkpoint at end
        greater_is_better=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = MoETrainer(
        tokenizer=tokenizer,
        model=moe_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[WandbConfigCallback()],
    )

    # register trainer for emergency save on crash
    _trainer_ref = trainer

    # ── Train ─────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("\nStarting training...")
        print(f"  Steps per epoch: {len(train_dataset) // (TRAIN_BATCH * GRAD_ACCUM)}")
        print(f"  Total steps: {MAX_STEPS if MAX_STEPS > 0 else 'full'}")

    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save ──────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
