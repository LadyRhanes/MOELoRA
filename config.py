# ── dataset_config.py ─────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CommitPackFTConfig:
    min_samples    : int            = 10000
    priority_langs : set            = field(default_factory=lambda: {"python", "java", "javascript", "cpp"})
    fmt_split      : tuple          = (0.6, 0.4)   # (code_edit, message_gen)
    lang_counts    : Optional[dict] = None
    langs_to_keep  : set = field(default_factory=lambda: {  # modern languages filtered out
        "python",
        "javascript",
        "typescript",
        "go",
        "java",
        "c#",
        "shell",
        "css",
        "scss",
        "cpp",
    })


@dataclass
class DatasetConfig:
    name    : str
    weight  : float
    # commitpackft-specific — ignored by all other loaders
    commitpackft: CommitPackFTConfig = field(default_factory=CommitPackFTConfig)


# ── GLOBAL TRAINING CONFIG ────────────────────────────────────────────────────

TOTAL_SAMPLES  = 10000    # 10k dev / 250k full run
EVAL_FRACTION  = 0.05

DATASET_CONFIGS = [
    # DatasetConfig(name="starcoderdata",       weight=0.32),
    # DatasetConfig(name="commitpackft",        weight=0.20, commitpackft=CommitPackFTConfig(
    #     min_samples    = 4000,
    #     priority_langs = {"python", "java", "javascript", "cpp"},
    #     fmt_split      = (0.7, 0.3),
    #     lang_counts    = None,   # paste hardcoded dict here after first run
    # )),
    # DatasetConfig(name="multipl-e",           weight=0.16),
    DatasetConfig(name="code_contests",       weight=0.15),
    DatasetConfig(name="codealpaca",          weight=0.15),
    DatasetConfig(name="python_instructions", weight=0.10),
    DatasetConfig(name="magicoder",           weight=0.15),
    DatasetConfig(name="leetcode",            weight=0.15),
    DatasetConfig(name="code_feedback",       weight=0.10),
    DatasetConfig(name="code_search_net",     weight=0.10),
    DatasetConfig(name="sql_context",     weight=0.10),
]

assert abs(sum(d.weight for d in DATASET_CONFIGS) - 1.0) < 1e-6, \
    "DATASET_CONFIGS weights must sum to 1.0"
    
    
    #because idk how packages work
@dataclass
class TrainingConfig:

    # LoRA MoE parameters
    EXPERTS_RANK: int        = 8          # MLP expert rank — keep low, MoE handles diversity
    ATTENTION_RANK: int      = 32         # Attention LoRA rank — higher, mentor said attn matters more
    EXPERTS_SCALE: float     = 1.0
    NUM_EXPERTS_PER_TOK: int = 2          # top-2 routing
    NUM_EXPERTS: int         = 8          # total experts
    ROUTER_AUX_COEF: float   = 0.001     # FIXED: was 0.01, too high causes over-regularization

    # Training parameters
    SEED: int            = 42
    NUM_EPOCHS: int      = 3
    TRAIN_BATCH: int     = 16            # FIXED: was 2, L4 can handle 4
    EVAL_BATCH: int      = 4             # FIXED: was 2
    CONTEXT_LENGTH: int  = 1024           # FIXED: was 1024, mean seq len is 430 anyway → 4x attn speedup
    LR: float            = 2e-4          # FIXED: was 1e-4, faster convergence
    EVAL_STEPS: int      = 100           # more frequent for visibility
    GRAD_ACCUM: int      = 1             # FIXED: was 4, effective batch = 8 same as before
    MAX_STEPS: int       = -1          # SET TO -1 FOR REAL TRAINING, 500 for dev runs
    LOGGING_STEPS: int   = 25            # see progress faster

    # Model parameters
    USE_8BIT_ADAM: bool  = True
    MIXED_PRECISION: str = "bf16"
    QUANTIZE: bool       = True
    MODEL_ID: str        = "Qwen/Qwen2.5-Coder-3B-Instruct"   # using 3B as discussed

    # Logging parameters
    RESUME_FROM: str | None   = None     # set to checkpoint path to resume e.g "./outputs/malora/checkpoint-400"
    OUTPUT_DIR: str           = "./outputs/malora"
    NUM_CHECKPOINT_LIMIT: int = 5        # FIXED: was 3, keep more checkpoints for safety
    LOGDIR: str               = "./logs"
    RUN_NAME: str             = "malora-run1"
    PROJECT_NAME: str         = "malora"
