from dataclasses import dataclass, field
from typing import Optional

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
    TRAIN_BATCH: int     = 24
    EVAL_BATCH: int      = 16
    CONTEXT_LENGTH: int  = 1024
    LR: float            = 2e-4          # FIXED: was 1e-4, faster convergence
    EVAL_STEPS: int      = 100           # more frequent for visibility
    GRAD_ACCUM: int      = 2             # SPEED TEST: batch 16 already gives effective batch 16
    MAX_STEPS: int       = -1          # SET TO -1 FOR REAL TRAINING, 500 for dev runs
    LOGGING_STEPS: int   = 25            # see progress faster

    # Model parameters
    USE_8BIT_ADAM: bool  = True
    MIXED_PRECISION: str = "bf16"
    QUANTIZE: bool       = False         # OPTION 1: full bf16, no 4-bit quantization — removes
                                          # dequant_4bit (~25%) + MatMul4Bit fwd/bwd (~53%) overhead
                                          # measured via profiler. Needs ~6GB for base weights
                                          # instead of ~1.4GB — trivial on L40S's 48GB.
    MODEL_ID: str        = "Qwen/Qwen2.5-Coder-3B-Instruct"   # using 3B as discussed

    # Logging parameters
    RESUME_FROM: str | None   = None     # set to checkpoint path to resume e.g "./outputs/malora/checkpoint-400"
    OUTPUT_DIR: str           = "./outputs/malora"
    NUM_CHECKPOINT_LIMIT: int = 2        # FIXED: was 3, keep more checkpoints for safety
    LOGDIR: str               = "./logs"
    RUN_NAME: str             = "10k-3ep-aton"
    PROJECT_NAME: str         = "malora"
