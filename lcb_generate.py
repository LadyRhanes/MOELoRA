# lcb_generate.py
"""
LiveCodeBench generation script for MALoRA (MoE-LoRA on Qwen2.5-Coder).

Reconstructs the exact same model wrap used during training (no quantization),
loads checkpoint weights, and generates n_samples solutions per problem.

Usage:
    python lcb_generate.py \
        --checkpoint_dir /teamspace/studios/this_studio/outputs/malora/checkpoint-100 \
        --base_model_id Qwen/Qwen2.5-Coder-7B-Instruct \
        --output_file lcb_outputs_ckpt100.json \
        --experts_rank 8 \
        --attention_rank 32 \
        --num_experts 8 \
        --num_experts_per_tok 2 \
        --experts_scale 1.0 \
        --router_aux_coef 0.001 \
        --n_samples 10 \
        --temperature 0.2 \
        --batch_size 1 \
        --release_version release_v5 \
        --resume
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/LiveCodeBench")
from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.benchmarks import load_code_generation_dataset
# ── point to your MALoRA repo root ────────────────────────────────────────────
MALORA_REPO = "/teamspace/studios/this_studio"
sys.path.insert(0, MALORA_REPO)
from modelling import LoraMoeModel
from configuration_lora_moe import LoraMoeConfig
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Solve the given competitive programming problem. "
    "Write only the complete Python solution code, no explanations."
)


def parse_args():
    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--checkpoint_dir",  required=True,
                   help="Path to checkpoint-XXXX folder")
    p.add_argument("--base_model_id",   default="Qwen/Qwen2.5-Coder-7B-Instruct",
                   help="HF model ID or local path for base Qwen2.5-Coder")
    p.add_argument("--output_file",     default="lcb_outputs.json")

    # MoE-LoRA hyperparams — must match what was used during training
    p.add_argument("--experts_rank",        type=int,   default=8)
    p.add_argument("--attention_rank",      type=int,   default=32)
    p.add_argument("--experts_scale",       type=float, default=1.0)
    p.add_argument("--num_experts",         type=int,   default=8)
    p.add_argument("--num_experts_per_tok", type=int,   default=2)
    p.add_argument("--router_aux_coef",     type=float, default=0.001)

    # generation
    p.add_argument("--n_samples",       type=int,   default=10)
    p.add_argument("--temperature",     type=float, default=0.2)
    p.add_argument("--max_new_tokens",  type=int,   default=2048)
    p.add_argument("--release_version", default="release_v5")

    # misc
    p.add_argument("--resume", action="store_true",
                   help="Skip problems already present in output_file")
    return p.parse_args()


def build_model(args):
    """
    Reconstructs the MALoRA model exactly as training did:
      1. Load base Qwen2.5-Coder in bf16 (no quantization for inference)
      2. Build LoraMoeConfig with same hyperparams
      3. Wrap with LoraMoeModel (patches layers + forward)
      4. Load checkpoint weights into base_model
    """
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"[1/4] Loading base model: {args.base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = True   # enable KV cache for generation

    print(f"[2/4] Building LoraMoeConfig")
    moe_config = LoraMoeConfig.from_pretrained(args.base_model_id)
    moe_config.experts_rank         = args.experts_rank
    moe_config.attention_rank       = args.attention_rank
    moe_config.experts_scale        = args.experts_scale
    moe_config.num_experts_per_tok  = args.num_experts_per_tok
    moe_config.num_local_experts    = args.num_experts
    moe_config.output_router_logits = False   # not needed for inference
    moe_config.router_aux_loss_coef = args.router_aux_coef
    moe_config.use_attention_lora   = True

    print(f"[3/4] Wrapping with LoraMoeModel (patches decoder layers + forward)")
    moe_model = LoraMoeModel(base_model, moe_config)

    print(f"[4/4] Loading checkpoint weights from: {checkpoint_dir}")
    state_dict_path = checkpoint_dir / "model.safetensors"
    if state_dict_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path, device="cpu")
    else:
        # fallback to pytorch_model.bin
        state_dict = torch.load(
            checkpoint_dir / "pytorch_model.bin",
            map_location="cpu",
        )

    # The checkpoint was saved via base_model.save_pretrained(), so keys
    # are base_model keys — load directly into moe_model.base_model
    missing, unexpected = moe_model.base_model.load_state_dict(
        state_dict, strict=False
    )
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    moe_model.base_model.eval()
    device = next(moe_model.base_model.parameters()).device
    print(f"Model ready on: {device}")
    return moe_model


def load_tokenizer(checkpoint_dir: str, base_model_id: str):
    # tokenizer is saved in the checkpoint dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, padding_side="left"
        )
    except Exception:
        print("[warn] Tokenizer not found in checkpoint, loading from base model")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, padding_side="left"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(problem_statement: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem_statement},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_existing(output_file: str) -> dict:
    p = Path(output_file)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        print(f"[resume] {len(data)} problems already done")
        return {item["question_id"]: item for item in data}
    return {}


@torch.inference_mode()
def generate_samples(model, tokenizer, prompt: str, args) -> list[str]:
    """
    Generates n_samples completions for one prompt.
    One sample at a time — avoids padding waste and keeps VRAM stable.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    codes = []

    for _ in range(args.n_samples):
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        generated = output[0][input_len:]
        code = tokenizer.decode(generated, skip_special_tokens=True).strip()
        codes.append(code)

    return codes


def main():
    args = parse_args()

    print(f"Loading LCB dataset ({args.release_version})")
    # ds = load_dataset(
    # "livecodebench/code_generation_lite",
    # version_tag=args.release_version,
    # split="test",
    # #trust_remote_code=True,
    # )
    # print(f"Total problems: {len(ds)}")
    

    # replace the load_dataset block with:
    ds = load_code_generation_dataset(release_version=args.release_version)
    print(f"Total problems: {len(ds)}")

    model    = build_model(args)
    tokenizer = load_tokenizer(args.checkpoint_dir, args.base_model_id)

    results  = load_existing(args.output_file) if args.resume else {}
    problems = [p for p in ds if p.question_id not in results]    
    print(f"Problems remaining: {len(problems)}")

    # for problem in tqdm(problems, desc="Generating"):
    #     prompt = format_prompt(problem["question_content"], tokenizer)

    #     try:
    #         code_list = generate_samples(model, tokenizer, prompt, args)
    #     except torch.cuda.OutOfMemoryError:
    #         torch.cuda.empty_cache()
    #         print(f"\n[OOM] {problem['question_id']} — trying with max_new_tokens=512")
    #         args_copy       = argparse.Namespace(**vars(args))
    #         args_copy.max_new_tokens = 512
    #         try:
    #             code_list = generate_samples(model, tokenizer, prompt, args_copy)
    #         except torch.cuda.OutOfMemoryError:
    #             torch.cuda.empty_cache()
    #             print(f"[OOM again] Skipping {problem['question_id']}")
    #             code_list = ["# OOM"] * args.n_samples

    #     results[problem["question_id"]] = {
    #         "question_id": problem["question_id"],
    #         "code_list":   code_list,
    #     }

    #     # write after every problem — safe against session kills
    #     with open(args.output_file, "w") as f:
    #         json.dump(list(results.values()), f, indent=2)
    for problem in tqdm(problems, desc="Generating"):
        prompt = format_prompt(problem.question_content, tokenizer)

        try:
            code_list = generate_samples(model, tokenizer, prompt, args)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n[OOM] {problem.question_id} — trying with max_new_tokens=512")
            args_copy       = argparse.Namespace(**vars(args))
            args_copy.max_new_tokens = 512
            try:
                code_list = generate_samples(model, tokenizer, prompt, args_copy)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[OOM again] Skipping {problem.question_id}")
                code_list = ["# OOM"] * args.n_samples

        results[problem.question_id] = {
            "question_id": problem.question_id,
            "code_list":   code_list,
        }

        # write after every problem — safe against session kills
        with open(args.output_file, "w") as f:
            json.dump(list(results.values()), f, indent=2)

    print(f"\nDone. {len(results)} problems in {args.output_file}")


if __name__ == "__main__":
    main()