"""
    # eval the 20k attn-off final model on both benchmarks
    python eval_malora.py --folder 20k_1ep_atoff --dataset both

    # eval a specific checkpoint
    python eval_malora.py --folder 20k_1ep_atoff/checkpoint-777 --dataset humaneval

    # eval the 10k run
    python eval_malora.py --folder 10k_1ep_atoff --dataset both

    # just run inference to sanity check (no benchmarks)
    python eval_malora.py --folder 20k_1ep_atoff --sanity-only
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

HF_TOKEN    = "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"
HF_REPO_ID  = "godofwar1007/moelora"
BASE_MODEL  = "Qwen/Qwen2.5-Coder-3B-Instruct"

MAX_NEW_TOKENS = 512
OUTPUT_DIR     = "eval_outputs"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel

def load_model(hf_folder: str):
    """
    Downloads the checkpoint from HF Hub and loads it correctly.

    hf_folder: subfolder inside godofwar1007/moelora, e.g. '20k_1ep_atoff'
               or '20k_1ep_atoff/checkpoint-777'
    """
    print(f"\n{'='*60}")
    print(f"Loading: {HF_REPO_ID}/{hf_folder}")
    print(f"{'='*60}\n")

    print("Downloading checkpoint from HF Hub...")
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        allow_patterns=[f"{hf_folder}/*", f"{hf_folder}/model.safetensors"],
        local_dir=f"./hf_cache/{hf_folder.replace('/', '_')}",
    )
    ckpt_path = os.path.join(local_dir, hf_folder, "model.safetensors")

    if not os.path.exists(ckpt_path):
        
        ckpt_path = os.path.join(local_dir, "model.safetensors")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"model.safetensors not found.\n"
            f"Looked at:\n"
            f"  {os.path.join(local_dir, hf_folder, 'model.safetensors')}\n"
            f"  {os.path.join(local_dir, 'model.safetensors')}\n"
            f"Files in local_dir: {os.listdir(local_dir)}"
        )

    print(f"Checkpoint found at: {ckpt_path}")
    print(f"Size: {os.path.getsize(ckpt_path)/1e9:.2f} GB\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # left pad for generation

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = True   # enable KV cache for generation
    base_model.enable_input_require_grads()

    print("Wrapping with LoraMoeModel...")
    moe_config = LoraMoeConfig.from_pretrained(BASE_MODEL)
    moe_config.experts_rank         = 8
    moe_config.attention_rank       = 32
    moe_config.experts_scale        = 1.0
    moe_config.num_experts_per_tok  = 2
    moe_config.num_local_experts    = 8
    moe_config.output_router_logits = False   # off for inference
    moe_config.router_aux_loss_coef = 0.001
    moe_config.use_attention_lora   = False   # attn lora off

    moe_model = LoraMoeModel(base_model, moe_config)

    print("Loading saved weights...")
    saved_sd = load_file(ckpt_path, device="cpu")

    # get the wrapped model's current state dict to compare keys
    model_sd = moe_model.state_dict()

    print(f"  Checkpoint keys: {len(saved_sd)}")
    print(f"  Model keys:      {len(model_sd)}")

    # find key mapping — checkpoint may be saved under base_model.* or directly
    # try direct load first
    matched   = {k: v for k, v in saved_sd.items() if k in model_sd}
    unmatched = [k for k in saved_sd if k not in model_sd]

    if len(matched) == 0:
        print("  Direct keys didn't match, trying prefix remapping...")
        remapped = {}
        for k, v in saved_sd.items():
         
            new_k = "base_model." + k
            if new_k in model_sd:
                remapped[new_k] = v
                continue

            if k.startswith("base_model."):
                new_k = k[len("base_model."):]
                if new_k in model_sd:
                    remapped[new_k] = v
                    continue
            remapped[k] = v   # keep as-is
        matched = {k: v for k, v in remapped.items() if k in model_sd}
        unmatched = [k for k in remapped if k not in model_sd]

    print(f"  Matched:   {len(matched)} keys")
    print(f"  Unmatched: {len(unmatched)} keys")
    if unmatched[:5]:
        print(f"  Sample unmatched: {unmatched[:5]}")

    # load matched weights
    missing = moe_model.load_state_dict(matched, strict=False)
    print(f"  Missing from checkpoint (frozen base is fine): {len(missing.missing_keys)}")

    # verification
    lora_keys_loaded = [k for k in matched if 'lora' in k.lower() or 'router' in k.lower() or 'gate' in k.lower()]
    print(f"  LoRA/router keys loaded: {len(lora_keys_loaded)}")
    if len(lora_keys_loaded) == 0:
        print("\n  WARNING: No LoRA keys found in checkpoint!")
        print("  This means either the checkpoint is wrong or key names don't match.")
        print("  Checkpoint key samples:", list(saved_sd.keys())[:10])
        print("  Model key samples:", list(model_sd.keys())[:10])

    moe_model.eval()
    print("\nModel ready.\n")
    return moe_model, tokenizer


#sanity check inference 

def sanity_check(model, tokenizer):
    print("Running sanity check inference...")
    test_prompts = [
        "def fibonacci(n):",
        "def binary_search(arr, target):",
        "# Write a function to reverse a string\ndef reverse_string(s):",
    ]
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are a Python coding assistant."},
            {"role": "user",   "content": f"Complete this Python function:\n\n{prompt}"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {completion[:200]}")
        print("-" * 40)


#generation 

def strip_markdown(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


# HumanEval+

def run_humaneval(model, tokenizer, tag: str):
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    print(f"\nHumanEval+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the function below. Return ONLY the function body — no explanation, no markdown, no extra text."},
            {"role": "user",   "content": f"Complete this Python function:\n\n{problem['prompt']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        start = time.time()
        completion = strip_markdown(generate(model, tokenizer, prompt))
        elapsed = time.time() - start

        results.append({"task_id": task_id, "completion": completion})

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(problems)}] last: {elapsed:.1f}s")

    out_path = os.path.join(OUTPUT_DIR, f"humaneval_{tag}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}")
    return out_path


# MBPP+ 
def run_mbpp(model, tokenizer, tag: str):
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    print(f"\nMBPP+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        tests = "\n".join(problem.get("test_list", [])[:2])
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Write a Python function that solves the given task. Return ONLY the function — no explanation, no markdown, no extra text."},
            {"role": "user",   "content": f"Task: {problem['prompt']}\n\nExample tests:\n{tests}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = strip_markdown(generate(model, tokenizer, prompt))
        results.append({"task_id": task_id, "completion": completion})

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(problems)}]")

    out_path = os.path.join(OUTPUT_DIR, f"mbpp_{tag}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}")
    return out_path


# scoring 

def score(dataset: str, completions_path: str):
    print(f"\nScoring {dataset}...")
    os.system(f"python -m evalplus.evaluate --dataset {dataset} --samples {completions_path}")


# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",      required=True,
                        help="HF subfolder, e.g. '20k_1ep_atoff' or '20k_1ep_atoff/checkpoint-777'")
    parser.add_argument("--dataset",     choices=["humaneval", "mbpp", "both"], default="both")
    parser.add_argument("--sanity-only", action="store_true",
                        help="just run 3 inference prompts to verify loading, skip benchmarks")
    args = parser.parse_args()

    tag = args.folder.replace("/", "_")

    model, tokenizer = load_model(args.folder)

    sanity_check(model, tokenizer)

    if args.sanity_only:
        print("\nSanity check done. Exiting (--sanity-only was set).")
        exit(0)

    if args.dataset in ("humaneval", "both"):
        score("humaneval", run_humaneval(model, tokenizer, tag))

    if args.dataset in ("mbpp", "both"):
        score("mbpp", run_mbpp(model, tokenizer, tag))
