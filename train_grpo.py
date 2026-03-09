import os as _os_early
_os_early.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

"""
Step 3: RLVR with GRPO — Group Relative Policy Optimisation

Purpose:
  Phase 2 of training. Starting from the SFT checkpoint, use reinforcement
  learning to improve the MODEL's actual verification ACCURACY.

  In SFT, the model learned to mimic our pre-written explanations.
  In GRPO, the model must generate reasoning itself — it only gets
  feedback on whether the final label is correct.

  This forces the model to develop genuine numerical reasoning rather than
  pattern-matching to template explanations.

How GRPO works (simplified):
  For each prompt:
    1. Generate G=8 independent completions (with sampling)
    2. Score each with the reward function (format + accuracy)
    3. Compute advantage = reward - mean_reward_for_this_prompt
       (positive advantage → completion was better than average)
    4. Update model to increase probability of high-advantage completions
       and decrease probability of low-advantage ones
    5. A KL penalty prevents the model from drifting too far from SFT

  GRPO requires NO separate critic/value network (unlike PPO),
  making it much more memory-efficient for single-GPU training.

Key difference from SFT training data:
  SFT:  messages include assistant turn with full CoT explanation
  GRPO: messages include only system + user (model must reason itself)
        ground truth label is stored separately for the reward function

Run:
  python train_grpo.py
  python train_grpo.py --sft_checkpoint outputs/sft_checkpoint --num_steps 500
"""

import argparse
import re

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_fn import combined_reward


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    import sys
    sys.argv = [sys.argv[0]] + [a.strip() for a in sys.argv[1:]]
    p = argparse.ArgumentParser()
    # p.add_argument("--base_model",     default="Qwen/Qwen2.5-7B-Instruct",
    p.add_argument("--base_model",     default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Original base model (same as used for SFT)")
    p.add_argument("--sft_checkpoint", default="outputs/sft_checkpoint",
                   help="Path to LoRA adapter saved by train_sft.py")
    p.add_argument("--train_file",     default="data/grpo_train.jsonl")
    p.add_argument("--val_file",       default="data/grpo_val.jsonl")
    p.add_argument("--output_dir",     default="outputs/grpo_checkpoint")
    p.add_argument("--num_steps",      type=int, default=1000,
                   help="GRPO trains by steps, not epochs (data is large)")
    p.add_argument("--num_generations", type=int, default=8,
                   help="G: completions per prompt. Higher → better gradient estimate "
                        "but more VRAM. Reduce to 4 if OOM.")
    p.add_argument("--max_completion_length", type=int, default=1024,
                   help="Max tokens for model's CoT + answer generation")
    p.add_argument("--lr",             type=float, default=5e-6,
                   help="Much lower than SFT LR — we're fine-tuning an already-trained model")
    p.add_argument("--beta",           type=float, default=0.01,
                   help="KL penalty weight. Higher → stay closer to SFT policy. "
                        "Lower → allow more exploration but risk instability.")
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum",     type=int, default=4)
    p.add_argument("--use_wandb",      action="store_true")
    return p.parse_args()


# ── Reward wrapper for GRPOTrainer ───────────────────────────────────────────

def make_reward_fn():
    """
    GRPOTrainer expects a reward function with signature:
      fn(completions: list[str], **kwargs) -> list[float]

    The 'ground_truth' column from our dataset is passed as a kwarg.
    We wrap our combined_reward from reward_fn.py to match this interface.
    """
    def reward_fn(completions: list[str], ground_truth: list[str] = None, **kwargs) -> list[float]:
        if ground_truth is None:
            raise ValueError("ground_truth must be provided via dataset column")
        return combined_reward(completions, ground_truth=ground_truth)
    return reward_fn


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_grpo_dataset(path: str, tokenizer):
    """
    GRPO dataset format required by TRL's GRPOTrainer:
      - "prompt": the formatted prompt string (system + user, no assistant)
      - "ground_truth": the correct label (passed to reward function)

    We apply the chat template here so GRPOTrainer receives ready-to-use prompts.
    The template adds the model's special tokens (e.g. <|im_start|>system...).
    """
    raw = load_dataset("json", data_files=path, split="train")

    def format_row(example):
        # Apply chat template to system + user messages only
        # add_generation_prompt=True appends the assistant start token,
        # so the model knows it should continue generating
        prompt = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "ground_truth": example["label"],
        }

    return raw.map(format_row, remove_columns=raw.column_names)


def main():
    args = parse_args()

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    # Use SFT checkpoint if it exists locally; otherwise fall back to base model.
    # (SFT checkpoint won't exist until train_sft.py has been run.)
    import os
    tokenizer_path = args.sft_checkpoint if os.path.isdir(args.sft_checkpoint) else args.base_model
    print(f"Loading tokenizer from: {tokenizer_path}")
    if tokenizer_path == args.base_model:
        print("  [Warning] SFT checkpoint not found — loading tokenizer from base model.")
        print("  Run train_sft.py first for full pipeline.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Prepare datasets ──────────────────────────────────────────────────
    print("Preparing GRPO datasets ...")
    train_dataset = prepare_grpo_dataset(args.train_file, tokenizer)
    val_dataset   = prepare_grpo_dataset(args.val_file,   tokenizer)
    print(f"  train={len(train_dataset):,}  val={len(val_dataset):,}")
    print(f"  Sample prompt (first 300 chars): {train_dataset[0]['prompt'][:300]}")

    # ── 3. GRPOConfig ────────────────────────────────────────────────────────
    # Key settings:
    #   num_generations: G — how many completions per prompt to sample.
    #     More = better advantage estimates, but G × batch_size responses
    #     must fit in VRAM at once during generation.
    #   kl_coeff (β): KL divergence penalty weight.
    #     Prevents the policy from moving too far from the SFT checkpoint.
    #     β=0.01 is permissive; increase to 0.1 if training becomes unstable.
    #   temperature=0.7: sampling temperature during generation.
    #     Must be > 0 for diversity across the G completions.
    #     If all G completions are identical, advantage = 0 → no learning signal.
    import torch
    import os as _os
    use_cuda = torch.cuda.is_available()
    # MPS has known matmul bugs that crash GRPO generation — must use CPU on Apple Silicon
    device = "cuda" if use_cuda else "cpu"
    print(f"  Device: {device.upper()} — bf16={'on' if use_cuda else 'off'}")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.num_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.beta,
        # Generation settings
        temperature=0.7,
        # Memory: bf16 and 8bit optimizer only on CUDA
        use_cpu=(device == "cpu"),
        gradient_checkpointing=use_cuda,
        bf16=use_cuda,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        # Logging
        logging_steps=10,
        report_to="wandb" if args.use_wandb else "none",
        run_name="grpo_medcalc_rlvr",
    )

    # ── 4. Load SFT model + LoRA ─────────────────────────────────────────────
    # Load the base model, then apply the SFT LoRA adapter on top.
    # GRPOTrainer will keep a frozen reference copy of this model
    # to compute the KL penalty during training.
    print(f"Loading base model: {args.base_model}")
    print(f"Applying SFT LoRA from: {args.sft_checkpoint}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device,
    )
    if os.path.isdir(args.sft_checkpoint):
        model = PeftModel.from_pretrained(base_model, args.sft_checkpoint, is_trainable=True)
    else:
        # No SFT checkpoint yet — train GRPO directly on the base model.
        # This is valid for testing but skips the CoT SFT warm-up phase.
        print("  [Warning] No SFT LoRA adapter found — using base model for GRPO.")
        model = base_model

    # ── 5. GRPOTrainer ───────────────────────────────────────────────────────
    # reward_funcs: list of reward functions. Each receives (completions, **dataset_cols)
    # The dataset column "ground_truth" is automatically passed as a kwarg
    # because it's a column in our dataset.
    # Compatibility patch: transformers 5.x removed warnings_issued but TRL 0.14 still uses it
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    print("Initialising GRPOTrainer ...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=[make_reward_fn()],
        processing_class=tokenizer,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("Starting GRPO (RLVR) training ...")
    print(f"  SFT checkpoint:   {args.sft_checkpoint}")
    print(f"  Num generations:  {args.num_generations} per prompt")
    print(f"  Max completion:   {args.max_completion_length}")
    print(f"  KL coeff (β):     {args.beta}")
    print(f"  LR:               {args.lr}")
    print()
    print("Watch for:")
    print("  rewards/accuracy_reward — should increase over steps")
    print("  rewards/format_reward   — should stay near 0.2 (format maintained)")
    print("  kl                      — should stay small (< 1.0 is healthy)")
    print()

    trainer.train()

    # ── 7. Save ───────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nGRPO checkpoint saved to: {args.output_dir}")
    print("Next step: run evaluate.py on outputs/grpo_checkpoint vs outputs/sft_checkpoint")


if __name__ == "__main__":
    main()
