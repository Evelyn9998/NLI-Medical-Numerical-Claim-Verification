"""
Step 2: CoT SFT — Supervised Fine-Tuning with Chain of Thought

Purpose:
  Teach the model the REASONING FORMAT for medical claim verification:
    1. Extractive verification (check each entity against the note)
    2. Calculation verification (check the numerical result)
    3. Final verdict: true / partially true / false

  The model learns by imitating our pre-written Explanation field,
  which is already structured as a step-by-step CoT trace.

Why SFT first (before RLVR)?
  RLVR (GRPO) works by generating multiple completions and comparing their
  rewards. If the model doesn't know the expected output format at all,
  it will generate random text and GRPO training is very unstable.
  SFT gives the model a solid starting point for GRPO to refine.

Key choices:
  - LoRA (r=16): full fine-tuning needs ~80GB VRAM; LoRA fits in 24GB
  - Loss on assistant turn only: prevents model from learning to copy
    the patient note verbatim (instruction masking)
  - max_seq_length=8192: covers >90th percentile of input lengths
  - paged_adamw_8bit: reduces optimizer memory from ~14GB to ~3.5GB

Run:
  python train_sft.py
  python train_sft.py --model_name Qwen/Qwen2.5-3B-Instruct  # lighter option
"""

import argparse
import os

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--model_name",    default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--model_name",    default="Qwen/Qwen2.5-0.5B-Instruct")
    # p.add_argument("--train_file",    default="data/sft_train.jsonl")
    # p.add_argument("--val_file",      default="data/sft_val.jsonl")
    p.add_argument("--train_file",    default="data/sft_train_small.jsonl")
    p.add_argument("--val_file",      default="data/sft_val_small.jsonl")
    p.add_argument("--output_dir",    default="outputs/sft_checkpoint")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--num_epochs",    type=int, default=3)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum",    type=int, default=8)
    p.add_argument("--lora_r",        type=int, default=16)
    p.add_argument("--lora_alpha",    type=int, default=32)
    p.add_argument("--use_wandb",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    # We need the tokenizer to apply the chat template (system/user/assistant
    # formatting) and to set the pad token.
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen2.5 uses eos as pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length

    # ── 2. Load datasets ─────────────────────────────────────────────────────
    # Each line in our JSONL has a "messages" key (list of role/content dicts).
    # HuggingFace datasets will load this as-is.
    print("Loading datasets ...")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    print(f"  train={len(dataset['train']):,}  val={len(dataset['validation']):,}")
    # SFTTrainer's chat template processor only accepts the 'messages' column;
    # extra columns like 'label' cause a KeyError in newer TRL versions.
    for split in dataset:
        extra_cols = [c for c in dataset[split].column_names if c != "messages"]
        if extra_cols:
            dataset[split] = dataset[split].remove_columns(extra_cols)

    # ── 3. LoRA config ───────────────────────────────────────────────────────
    # We apply LoRA to the attention projection matrices (q, k, v, o).
    # These are the most important parameters for reasoning, and targeting
    # them alone is sufficient while keeping trainable params << 1% of total.
    #
    # r=16, alpha=32: standard starting point for 7B models.
    # Effective learning rate for LoRA = lr * (alpha / r) = 2e-4 * 2 = 4e-4.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # ── 4. Training arguments ─────────────────────────────────────────────────
    # Key settings explained:
    #   gradient_checkpointing: trades compute for memory (re-computes activations
    #     during backward pass instead of storing them → saves ~8GB VRAM)
    #   optim="paged_adamw_8bit": stores optimizer states in 8-bit, paged to CPU
    #     if VRAM is full → saves ~10GB vs standard AdamW
    #   load_best_model_at_end: keeps the checkpoint with lowest eval loss
    import torch
    import os as _os
    use_cuda = torch.cuda.is_available()
    use_mps = not use_cuda and torch.backends.mps.is_available()
    if use_mps:
        _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    print(f"  Device: {device.upper()} — bf16={'on' if use_cuda else 'off'}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        dataset_text_field=None,
        # Memory optimizations: bf16 and 8bit optimizer only on CUDA
        use_cpu=(device == "cpu"),
        gradient_checkpointing=use_cuda,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        bf16=use_cuda,
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Logging
        logging_steps=50,
        report_to="wandb" if args.use_wandb else "none",
        run_name="sft_medcalc_cot",
        # Misc
        dataloader_num_workers=0,  # 0 avoids multiprocessing deadlocks on Mac
    )

    # ── 5. SFTTrainer ────────────────────────────────────────────────────────
    # SFTTrainer handles:
    #   - Applying the model's chat template to our "messages" format
    #   - Masking loss on system/user turns (only train on assistant output)
    #   - Packing short sequences to improve GPU utilization
    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device,
        trust_remote_code=True,
    )

    print("Initialising SFTTrainer ...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("Starting CoT SFT training ...")
    print(f"  Model:        {args.model_name}")
    print(f"  Train size:   {len(dataset['train']):,}")
    print(f"  Max seq len:  {args.max_seq_length}")
    print(f"  LoRA r/alpha: {args.lora_r}/{args.lora_alpha}")
    print(f"  Effective batch: {args.per_device_batch * args.grad_accum}")
    print()

    trainer.train()

    # ── 7. Save final checkpoint ──────────────────────────────────────────────
    # Save both the LoRA adapter weights and the tokenizer.
    # These are the inputs to the GRPO training phase.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSFT checkpoint saved to: {args.output_dir}")
    print("Next step: run python train_grpo.py")


if __name__ == "__main__":
    main()
