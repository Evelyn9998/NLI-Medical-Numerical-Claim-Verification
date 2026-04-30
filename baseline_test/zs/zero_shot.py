"""
LLM Fact-Check Evaluator
Uses DeepSeek-R1-7B (or Llama-3.1-8B-Instruct / Qwen) via local HuggingFace model.
Zero-shot Direct Prompting: Evidence + Claim -> JSON label via Logical Reasoner.

Prompting strategy: Direct prompting (no explicit Chain-of-Thought steps in the
prompt). DeepSeek R1 reasons internally via its <think> block by default; the
system prompt states the task and JSON schema directly without coaching the model
through numbered steps.

Benchmark accuracy criteria (MedCalc-Bench §3.1, Table 2):
  - Rule-based calculators (integer scores from summed criteria) : exact match
  - Equation-based calculators (continuous decimal output)       : within 5%
  - Date-based calculators (calendar date output)               : exact match

The model self-classifies the calculator type from the claim and applies
the correct criterion. No Category column is required in the dataset.

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match | model_reasoning | dataset_explanation

Reproducibility controls:
  - Always fp16 (4-bit quantization removed; consistent precision across GPUs)
  - --batch-size defaults to 1 (eliminates padding-induced numerical drift)
  - FIXED_SEED = 42 hardcoded; seeds torch / random / numpy before any operation
  - Greedy decoding (temperature=0, do_sample=False) is enforced throughout
  - torch.backends.cudnn.deterministic = True

Install dependencies:
    pip install transformers torch

Usage:
    python zero_shot.py --data "mrt_claim_cleaned.csv" --model "$HOME/models/DeepSeek-R1-7B"
    python zero_shot.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct"
    python zero_shot.py --data "train_300.csv"          --model "$HOME/models/Qwen2.5-14B-Instruct" --samples 20 --random
    python zero_shot.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0
    python zero_shot.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --batch-size 1
"""

import argparse
import csv
import json
import os
import re
import random
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Fixed random seed — hardcoded to ensure full reproducibility without needing
# a CLI flag. Seeds Python / NumPy / PyTorch before any data or model op.
# ---------------------------------------------------------------------------
FIXED_SEED = 42

# ---------------------------------------------------------------------------
# System prompt — Zero-shot Direct Prompting for DeepSeek R1 7B
#
# Pure direct prompt: no reasoning structure, no JSON schema, no step
# decomposition. The model receives the task definition and label options only.
# DeepSeek R1 reasons internally via its <think> block; we strip that block
# in _parse_response and extract the label from the remaining free text.
#
# This is the baseline. Structured reasoning (CoT, PoT, etc.) will be
# introduced as separate prompting strategies on top of this foundation.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given clinical evidence and a clinical claim, determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The claim is a comma-separated list of entities. All segments EXCEPT the last are input parameters; the LAST segment is the final computed result.

Check each input entity against the evidence, compute the correct result using evidence-derived values, and assign the label:
- "true"          : ALL input entities match the evidence AND the computed result matches the claimed result.
- "partially true": ALL input entities match the evidence BUT the computed result does NOT match.
- "false"         : ANY input entity does NOT match the evidence.

Use the appropriate accuracy criterion:
- Rule-based calculator (integer score — e.g. HEART, CHA2DS2-VASc, Wells, CURB-65, GCS, Caprini, APACHE II, SOFA): exact match.
- Equation-based calculator (decimal output — e.g. GFR, LDL, BMI, MAP, FIB-4, Anion Gap, Serum Osmolality, Cockcroft-Gault, Framingham Risk): within 5%, i.e. |claimed − correct| / |correct| ≤ 0.05.
- Date-based calculator (calendar date — e.g. Estimated Due Date, Gestational Age): exact match.

For every INPUT entity (all except the last) mentioned in the claim, check the evidence and set:
- correct = true  if the claim value matches the evidence value.
- correct = false if the claim value does not match, or if a required condition is absent.
Special cases:
  - A positively-framed entity (e.g., "Liver disease criteria for HAS-BLED") must be present in the evidence. If not found → correct = false.
  - A negatively-framed entity (e.g., "Cough Absent") describes absence. If the condition is not mentioned in the evidence → correct = true."""





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_label(label: str) -> str:
    label = (label or "").lower().strip()
    if label == "true":
        return "true"
    if label == "false":
        return "false"
    if "partial" in label:
        return "partially true"
    return label


def _truncate_repetition(text: str, phrase_min_len: int = 20, threshold: int = 3) -> str:
    pattern = re.compile(
        r'(.{' + str(phrase_min_len) + r',}?)\1{' + str(threshold - 1) + r',}',
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        return text[:m.start()].strip()
    return text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Evidence") and row.get("Claim") and row.get("Label"):
                rows.append({
                    "evidence":    row["Evidence"],
                    "claim":       row["Claim"],
                    "label":       normalize_label(row["Label"]),
                    "explanation": row.get("Explanation", "").strip(),
                })
    print(f"Loaded {len(rows)} samples from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Model loading  (always fp16 — no 4-bit quantization)
# ---------------------------------------------------------------------------

def load_model(model_name: str, batch_size: int) -> dict:
    """
    Load the model in fp16 on GPU, or fp32 on CPU.

    4-bit quantization has been intentionally removed:
      - bitsandbytes quantization introduces hardware-dependent rounding error
        that varies between GPU architectures, making results non-reproducible
        across different machines even with the same seed and batch size.
      - fp16 is consistent across GPUs of the same compute capability and
        is the correct baseline for controlled experiments.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure a pad token exists; use left-padding to keep the generated
    # token positions stable regardless of sequence length in a batch.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_gpu = torch.cuda.is_available()
    print(f"  GPU available: {has_gpu}")

    if has_gpu:
        print("  Loading in fp16 (deterministic across same GPU architecture)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        print("  No GPU — loading in fp32 on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    print(f"Model ready: {model_name}  (batch_size={batch_size}, dtype=fp16)")
    return {"pipeline": pipe, "model_name": model_name}


# ---------------------------------------------------------------------------
# Single-response post-processing (shared between batch and retry paths)
# ---------------------------------------------------------------------------

def _parse_response(raw_text: str, claim: str = "") -> dict:
    """
    Convert raw model text into {"reasoning": str, "label": str}.

    Pipeline:
      1. Truncate runaway repetitions (R1 looping guard).
      2. Strip <think> blocks (DeepSeek R1 internal reasoning) in all forms:
           a. Properly closed  <think>...</think>
           b. Orphaned </think> with no opening tag
           c. Unclosed <think> — model ran out of tokens mid-thinking
      3. Extract the label from the remaining free text via _keyword_label.
    """
    raw_text = _truncate_repetition(raw_text)

    # ── Strip <think> reasoning block (DeepSeek R1) ────────────────────────
    # Case a: properly closed <think>...</think>
    raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    # Case b: orphaned </think> with no opening tag
    raw_text = re.sub(r"^[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    # Case c: unclosed <think> — strip from <think> to end of string
    if re.search(r"<think>", raw_text, re.IGNORECASE):
        raw_text = re.sub(r"<think>[\s\S]*$", "", raw_text, flags=re.IGNORECASE).strip()

    clean = raw_text.strip()

    # ── Extract label from free text ───────────────────────────────────────
    label = _keyword_label(clean)
    return {"reasoning": clean, "label": label}


def _keyword_label(text: str) -> str:
    """
    Last-resort label extraction from free text when JSON parsing fails.
    Priority: partially true > false > true.
    Defaults to "false" (never returns "unknown").
    """
    t = text.lower()
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    if re.search(r'label[":\s]+false', t) or re.search(r'assign.*\bfalse\b', t):
        return "false"
    if re.search(r'label[":\s]+true', t) or re.search(r'assign.*\btrue\b', t):
        return "true"
    last_false = t.rfind('"false"')
    last_true  = t.rfind('"true"')
    if last_false > last_true:
        return "false"
    if last_true > last_false:
        return "true"
    if "false" in t:
        return "false"
    if "true" in t:
        return "true"
    return "false"


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def _build_messages(item: dict, ev_limit: int, no_think: bool = False) -> list:
    """
    Build the chat message list for one sample.

    Direct prompting: the user message contains only the evidence and claim.
    No JSON instructions, no step scaffolding.

    no_think is accepted for interface compatibility but has no effect here —
    DeepSeek R1's internal <think> block is stripped in _parse_response.
    """
    user_msg = (
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def run_batch(model: dict, batch: list, ev_limit: int,
              max_tokens: int,
              no_think: bool = False) -> list:
    """
    Run greedy (temperature=0) inference on a list of items in one pipeline call.
    Returns a list of {"reasoning": str, "label": str} dicts, one per item.

    temperature and repetition_penalty arguments have been removed:
      - do_sample=False (greedy) is always used for reproducibility.
      - repetition_penalty has no effect under greedy decoding and is omitted.
    """
    pipe         = model["pipeline"]
    all_messages = [_build_messages(item, ev_limit, no_think=no_think) for item in batch]

    results = pipe(
        all_messages,
        max_new_tokens=max_tokens,
        do_sample=False,          # greedy decoding — deterministic
        temperature=None,         # must be None when do_sample=False
        repetition_penalty=1.0,   # no penalty; no effect under greedy anyway
        return_full_text=False,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )

    return [
        _parse_response(r[0]["generated_text"].strip(), item.get("claim", ""))
        for r, item in zip(results, batch)
    ]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    n = len(results)
    if n == 0:
        return

    correct = sum(1 for r in results if r["match"])
    print(f"\n=== Summary ===")
    print(f"Total   : {n}")
    print(f"Correct : {correct}")
    print(f"Wrong   : {n - correct}")
    print(f"Accuracy: {correct / n * 100:.1f}%")

    from collections import defaultdict
    label_stats = defaultdict(lambda: [0, 0])
    pred_dist   = defaultdict(int)
    for r in results:
        label_stats[r["true_label"]][1] += 1
        if r["match"]:
            label_stats[r["true_label"]][0] += 1
        pred_dist[r["predicted"]] += 1

    print("\nAccuracy by true label:")
    for lbl in ["true", "false", "partially true"]:
        c, t = label_stats[lbl]
        if t:
            print(f"  {lbl:<16} {c}/{t}  ({c/t*100:.0f}%)")

    print("\nPrediction distribution:")
    for lbl, cnt in sorted(pred_dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<16} {cnt}  ({cnt/n*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HuggingFace evaluator — Zero-shot Fact-Check with Structured CoT "
            "+ Logical Reasoner. Optimised for DeepSeek-R1; also compatible with "
            "Llama-3 and Qwen. Requires only Evidence, Claim, and Label columns."
        )
    )
    parser.add_argument("--data", required=True,
                        help="Path to CSV dataset, e.g. mrt_claim_cleaned.csv")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to test (default: 10; set 0 for all)")
    parser.add_argument("--random", action="store_true",
                        help="Randomly sample instead of taking from the start")
    parser.add_argument("--ev-limit", type=int, default=2000,
                        help="Max characters of Evidence passed to model (default: 2000)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help=(
                            "Max output tokens per inference (default: 4096). "
                            "Llama outputs JSON directly without a reasoning block, "
                            "so 2048 is sufficient. For Qwen without --no-think, "
                            "increase to 8192 to fit the <think> block + JSON."
                        ))
    # -----------------------------------------------------------------------
    # Reproducibility: batch-size defaults to 1.
    #
    # When batch_size > 1, sequences are padded to the same length inside the
    # batch. Even with causal masking, different padding lengths change the
    # floating-point accumulation in attention layers, so the same sample can
    # produce a different output depending on which other samples it is batched
    # with. batch_size=1 eliminates this entirely.
    #
    # If you increase batch_size for throughput, results may differ from a
    # batch_size=1 run — this is expected and not a bug.
    # -----------------------------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=1,
                        help=(
                            "Samples per GPU batch (default: 1 for reproducibility). "
                            "Increasing this pads sequences together and causes "
                            "non-deterministic results across different batch sizes "
                            "or GPU architectures. Increase only for throughput "
                            "experiments where exact reproducibility is not required."
                        ))
    parser.add_argument("--start", type=int, default=1,
                        help="1-based index to start from (default: 1)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between batches (default: 0.0)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: auto-generated with timestamp)")
    parser.add_argument("--no-think", action="store_true",
                        help=(
                            "Prepend '/no_think' to user messages. "
                            "Disables the <think> reasoning block for Qwen3 thinking "
                            "models, so the full JSON fits within --max-tokens. "
                            "Has no effect on DeepSeek R1 or Llama."
                        ))
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Seed everything before any data or model operation.
    # FIXED_SEED is hardcoded (= 42) — no CLI flag needed.
    # Combined with batch_size=1 and greedy decoding this makes results fully
    # reproducible on the same GPU architecture.
    # -----------------------------------------------------------------------
    import numpy as np
    import torch

    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    print(f"Seed: {FIXED_SEED} (fixed)  |  batch_size: {args.batch_size}  |  dtype: fp16  |  decoding: greedy")

    if args.random and args.start != 1:
        print(f"Warning: --start {args.start} is ignored when --random is set.")

    if args.no_think:
        print("--no-think enabled: prepending '/no_think' to all user messages.")

    # -- Load data ---------------------------------------------------------
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples (seed={FIXED_SEED})")
    else:
        start_idx = args.start - 1
        subset = (all_data[start_idx:start_idx + n]
                  if args.samples != 0
                  else all_data[start_idx:])
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # -- Load model --------------------------------------------------------
    model = load_model(args.model, args.batch_size)

    # -- Prepare output CSV ------------------------------------------------
    out_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "model_reasoning", "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # -- Batched evaluation loop -------------------------------------------
    results = []
    total   = len(subset)
    print(f"Starting evaluation ({total} samples, batch_size={args.batch_size})...\n")

    for batch_start in range(0, total, args.batch_size):
        batch      = subset[batch_start : batch_start + args.batch_size]
        batch_idxs = list(range(args.start + batch_start,
                                args.start + batch_start + len(batch)))

        print(f"[{batch_idxs[0]}-{batch_idxs[-1]}/{args.start + total - 1}] "
              f"Running batch of {len(batch)}...")

        try:
            resps = run_batch(
                model, batch, args.ev_limit,
                args.max_tokens,
                no_think=args.no_think,
            )
        except Exception as e:
            print(f"  Batch error ({e}), falling back to one-by-one...")
            resps = []
            for item in batch:
                for attempt in range(3):
                    try:
                        msgs = _build_messages(item, args.ev_limit,
                                               no_think=args.no_think)
                        pipe = model["pipeline"]
                        raw  = pipe(
                            msgs,
                            max_new_tokens=args.max_tokens,
                            do_sample=False,
                            temperature=None,
                            repetition_penalty=1.0,
                            return_full_text=False,
                        )[0]["generated_text"].strip()
                        resps.append(_parse_response(raw, item.get("claim", "")))
                        break
                    except Exception as e2:
                        print(f"    [attempt {attempt+1}/3] Error: {e2}")
                        if attempt < 2:
                            time.sleep(30 * (attempt + 1))
                        else:
                            resps.append(
                                {"reasoning": f"Error: {e2}", "label": "false"}
                            )

        # Process results for this batch
        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for item, idx, resp in zip(batch, batch_idxs, resps):
                predicted = normalize_label(resp["label"])
                if predicted not in ("true", "partially true", "false"):
                    predicted = "false"

                match = predicted == item["label"]
                print(f"  [{idx}] true={item['label']}  predicted={predicted}  match={match}")

                row = {
                    "index":       idx,
                    "evidence":    item["evidence"],
                    "claim":       item["claim"],
                    "true_label":  item["label"],
                    "predicted":   predicted,
                    "match":       match,
                    "reasoning":   resp["reasoning"],
                    "explanation": item["explanation"],
                }
                results.append(row)

                writer.writerow({
                    "index":               row["index"],
                    "evidence":            row["evidence"],
                    "claim":               row["claim"],
                    "true_label":          row["true_label"],
                    "predicted_label":     row["predicted"],
                    "match":               "TRUE" if row["match"] else "FALSE",
                    "model_reasoning":     row["reasoning"].replace("\n", " "),
                    "dataset_explanation": row["explanation"].replace("\n", " "),
                })

        if args.delay > 0 and batch_start + args.batch_size < total:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()