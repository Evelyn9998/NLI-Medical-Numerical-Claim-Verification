"""
LLM Fact-Check Evaluator
Uses Llama-3.1-8B-Instruct via local HuggingFace model.
Zero-shot Direct Prompting: Evidence + Claim -> Label prediction with CoT.

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match | model_reasoning | dataset_explanation

Install dependencies:
    pip install transformers torch

Usage:
    python evaluate.py --data "train_300.csv" --model "$HOME/models/Llama-3.1-8B-Instruct"
    python evaluate.py --data "train_300.csv" --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 20 --random
    python evaluate.py --data "train_300.csv" --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given clinical evidence and a clinical claim that involves numerical entities and a medical calculation, determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The claim contains a list of entities (input values extracted from the patient record) followed by a final computed result (the last entity). Work through the following steps carefully.

---

STEP 1 — Check each input entity (all entities EXCEPT the last one):
For every input entity mentioned in the claim:
a) Look for that entity in the evidence.
   - If the entity is NOT present in the evidence AND the claim states its value as "none" or "no", treat it as CORRECT.
   - If the entity is NOT present in the evidence AND the claim states a specific numeric value, treat it as INCORRECT → the overall label is FALSE.
b) If the entity IS present in the evidence, compare the claim value with the evidence value and note whether they match.

STEP 2 — Perform the medical calculation step by step:
a) State the formula used.
b) Substitute each input value from the evidence (not from the claim) into the formula.
c) Show every arithmetic step and arrive at the correct computed result.
d) Compare the correct computed result with the value stated in the claim's last entity.

STEP 3 — Assign the label using these definitions:

- "true": ALL input entities (all except the last) are correct AND the final calculation result is correct.
  Correctness criteria for the calculation result:
  • Rule-based calculators (e.g., scoring systems with discrete categories): the claimed answer must exactly match the ground-truth answer.
  • Equation-based calculators (lab tests, physical measurements, dosage conversions): the claimed answer must be within 5% of the correct computed answer.
  • Date-based equations: the claimed date must exactly match the correct date.

- "partially true": ALL input entities (all except the last) are correct, BUT the final calculation result in the claim is incorrect (i.e., it deviates from the correct value). Note: the last entity (the computed result) is NOT used to judge whether entities are correct — only the preceding input entities are checked for "partially true".

- "false": One or more input entities (all except the last) are incorrect AND the final calculation result is also incorrect.

---

IMPORTANT RULES:
- An entity not found in the evidence is treated as INCORRECT unless the claim explicitly states its value as "none" or "no".
- Always use the evidence values (not the claim's input values) when performing your calculation in Step 2.
- Show the full formula and every arithmetic step before assigning the label.

Respond in this exact JSON format (no other text, no markdown fences):
{
  "reasoning": "<Step 1: entity-by-entity check | Step 2: formula, full calculation, result vs. claim | Step 3: label with justification>",
  "label": "<true|partially true|false>"
}"""

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
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    has_gpu = torch.cuda.is_available()
    print(f"  GPU available: {has_gpu}")

    if has_gpu:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("  4-bit quantization enabled (bitsandbytes)")
        except Exception as e:
            print(f"  bitsandbytes not available ({e}), loading in fp16")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        print("  No GPU, loading on CPU (fp32)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"Model ready: {model_name}")
    return pipe, model_name


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(model, evidence: str, claim: str, ev_limit: int,
               temperature: float, max_tokens: int) -> dict:
    pipe, model_name = model
    user_msg = (
        f"Evidence:\n{evidence[:ev_limit]}\n\n"
        f"Claim:\n{claim}\n\n"
        "Respond only with the JSON object."
    )

    do_sample = temperature > 0.0
    result = pipe(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": user_msg}],
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        repetition_penalty=1.2,
        return_full_text=False,
    )
    raw_text = result[0]["generated_text"].strip()
    clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    try:
        parsed = json.loads(clean)
        return {
            "reasoning": parsed.get("reasoning", ""),
            "label":     parsed.get("label", ""),
        }
    except json.JSONDecodeError:
        lm = re.search(r'"label"\s*:\s*"([^"]+)"', clean, re.IGNORECASE)
        rm = re.search(r'"reasoning"\s*:\s*"([\s\S]+?)"(?:\s*[,}])', clean, re.IGNORECASE)
        return {
            "reasoning": rm.group(1) if rm else clean,
            "label":     lm.group(1) if lm else _fallback_label(clean),
        }


def _fallback_label(text: str) -> str:
    t = text.lower()
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    if '"true"' in t or re.search(r"\blabel\b.*\btrue\b", t):
        return "true"
    if '"false"' in t or re.search(r"\blabel\b.*\bfalse\b", t):
        return "false"
    return "unknown"


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

def save_results(results: list, output_path: str):
    if not results:
        return

    fieldnames = [
        "index",
        "evidence",
        "claim",
        "true_label",
        "predicted_label",
        "match",
        "model_reasoning",
        "dataset_explanation",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "index":               r["index"],
                "evidence":            r["evidence"],
                "claim":               r["claim"],
                "true_label":          r["true_label"],
                "predicted_label":     r["predicted"],
                "match":               "TRUE" if r["match"] else "FALSE",
                "model_reasoning":     r["reasoning"].replace("\n", " "),
                "dataset_explanation": r["explanation"].replace("\n", " "),
            })

    print(f"Results saved to: {output_path}")


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
        description="Llama-3.1-8B Groq API evaluator — Zero-shot Fact-Check with CoT"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to CSV dataset, e.g. train_200.csv"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of samples to test (default: 10; set 0 for all)"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Randomly sample instead of taking from the start"
    )
    parser.add_argument(
        "--ev-limit", type=int, default=2000,
        help="Max characters of Evidence passed to model (default: 2000)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max output tokens per inference (default: 2048)"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="1-based index to start from (default: 1)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to sleep between API calls (default: 2.0)"
    )
    parser.add_argument(
        "--output", default="",
        help="Output CSV path (default: auto-generated with timestamp)"
    )
    args = parser.parse_args()

    # Load dataset
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples")
    else:
        start_idx = args.start - 1
        subset = all_data[start_idx:start_idx + n] if args.samples != 0 else all_data[start_idx:]
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # Load model (Groq API client)
    model = load_model(args.model)

    # Prepare output file (write header immediately)
    out_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["index", "evidence", "claim", "true_label", "predicted_label",
                  "match", "model_reasoning", "dataset_explanation"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # Evaluation loop
    results = []
    n = len(subset)
    print(f"Starting evaluation ({n} samples)...\n")

    for i, item in enumerate(subset, start=args.start):
        print(f"[{i}/{args.start + n - 1}] Running inference...")
        predicted = "unknown"
        reasoning = ""
        for attempt in range(3):
            try:
                resp = call_model(
                    model=model,
                    evidence=item["evidence"],
                    claim=item["claim"],
                    ev_limit=args.ev_limit,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                predicted = normalize_label(resp["label"])
                reasoning = resp["reasoning"]
                if predicted != "unknown" or attempt == 2:
                    break
            except Exception as e:
                print(f"  [attempt {attempt+1}/3] Error: {e}")
                if attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    reasoning = f"Error: {e}"

        match = predicted == item["label"]
        row = {
            "index":       i,
            "evidence":    item["evidence"],
            "claim":       item["claim"],
            "true_label":  item["label"],
            "predicted":   predicted,
            "match":       match,
            "reasoning":   reasoning,
            "explanation": item["explanation"],
        }
        results.append(row)

        # Append this row to CSV immediately
        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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

        print(f"  true={item['label']}  predicted={predicted}  match={match}")
        if i < args.start + n - 1:
            time.sleep(args.delay)

    # Summary
    print_summary(results)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
