"""
LLM Fact-Check Evaluator
Uses Llama-3.1-8B-Instruct via Groq API (free, no local GPU required).
Zero-shot Direct Prompting: Evidence + Claim -> Label prediction with CoT.

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match | model_reasoning | dataset_explanation

Install dependencies:
    pip install groq

Usage:
    export GROQ_API_KEY="your_groq_api_key_here"
    python evaluate.py --data "train_200.csv"
    python evaluate.py --data "train_200.csv" --samples 20 --random
    python evaluate.py --data "train_200.csv" --samples 0
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given clinical evidence and a claim, determine if the claim is TRUE, FALSE, or PARTIALLY TRUE based solely on the provided evidence.

Think step by step, then respond in this exact JSON format (no other text, no markdown fences):
{
  "reasoning": "<your step-by-step chain of thought>",
  "label": "<true|false|partially true>"
}

Label definitions:
- "true": the claim is fully and accurately supported by the evidence
- "false": the claim contradicts the evidence or lacks any support
- "partially true": the claim is partly correct but contains inaccuracies or important omissions"""

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

def load_model(groq_model: str):
    try:
        from groq import Groq
    except ImportError:
        print("groq is not installed. Run:  pip install groq")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Get a free key at console.groq.com, then run:")
        print('  export GROQ_API_KEY="gsk_..."')
        sys.exit(1)

    print(f"Using Groq API  |  model: {groq_model}")
    client = Groq(api_key=api_key)
    return client, groq_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(model, evidence: str, claim: str, ev_limit: int,
               temperature: float, max_tokens: int) -> dict:
    client, groq_model = model
    user_msg = (
        f"Evidence:\n{evidence[:ev_limit]}\n\n"
        f"Claim:\n{claim}\n\n"
        "Respond only with the JSON object."
    )

    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_text = response.choices[0].message.content.strip()
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
        "--groq-model", default="llama-3.1-8b-instant",
        help="Groq model name (default: llama-3.1-8b-instant)"
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
        "--max-tokens", type=int, default=512,
        help="Max output tokens per inference (default: 512)"
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
        subset = all_data[:n]
        print(f"Using first {n} samples")

    # Load model (Groq API client)
    model = load_model(args.groq_model)

    # Evaluation loop
    results = []
    print(f"Starting evaluation ({n} samples)...\n")

    for i, item in enumerate(subset, start=1):
        print(f"[{i}/{n}] Running inference...")
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
        except Exception as e:
            predicted = "unknown"
            reasoning = f"Error: {e}"

        match = predicted == item["label"]
        results.append({
            "index":       i,
            "evidence":    item["evidence"],
            "claim":       item["claim"],
            "true_label":  item["label"],
            "predicted":   predicted,
            "match":       match,
            "reasoning":   reasoning,
            "explanation": item["explanation"],
        })
        print(f"  true={item['label']}  predicted={predicted}  match={match}")

    # Summary
    print_summary(results)

    # Save CSV
    out_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results(results, out_path)


if __name__ == "__main__":
    main()
