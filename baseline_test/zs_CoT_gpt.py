"""
LLM Fact-Check Evaluator — OpenAI GPT version
Uses OpenAI API (gpt-4o-mini by default) for zero-shot CoT fact-checking.
Structured JSON output with three explicit steps + Logical Reasoner.

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match | model_reasoning | dataset_explanation

Install dependencies:
    pip install openai

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline_test/zs_CoT_gpt.py --data mrt_claim_full.csv --model gpt-4o-mini --samples 2
    python baseline_test/zs_CoT_gpt.py --data mrt_claim_full.csv --samples 20 --random
    python baseline_test/zs_CoT_gpt.py --data mrt_claim_full.csv --samples 0
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
# System prompt — structured JSON output with three explicit steps
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given clinical evidence and a clinical claim that involves numerical entities and a medical calculation, determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The claim contains a list of entities (input values extracted from the patient record) followed by a final computed result (the last entity). Work through the following steps carefully.

---

STEP 1 — Check each INPUT entity (every entity in the claim EXCEPT the very last one):
CRITICAL: The last entity in the claim is ALWAYS the final computed result. It must NOT appear in step1_entities — it is evaluated only in step2. If you include it in step1_entities, your answer is wrong.
For every input entity (all except the last) mentioned in the claim:
a) Look for that entity in the evidence.
   - If the entity is NOT present in the evidence AND the claim states its value as "none" or "no", treat it as correct (correct: true).
   - If the entity is NOT present in the evidence AND the claim states a specific numeric value, treat it as incorrect (correct: false).
b) If the entity IS present in the evidence, compare the claim value with the evidence value and set correct accordingly.

STEP 2 — Perform the medical calculation step by step:
a) State the formula used.
b) Substitute each input value FROM THE EVIDENCE (not from the claim) into the formula.
c) Show every arithmetic step and arrive at the correct computed result.
d) Compare the correct computed result with the value stated in the claim's last entity.
e) Set "correct": true if the result matches (rule-based: exact match; equation-based: within 5%; date-based: exact match), otherwise false.

STEP 3 — Assign the label using these rules:
- "true"          : ALL step1 entities have correct=true  AND  step2 calculation correct=true
- "partially true": ALL step1 entities have correct=true  AND  step2 calculation correct=false
- "false"         : ANY step1 entity has correct=false (regardless of calculation)

---

IMPORTANT RULES:
- An entity not found in the evidence is incorrect (correct: false) UNLESS the claim explicitly states its value as "none" or "no".
- Always use EVIDENCE values (not claim input values) when performing your calculation in Step 2.
- Show the full formula and every arithmetic step before deciding step2 correct.
- step1_entities must contain ONLY the input entities (all entities except the last one). Never include the final computed result in step1_entities.

Respond ONLY with this exact JSON object (no other text, no markdown fences):
{
  "step1_entities": [
    {
      "name": "<entity name — input entities only, NOT the last entity>",
      "claim_value": "<value stated in claim>",
      "evidence_value": "<value found in evidence, or 'not found'>",
      "correct": true or false
    }
  ],
  "step2_calculation": {
    "formula": "<formula name and expression>",
    "substitution": "<formula with evidence values plugged in>",
    "steps": "<every arithmetic step shown>",
    "correct_result": "<computed answer>",
    "claimed_result": "<last entity value from claim>",
    "correct": true or false
  },
  "step3_label": "<true|partially true|false>"
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
# Logical Reasoner
# ---------------------------------------------------------------------------
#
#  FACT_1 = all step1_entities are correct
#  FACT_2 = step2_calculation is correct
#
#  FACT_1=True  AND FACT_2=True   →  "true"
#  FACT_1=True  AND FACT_2=False  →  "partially true"
#  FACT_1=False (any)             →  "false"   (FACT_2 irrelevant)
#
# ---------------------------------------------------------------------------

def _logical_reasoner(parsed: dict) -> str:
    entities = parsed.get("step1_entities", [])
    calc     = parsed.get("step2_calculation", {})

    fact1 = bool(entities) and all(
        bool(e.get("correct", False)) for e in entities
    )
    fact2 = bool(calc.get("correct", False))

    if fact1 and fact2:
        return "true"
    elif fact1 and not fact2:
        return "partially true"
    else:
        return "false"


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

def load_model(args):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()
    print(f"OpenAI client initialised. Model: {args.model}")
    return (client, args.model)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_model(model, evidence: str, claim: str, ev_limit: int,
               temperature: float, max_tokens: int) -> dict:
    client, model_name = model

    user_msg = (
        f"Evidence:\n{evidence[:ev_limit]}\n\n"
        f"Claim:\n{claim}\n\n"
        "Respond only with the JSON object."
    )

    from openai import RateLimitError

    last_exc = None
    raw_text = ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content.strip()
            break
        except RateLimitError as e:
            last_exc = e
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            print(f"  [attempt {attempt+1}/3] RateLimitError — waiting {wait}s...")
            time.sleep(wait)
    else:
        raise last_exc

    clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    # ── Primary path: structured JSON → logical reasoner ──────────────────
    try:
        parsed = json.loads(clean)
        label  = _logical_reasoner(parsed)

        # If step1_entities was empty (model skipped it), fall back to
        # model's own step3_label rather than always returning "false"
        if not parsed.get("step1_entities"):
            fallback = normalize_label(parsed.get("step3_label", ""))
            if fallback in ("true", "partially true", "false"):
                label = fallback

        reasoning_out = json.dumps(
            {
                "entities":     parsed.get("step1_entities", []),
                "calculation":  parsed.get("step2_calculation", {}),
                "derived_label": label,
            },
            ensure_ascii=False,
            indent=None,
        )
        return {"reasoning": reasoning_out, "label": label}

    # ── Fallback path: JSON parse failed — scan text for keywords ─────────
    except json.JSONDecodeError:
        return {
            "reasoning": clean,
            "label":     _keyword_label(clean),
        }


def _keyword_label(text: str) -> str:
    """Last-resort label extraction from free text. Defaults to 'false'."""
    t = text.lower()
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    if '"true"' in t or re.search(r'\blabel\b.*\btrue\b', t):
        return "true"
    if '"false"' in t or re.search(r'\blabel\b.*\bfalse\b', t):
        return "false"
    return "false"


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
        description="OpenAI GPT evaluator — Zero-shot Fact-Check with Structured CoT + Logical Reasoner"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to CSV dataset, e.g. mrt_claim_full.csv"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)"
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
        subset = (
            all_data[start_idx:start_idx + n]
            if args.samples != 0
            else all_data[start_idx:]
        )
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # Load model (OpenAI client)
    model = load_model(args)

    # Prepare output file (write header immediately so results stream live)
    out_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "model_reasoning", "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # Evaluation loop
    results = []
    total   = len(subset)
    print(f"Starting evaluation ({total} samples)...\n")

    for i, item in enumerate(subset, start=args.start):
        print(f"[{i}/{args.start + total - 1}] Running inference...")
        predicted = "false"
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
                if predicted in ("true", "partially true", "false"):
                    break
            except Exception as e:
                print(f"  [attempt {attempt+1}/3] Error: {e}")
                if attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    reasoning = f"Error: {e}"
                    predicted = "false"

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

        # Append to CSV immediately (live streaming)
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
        if i < args.start + total - 1:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
