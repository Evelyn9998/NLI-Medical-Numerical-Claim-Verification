"""
LLM-as-Judge Error Classifier — Few-shot with OpenAI API
=========================================================
Uses 12 human-validated few-shot examples to classify error types
for all remaining incorrect predictions.

Usage:
    pip install openai

    # Validation set quick test
    python error_analysis_fewshot_api.py --data ../results/zs_cot/zs_cot_llama_full.csv --samples 9

    # Run with GPT-4o
    python error_analysis_fewshot_api.py --data ../results/zs_cot/zs_cot_llama_full.csv --model gpt-4o

    # Run with GPT-4o-mini
    python error_analysis_fewshot_api.py --data ../results/zs_cot/zs_cot_llama_full.csv --model gpt-4o-mini

    # Full run, skip held-out validation set (indices: 18,52,95,50,35,53,74,86,9)
    python error_analysis_fewshot_api.py --data ../results/zs_cot/zs_cot_llama_full.csv --model gpt-4o --skip-validation
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

import openai

from few_shot_examples import FEW_SHOT_EXAMPLES

# ── Held-out validation indices (DO NOT include in few-shot) ──────────────────
VALIDATION_INDICES = {
    9, 18, 26, 27, 29, 30, 33, 35, 44, 50, 52, 53, 61, 62, 66, 68, 69,
    74, 75, 79, 83, 86, 89, 95, 98, 113, 116, 117, 118, 119, 120, 121,
    123, 131, 132, 134, 137, 141,
}

# ── Error type keys (JSON output order) ───────────────────────────────────────
VALID_TYPES = [
    "failed_parameter_extraction",
    "verification_logic_error",
    "incorrect_formula_criteria",
    "computation_error",
    "omitted_calculation_process_or_result",
    "other_error",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert auditor of medical claim-verification systems.

Your task: identify ALL error types present in an LLM response that produced an INCORRECT label
when verifying a medical calculation claim. A single response may contain multiple errors.

=== LABELLING RULES (dataset ground truth) ===
- "true"          : ALL input entities (all except the last entity) are correct AND
                    the final calculation result is correct.
- "partially true": ALL input entities (all except the last) are correct, BUT the final
                    calculation result is incorrect.
- "false"         : One or more input entities are incorrect
                    (regardless of whether the final result is correct).

=== SIX ERROR TYPES ===
1. failed_parameter_extraction
   A required input variable is missing, extracted with the wrong value, wrong time point,
   or wrong unit. Also applies when the LLM mishandles absent variables (variables not mentioned
   in the evidence should be assumed false/0 per dataset convention).

2. verification_logic_error
   The LLM fails to apply the correct label boundary. E.g. labels a claim "false" when all
   input entities are correct (should be "partially true"), or labels "partially true" when
   input entities are wrong (should be "false").

3. incorrect_formula_criteria
   Wrong formula/scoring system selected, or correct formula used with wrong coefficients,
   wrong operators, hallucinated/omitted terms, or wrong scoring criteria.

4. computation_error
   Arithmetic errors in the computation process — calculation mistakes,
   wrong order of operations, or rounding errors outside the 5% tolerance.
   May co-occur with incorrect_formula_criteria when the LLM additionally
   uses the wrong formula.

5. omitted_calculation_process_or_result
   The LLM skips required intermediate steps, does not apply the formula, or accepts
   the claim value without performing the calculation.

6. other_error
   Any error that does not fit the five categories above.

=== RULES ===
- Check each type independently. Multiple types may be set to 1 simultaneously.
- Do NOT apply a priority rule.
- Set other_error to 0 unless truly uncategorisable.
- At least one flag must be 1.

=== OUTPUT FORMAT ===
Respond with ONLY valid JSON — no markdown, no preamble:
{
  "failed_parameter_extraction": <0 or 1>,
  "verification_logic_error": <0 or 1>,
  "incorrect_formula_criteria": <0 or 1>,
  "computation_error": <0 or 1>,
  "omitted_calculation_process_or_result": <0 or 1>,
  "other_error": <0 or 1>,
  "explanation": "<2-4 sentences describing each error type that is present>"
}"""


def build_few_shot_block(examples: list, ev_limit: int = 600,
                          reasoning_limit: int = 500) -> str:
    lines = ["=== FEW-SHOT EXAMPLES ===\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Example {i} (primary type: {ex['primary_type']}) ---")
        lines.append(f"TRUE LABEL : {ex['true_label']}")
        lines.append(f"PREDICTED  : {ex['predicted_label']}")
        lines.append(f"CLAIM:\n{ex['claim']}")
        lines.append(f"\nEVIDENCE (truncated):\n{ex['evidence'][:ev_limit]}")
        lines.append(f"\nLLM REASONING (truncated):\n{ex['model_reasoning'][:reasoning_limit]}")
        lines.append(f"\nCORRECT ANNOTATION:\n{json.dumps(ex['annotation'], indent=2)}")
        lines.append("")
    lines.append("=== NOW ANNOTATE THE FOLLOWING ===\n")
    return "\n".join(lines)


def build_user_message(row: dict, few_shot_block: str,
                        ev_limit: int = 1000, reasoning_limit: int = 700) -> str:
    return (
        few_shot_block
        + f"TRUE LABEL : {row['true_label']}\n"
        + f"PREDICTED  : {row['predicted_label']}\n\n"
        + f"CLAIM:\n{row['claim']}\n\n"
        + f"EVIDENCE (truncated):\n{row['evidence'][:ev_limit]}\n\n"
        + f"LLM REASONING (truncated):\n{row['model_reasoning'][:reasoning_limit]}\n\n"
        + "Identify ALL error types present. Respond only with the JSON object."
    )


def parse_response(raw: str) -> dict:
    clean = re.sub(r"<\|[^|]*\|>", "", raw)
    clean = re.sub(r"```(?:json)?|```", "", clean).strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]
    last = clean.rfind("}")
    if last != -1:
        clean = clean[:last + 1]

    default = {t: 0 for t in VALID_TYPES}
    default["explanation"] = ""

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        result = dict(default)
        for slug in VALID_TYPES:
            if re.search(rf'"{slug}"\s*:\s*1', clean):
                result[slug] = 1
        ex = re.search(r'"explanation"\s*:\s*"([\s\S]+?)"(?:\s*[,}])', clean)
        result["explanation"] = ex.group(1).strip() if ex else raw[:300]
        if not any(result[t] for t in VALID_TYPES):
            result["other_error"] = 1
        return result

    result = {}
    for slug in VALID_TYPES:
        val = parsed.get(slug, 0)
        result[slug] = 1 if str(val).strip() in ("1", "true", "True") else 0
    result["explanation"] = str(parsed.get("explanation", "")).strip()
    if not any(result[t] for t in VALID_TYPES):
        result["other_error"] = 1
    return result


def normalize_label(label: str) -> str:
    label = (label or "").lower().strip()
    if label == "true":    return "true"
    if label == "false":   return "false"
    if "partial" in label: return "partially true"
    return label


def load_errors(csv_path: str, skip_indices: set) -> list:
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tl = normalize_label(row.get("true_label", ""))
            pl = normalize_label(row.get("predicted_label", ""))
            if tl == pl:
                continue
            idx = row.get("index", "")
            try:
                if int(float(idx)) in skip_indices:
                    continue
            except:
                pass
            rows.append({
                "index":           idx,
                "evidence":        row.get("evidence", ""),
                "claim":           row.get("claim", ""),
                "true_label":      tl,
                "predicted_label": pl,
                "model_reasoning": row.get("model_reasoning", ""),
            })
    return rows


def call_api(client, model: str, user_content: str) -> str:
    msg = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )
    return msg.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",            default="validation_set.csv")
    parser.add_argument("--api-key",         default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model",           default="gpt-4o")
    parser.add_argument("--samples",          type=int, default=0)
    parser.add_argument("--skip-validation",  action="store_true",
                        help="Exclude held-out validation indices from processing")
    parser.add_argument("--validation-only",  action="store_true",
                        help="Run ONLY the 9 held-out validation indices")
    parser.add_argument("--output",           default="")
    args = parser.parse_args()

    if not args.api_key:
        print("Set OPENAI_API_KEY or pass --api-key")
        sys.exit(1)

    client = openai.OpenAI(api_key=args.api_key)
    print(f"Model: {args.model}")

    few_shot_indices = {ex["index"] for ex in FEW_SHOT_EXAMPLES}

    if args.validation_only:
        all_errors = load_errors(args.data, skip_indices=few_shot_indices)
        subset = [r for r in all_errors if int(float(r["index"])) in VALIDATION_INDICES]
    else:
        skip = VALIDATION_INDICES if args.skip_validation else set()
        skip = skip | few_shot_indices
        all_errors = load_errors(args.data, skip_indices=skip)
        subset = all_errors if args.samples == 0 else all_errors[:args.samples]
    n = len(subset)
    print(f"Loaded {n} samples  |  few-shot: {len(FEW_SHOT_EXAMPLES)}")

    few_shot_block = build_few_shot_block(FEW_SHOT_EXAMPLES)

    out_path = args.output or f"error_fewshot_{args.model.replace('/', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["index", "true_label", "predicted_label"] + VALID_TYPES + [
        "explanation", "evidence_snippet", "claim", "model_reasoning_snippet"]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing to: {out_path}\n")

    results = []
    t_start = time.time()

    for i, row in enumerate(subset, 1):
        print(f"[{i}/{n}] true={row['true_label']:<16} pred={row['predicted_label']}")

        res = {t: 0 for t in VALID_TYPES}
        res["explanation"] = "inference error"
        res["other_error"] = 1

        for attempt in range(3):
            try:
                raw = call_api(client, args.model,
                               build_user_message(row, few_shot_block))
                res = parse_response(raw)
                break
            except Exception as e:
                print(f"  [attempt {attempt+1}/3] {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    res["explanation"] = f"Failed after 3 attempts: {e}"

        active = [t for t in VALID_TYPES if res.get(t)]
        print(f"  -> {active or ['other_error']}")
        results.append({**row, **res})

        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "index":               row["index"],
                "true_label":          row["true_label"],
                "predicted_label":     row["predicted_label"],
                **{t: res[t] for t in VALID_TYPES},
                "explanation":         res["explanation"],
                "evidence_snippet":    row["evidence"][:200].replace("\n", " "),
                "claim":               row["claim"],
                "model_reasoning_snippet": row["model_reasoning"][:200].replace("\n", " "),
            })

        elapsed = time.time() - t_start
        rate    = i / elapsed
        eta     = (n - i) / rate if rate > 0 else 0
        print(f"  ({elapsed:.0f}s  |  ~{eta:.0f}s left  |  {rate:.2f} samples/s)")

    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
