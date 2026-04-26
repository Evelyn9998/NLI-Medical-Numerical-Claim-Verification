"""
LLM Fact-Check Evaluator
Uses Llama-3.1-8B-Instruct, Qwen, or DeepSeek-R1 via local HuggingFace model.
Zero-shot Direct Prompting: Evidence + Claim -> Structured reasoning -> Label via Logical Reasoner.

Benchmark accuracy criteria (MedCalc-Bench §3.1, Table 2):
  - Rule-based calculators (integer scores from summed criteria) : exact match
  - Equation-based calculators (continuous decimal output)       : within 5%
  - Date-based calculators (calendar date output)               : exact match

The model self-classifies the calculator type from the claim and applies
the correct criterion. No Category column is required in the dataset.

On the Python side, _override_step2_by_criteria infers the criterion solely
from the model's calculator_type field and the shape of correct_result:
  - model says "rule-based" or "date-based", OR value is an integer / date  -> exact match
  - model says "equation-based",             OR value is a decimal           -> 5% tolerance

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match | model_reasoning | dataset_explanation

Install dependencies:
    pip install transformers torch

Usage:
    python zero_shot_CoT.py --data "mrt_claim_cleaned.csv" --model "$HOME/models/DeepSeek-R1"
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct"
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Qwen2.5-14B-Instruct" --samples 20 --random
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --batch-size 4
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
# System prompt — optimised for DeepSeek R1 (also compatible with Llama / Qwen)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given clinical evidence and a clinical claim, determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The claim contains two types of sub-claims:
- Extractive Claims: parameter values that can be read directly from the evidence without computation (e.g., patient age, presence of hypertension, troponin level).
- One Implicit Calculation Claim: the final score or derived value, which requires applying a medical formula or rule-based scoring system to evidence parameters.

---
STEP 1 — Check each INPUT entity (every comma-separated segment EXCEPT the last):

HOW TO SPLIT THE CLAIM:
Split the claim on every comma ( , ). Each segment is one entity.
  - Every segment EXCEPT the last → INPUT parameters → go in step1_entities
  - The LAST segment              → final computed result → goes ONLY in step2_calculation

CRITICAL: The last comma-separated segment is ALWAYS the Implicit Calculation Claim.
It must NOT appear in step1_entities. If you include it there, your answer is wrong.

EXAMPLE:
  Claim: "Based on the patient's data where the Calcium is 7.2 mg/dL,
          the Albumin is 3.2 g/dL,
          the Corrected Calcium Concentration is 7.84 mg/dL."

  Split by comma → 3 segments:
    Segment 1: "Based on the patient's data where the Calcium is 7.2 mg/dL"
    Segment 2: "the Albumin is 3.2 g/dL"
    Segment 3: "the Corrected Calcium Concentration is 7.84 mg/dL."  ← LAST = step2 only

  step1_entities:    Calcium (7.2 mg/dL), Albumin (3.2 g/dL)
  step2_calculation: Corrected Calcium Concentration claimed = 7.84 mg/dL

CORRECTNESS RULES for each INPUT entity in step1_entities:
- A positively-framed entity (e.g., "Liver disease criteria for HAS-BLED") describes a condition that must be present in the evidence. If it is not found, mark correct = false.
- A negatively-framed entity (e.g., "Cough Absent") describes the absence of a condition. If the condition is not mentioned in the evidence, the absence is confirmed — mark correct = true.

---
STEP 2 — Evaluate the final computed result (step2_calculation):

Parameter extraction:
The claim may contain intentionally incorrect input values. Always extract all parameter values from the Evidence, not from the claim. Show the full formula name and expression, substitute every evidence-derived value, and show every arithmetic step before recording correct_result.

Accuracy criteria — infer the calculator type from the claim and apply the appropriate rule:

  Rule-based calculator: produces an integer score by summing criteria
  (e.g., HEART Score, CHA2DS2-VASc Score, Wells' Criteria for DVT/PE,
  CURB-65, Glasgow Coma Score, Caprini Score, APACHE II, SOFA Score).
  -> claimed_result must EXACTLY match correct_result (integer comparison).

  Equation-based calculator: produces a continuous decimal value via a formula
  (e.g., GFR, LDL, BMI, MAP, FIB-4 Index, Anion Gap, Serum Osmolality,
  Cockcroft-Gault, Framingham Risk Score).
  -> claimed_result is correct if it is within 5% of correct_result,
    i.e. |claimed - correct| / |correct| <= 0.05.

  Date-based calculator: produces a calendar date
  (e.g., Estimated Due Date, Estimated Date of Conception,
  Estimated Gestational Age).
  -> claimed_result must EXACTLY match correct_result (string comparison).

---
DERIVING THE LABEL — apply this rule exactly, with no exceptions:
  - All step1_entities correct = true  AND step2_calculation correct = true  -> "true"
  - All step1_entities correct = true  AND step2_calculation correct = false -> "partially true"
  - Any  step1_entity   correct = false (regardless of step2)               -> "false"

---
You may reason freely inside <think> tags. After your reasoning, output ONLY the JSON object below — no markdown fences, no preamble, no text after the closing brace.

{
  "step1_entities": [
    {
      "name": "<INPUT parameter name from the 'where' clause — NEVER the calculator name or its final output>",
      "claim_value": "<value stated in the claim>",
      "evidence_value": "<value found in evidence, or 'not found'>",
      "correct": true or false
    }
  ],
  "step2_calculation": {
    "calculator_type": "<rule-based | equation-based | date-based>",
    "formula": "<formula name and expression>",
    "substitution": "<formula with evidence-derived values substituted in>",
    "steps": "<every arithmetic step shown>",
    "correct_result": "<your computed answer, with units if applicable>",
    "claimed_result": "<the final 'the [calculator name] is [value]' from the claim>",
    "correct": true or false
  },
  "step3_label": "<true|partially true|false>"
}"""


# ---------------------------------------------------------------------------
# Benchmark accuracy criteria — mirrors MedCalc-Bench §3.1 Table 2
#
# The criterion is inferred entirely from the model's own output — no external
# metadata column is needed or used.
#
# Priority:
#   1. Model's calculator_type field  ("rule-based" / "date-based" / "equation-based")
#   2. Shape of correct_result:
#        - matches MM/DD/YYYY            -> date-based  -> exact match
#        - numeric with no decimal point -> rule-based  -> exact match
#        - numeric with decimal point    -> equation-based -> 5% tolerance
#        - unparseable                   -> exact match (safe default)
# ---------------------------------------------------------------------------

# Model's self-reported calculator_type values that map to each criterion
_MODEL_EXACT_TYPES     = {'rule-based', 'date-based'}
_MODEL_TOLERANCE_TYPES = {'equation-based'}

# Date pattern used by MedCalc-Bench (MM/DD/YYYY)
_DATE_RE = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')


def _infer_criteria(correct_result_str: str, model_calc_type: str = "") -> str:
    """
    Infer which accuracy criterion to apply.
    Returns one of: "exact" | "tolerance"

    Priority:
      1. Model's self-reported calculator_type field.
      2. Shape of correct_result string.
    """
    # 1. Trust the model's self-classification when present and recognised
    mct = (model_calc_type or "").strip().lower()
    if mct in _MODEL_EXACT_TYPES:
        return "exact"
    if mct in _MODEL_TOLERANCE_TYPES:
        return "tolerance"

    # 2. Fall back to value-shape heuristic
    s = correct_result_str.strip()

    if _DATE_RE.match(s):           # MM/DD/YYYY -> date-based -> exact
        return "exact"

    num_part = re.sub(r'[^\d.\-]', '', s).strip()
    if not num_part:                # unparseable -> safe default
        return "exact"
    if '.' not in num_part:         # integer -> rule-based -> exact
        return "exact"
    return "tolerance"              # decimal -> equation-based -> 5%


def _override_step2_by_criteria(parsed: dict) -> dict:
    """
    Re-evaluate step2_calculation.correct using the benchmark's accuracy
    criteria inferred from the model's own JSON output.
    Mutates parsed in-place and returns it.
    Falls back to the model's own judgment when values cannot be parsed.
    """
    calc    = parsed.get("step2_calculation", {})
    cr_str  = str(calc.get("correct_result",  "")).strip()
    clm_str = str(calc.get("claimed_result", "")).strip()
    mct     = str(calc.get("calculator_type", "")).strip().lower()

    # Cannot validate when values are missing or placeholder text
    if not cr_str or not clm_str or "(unknown)" in cr_str or "(unknown)" in clm_str:
        return parsed

    criterion = _infer_criteria(cr_str, mct)

    if criterion == "exact":
        cr_core  = re.sub(r'[^\d.\-]', '', cr_str).strip()
        clm_core = re.sub(r'[^\d.\-]', '', clm_str).strip()
        if cr_core and clm_core:
            correct = (cr_core == clm_core)
        else:
            # Full-string fallback handles date strings containing slashes
            correct = (cr_str.lower() == clm_str.lower())

    else:  # "tolerance"
        try:
            cr_m  = re.search(r'[\-\d.]+', cr_str)
            clm_m = re.search(r'[\-\d.]+', clm_str)
            if cr_m and clm_m:
                cr_val  = float(cr_m.group())
                clm_val = float(clm_m.group())
                if cr_val == 0:
                    correct = (clm_val == 0)
                else:
                    correct = abs(cr_val - clm_val) / abs(cr_val) <= 0.05
            else:
                return parsed   # no numeric content — leave model's judgment
        except (ValueError, TypeError):
            return parsed

    calc["correct"] = correct
    return parsed


def _remove_implicit_from_step1(parsed: dict, claim: str = "") -> dict:
    """
    Python-side safety net: remove the implicit calculation claim from
    step1_entities before the Logical Reasoner runs.

    Identification strategy — split the claim by commas:
      The claim is a comma-separated list of entities. The LAST segment is
      always the final computed result (step2); every other segment is an
      extractive input parameter (step1).

    Secondary source: step2_calculation.formula field (text before = or :).

    Matching: case-insensitive. An entity is removed when the extracted
    output name equals or is contained in the entity name — never the
    reverse, to avoid false positives (e.g., "Albumin" must not be removed
    just because the output is "Albumin Corrected Delta Ratio").
    """
    entities = parsed.get("step1_entities", [])
    if not entities:
        return parsed

    output_names = set()

    # Primary source — last comma-separated segment of the claim
    if claim:
        parts = [p.strip() for p in claim.split(",")]
        if parts:
            last_seg = parts[-1].rstrip(".")
            m = re.search(r"the\s+(.+?)\s+is\s+", last_seg, re.IGNORECASE)
            if m:
                output_names.add(m.group(1).strip().lower())

    # Secondary source — formula field in step2_calculation
    formula = parsed.get("step2_calculation", {}).get("formula", "")
    if formula:
        formula_name = re.split(r"[=:(]", formula)[0].strip().lower()
        if formula_name:
            output_names.add(formula_name)

    if not output_names:
        return parsed

    def _is_implicit(entity_name):
        n = entity_name.strip().lower()
        # Remove when output name equals entity name, or output name is
        # contained in entity name (handles unit suffixes, e.g. "Anion Gap (mEq/L)").
        # Never remove when entity name is contained in output name — that
        # would cause false positives for shared substrings like "Albumin".
        return any(n == out or out in n for out in output_names)

    filtered = [e for e in entities if not _is_implicit(e.get("name", ""))]

    # Only apply the filter when it actually removes something
    if len(filtered) < len(entities):
        parsed["step1_entities"] = filtered

    return parsed


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
    """
    If any substring of phrase_min_len+ chars repeats threshold+ times
    consecutively, truncate the text just before the repetition starts.
    Handles Qwen / R1 looping issues.
    """
    pattern = re.compile(
        r'(.{' + str(phrase_min_len) + r',}?)\1{' + str(threshold - 1) + r',}',
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        return text[:m.start()].strip()
    return text


# ---------------------------------------------------------------------------
# Logical Reasoner
# ---------------------------------------------------------------------------
#
#  Label definitions (ground truth):
#
#  TRUE          : ALL step1 entities correct=true  AND step2 correct=true
#  PARTIALLY TRUE: ALL step1 entities correct=true  AND step2 correct=false
#  FALSE         : ANY step1 entity  correct=false  (step2 value irrelevant)
#
#  Note: step2.correct is re-evaluated by _override_step2_by_criteria before
#  this function runs, so the benchmark accuracy criteria are already applied.
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

def load_model(model_name: str, batch_size: int) -> dict:
    """
    Returns {"pipeline": pipe, "model_name": model_name}.
    batch_size is baked into the pipeline so the GPU stays fed continuously.
    low_cpu_mem_usage=True loads weights directly to GPU layer-by-layer,
    cutting model load time by ~30-50% vs. the default full-CPU-then-move path.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Padding token required for batched inference — use eos_token if absent
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                low_cpu_mem_usage=True,
            )
            print("  4-bit quantization enabled (bitsandbytes)")
        except Exception as e:
            print(f"  bitsandbytes not available ({e}), loading in fp16")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
    else:
        print("  No GPU, loading on CPU (fp32)")
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
    print(f"Model ready: {model_name}  (batch_size={batch_size})")
    return {"pipeline": pipe, "model_name": model_name}


# ---------------------------------------------------------------------------
# Single-response post-processing (shared between batch and retry paths)
# ---------------------------------------------------------------------------

def _parse_response(raw_text: str, claim: str = "") -> dict:
    """
    Convert raw model text into {"reasoning": str, "label": str}.

    Processing order:
      1. Truncate runaway repetitions (Qwen / R1 looping guard).
      2. Strip <think>...</think> blocks emitted by reasoning models (R1).
      3. Strip markdown fences if present.
      4. Parse JSON.
      5. _remove_implicit_from_step1: strip the calculator output from step1
         before it can corrupt the Logical Reasoner.
      6. _override_step2_by_criteria: enforce benchmark accuracy tolerance.
      7. _logical_reasoner: derive the final label.
      8. If JSON fails, fall back to _keyword_label on the cleaned text.
    """
    raw_text = _truncate_repetition(raw_text)

    # ── Strip reasoning blocks ─────────────────────────────────────────────
    # Case 1: model wraps reasoning in proper <think>...</think> tags
    raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    # Case 2: model emits reasoning WITHOUT an opening <think> tag, ending
    # with a bare </think> before the JSON (observed in DeepSeek-R1 outputs).
    # Strip everything up to and including the orphaned </think>.
    raw_text = re.sub(r"^[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()

    clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    # ── JSON extraction ────────────────────────────────────────────────────
    parsed = None

    # Attempt 1: direct parse (clean text is pure JSON)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Attempt 2: model prefixed JSON with free text — find the object boundary
    if parsed is None:
        for m in re.finditer(r"\{", clean):
            candidate = clean[m.start():]
            last = candidate.rfind("}")
            if last == -1:
                continue
            try:
                parsed = json.loads(candidate[:last + 1])
                break
            except json.JSONDecodeError:
                continue

    # ── Structured path ────────────────────────────────────────────────────
    if parsed is not None:
        # Remove any implicit calculation claim that leaked into step1_entities
        parsed = _remove_implicit_from_step1(parsed, claim)

        # Apply benchmark accuracy criteria using only the model's own output
        parsed = _override_step2_by_criteria(parsed)

        label = _logical_reasoner(parsed)

        # Safety: if step1_entities is empty after cleanup, use the model's
        # own step3_label rather than silently defaulting to "false".
        if not parsed.get("step1_entities"):
            fallback = normalize_label(parsed.get("step3_label", ""))
            if fallback in ("true", "partially true", "false"):
                label = fallback

        reasoning_out = json.dumps(
            {
                "entities":      parsed.get("step1_entities", []),
                "calculation":   parsed.get("step2_calculation", {}),
                "derived_label": label,
            },
            ensure_ascii=False,
            indent=None,
        )
        return {"reasoning": reasoning_out, "label": label}

    # ── Fallback: no JSON found anywhere ──────────────────────────────────
    return {"reasoning": clean, "label": _keyword_label(clean)}


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

def _build_messages(item: dict, ev_limit: int) -> list:
    """Build the chat message list for one sample."""
    user_msg = (
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        "Respond only with the JSON object."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def run_batch(model: dict, batch: list, ev_limit: int,
              temperature: float, max_tokens: int,
              repetition_penalty: float) -> list:
    """
    Run inference on a list of items in one pipeline call.
    Returns a list of {"reasoning": str, "label": str} dicts, one per item.

    Note on repetition_penalty with temperature = 0:
      When do_sample=False (greedy decoding), the pipeline ignores
      repetition_penalty regardless of the argument value, so the default
      of 1.0 has no practical effect in that mode. The argument only
      matters when temperature > 0.
    """
    pipe      = model["pipeline"]
    do_sample = temperature > 0.0

    all_messages = [_build_messages(item, ev_limit) for item in batch]

    results = pipe(
        all_messages,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty if do_sample else 1.0,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )

    # pipeline returns List[List[dict]] when given a list of inputs
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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help=(
                            "Sampling temperature (default: 0.0 for greedy/deterministic "
                            "decoding). With temperature=0, repetition_penalty has no effect."
                        ))
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help=(
                            "Repetition penalty applied only when temperature > 0 "
                            "(default: 1.0). 1.0 (no penalty) is recommended for "
                            "DeepSeek-R1 and for greedy decoding where this argument "
                            "has no effect."
                        ))
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help=(
                            "Max output tokens per inference (default: 4096). "
                            "Set higher for DeepSeek-R1 whose <think> blocks "
                            "typically run 2000-8000 tokens."
                        ))
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Samples per GPU batch (default: 4; reduce if OOM)")
    parser.add_argument("--start", type=int, default=1,
                        help="1-based index to start from (default: 1)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between batches (default: 0.0)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: auto-generated with timestamp)")
    args = parser.parse_args()

    if args.random and args.start != 1:
        print(f"Warning: --start {args.start} is ignored when --random is set.")

    # -- Load data ---------------------------------------------------------
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples")
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

        # Attempt the whole batch; fall back to per-sample on failure
        try:
            resps = run_batch(
                model, batch, args.ev_limit,
                args.temperature, args.max_tokens, args.repetition_penalty,
            )
        except Exception as e:
            print(f"  Batch error ({e}), falling back to one-by-one...")
            resps = []
            for item in batch:
                for attempt in range(3):
                    try:
                        msgs      = _build_messages(item, args.ev_limit)
                        pipe      = model["pipeline"]
                        do_sample = args.temperature > 0.0
                        raw = pipe(
                            msgs,
                            max_new_tokens=args.max_tokens,
                            temperature=args.temperature if do_sample else None,
                            do_sample=do_sample,
                            repetition_penalty=(
                                args.repetition_penalty if do_sample else 1.0
                            ),
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