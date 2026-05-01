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
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Qwen2.5-14B-Instruct" --no-think
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0
    python zero_shot_CoT.py --data "train_300.csv"          --model "$HOME/models/Llama-3.1-8B-Instruct" --seed 42
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

Work through STEP 1 → STEP 2 → STEP 3 in order, then write the JSON.

---
STEP 1 — Check each INPUT entity (every entity in the claim EXCEPT the very last one):

CRITICAL: The last comma-separated entity in the claim is ALWAYS the final computed result.
It must NOT appear in step1_entities — it is evaluated only in step2_calculation.
If you include it in step1_entities, your answer is wrong.

HOW TO IDENTIFY INPUT ENTITIES:
Split the claim on every comma ( , ). Each segment is one entity.
  - Every segment EXCEPT the last → INPUT parameters → go in step1_entities
  - The LAST segment              → final computed result → goes ONLY in step2_calculation

EXAMPLE:
  Claim: "...the Calcium is 7.2 mg/dL, the Albumin is 3.2 g/dL, the Corrected Calcium Concentration is 7.84 mg/dL."
  Segments: Calcium | Albumin | Corrected Calcium Concentration  ← last = step2 only
  step1_entities:    Calcium (7.2 mg/dL), Albumin (3.2 g/dL)
  step2_calculation: Corrected Calcium Concentration = 7.84 mg/dL

For every INPUT entity (all except the last) mentioned in the claim, check the evidence and set:
- correct = true  if the claim value matches the evidence value.
- correct = false if the claim value does not match, or if a required condition is absent.
Special cases:
  - A positively-framed entity (e.g., "Liver disease criteria for HAS-BLED") must be present in the evidence. If not found → correct = false.
  - A negatively-framed entity (e.g., "Cough Absent") describes absence. If the condition is not mentioned in the evidence → correct = true.

---
STEP 2 — Perform the medical calculation step by step:
a) State the formula name and expression.
b) Substitute each input value FROM THE EVIDENCE (not from the claim) — the claim may contain intentionally incorrect values.
c) Show every arithmetic step and arrive at the correct computed result.
d) Compare the correct computed result with the value stated in the claim's last entity.
e) Set "correct": true if the result matches, using the appropriate criterion:
   - Rule-based calculator (integer score, e.g. HEART Score, CHA2DS2-VASc, Wells' Criteria, CURB-65, Glasgow Coma Score, Caprini Score, APACHE II, SOFA Score): exact match.
   - Equation-based calculator (decimal output, e.g. GFR, LDL, BMI, MAP, FIB-4, Anion Gap, Serum Osmolality, Cockcroft-Gault, Framingham Risk): within 5%, i.e. |claimed - correct| / |correct| <= 0.05.
   - Date-based calculator (calendar date, e.g. Estimated Due Date, Estimated Date of Conception, Estimated Gestational Age): exact match.

---
STEP 3 — Assign the label:
- "true"          : ALL step1 entities have correct=true  AND  step2 calculation correct=true
- "partially true": ALL step1 entities have correct=true  AND  step2 calculation correct=false
- "false"         : ANY step1 entity has correct=false (regardless of step2 result)

---
Now write the JSON. Rules:
- Field names must match EXACTLY as shown (step1_entities, step2_calculation, step3_label).
- Always include all three fields, even if step1_entities has only one item.
- No markdown fences, no preamble, no text after the closing brace.

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
# ---------------------------------------------------------------------------

_MODEL_EXACT_TYPES     = {'rule-based', 'date-based'}
_MODEL_TOLERANCE_TYPES = {'equation-based'}

_DATE_RE = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')


def _infer_criteria(correct_result_str: str, model_calc_type: str = "") -> str:
    mct = (model_calc_type or "").strip().lower()
    if mct in _MODEL_EXACT_TYPES:
        return "exact"
    if mct in _MODEL_TOLERANCE_TYPES:
        return "tolerance"

    s = correct_result_str.strip()

    if _DATE_RE.match(s):
        return "exact"

    num_part = re.sub(r'[^\d.\-]', '', s).strip()
    if not num_part:
        return "exact"
    if '.' not in num_part:
        return "exact"
    return "tolerance"


def _override_step2_by_criteria(parsed: dict) -> dict:
    calc    = parsed.get("step2_calculation", {})
    cr_str  = str(calc.get("correct_result",  "")).strip()
    clm_str = str(calc.get("claimed_result", "")).strip()
    mct     = str(calc.get("calculator_type", "")).strip().lower()

    if not cr_str or not clm_str or "(unknown)" in cr_str or "(unknown)" in clm_str:
        return parsed

    criterion = _infer_criteria(cr_str, mct)

    if criterion == "exact":
        cr_core  = re.sub(r'[^\d.\-]', '', cr_str).strip()
        clm_core = re.sub(r'[^\d.\-]', '', clm_str).strip()
        if cr_core and clm_core:
            correct = (cr_core == clm_core)
        else:
            correct = (cr_str.lower() == clm_str.lower())

    else:
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
                return parsed
        except (ValueError, TypeError):
            return parsed

    calc["correct"] = correct
    return parsed


def _remove_implicit_from_step1(parsed: dict, claim: str = "") -> dict:
    entities = parsed.get("step1_entities", [])
    if not entities:
        return parsed

    output_names = set()

    if claim:
        parts = [p.strip() for p in claim.split(",")]
        if parts:
            last_seg = parts[-1].rstrip(".")
            m = re.search(r"the\s+(.+?)\s+is\s+", last_seg, re.IGNORECASE)
            if m:
                output_names.add(m.group(1).strip().lower())

    formula = parsed.get("step2_calculation", {}).get("formula", "")
    if formula:
        formula_name = re.split(r"[=:(]", formula)[0].strip().lower()
        if formula_name:
            output_names.add(formula_name)

    if not output_names:
        return parsed

    def _is_implicit(entity_name):
        n = entity_name.strip().lower()
        return any(n == out or out in n for out in output_names)

    filtered = [e for e in entities if not _is_implicit(e.get("name", ""))]

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
# [Change 4] Model loading — always fp16, no 4-bit quantization, no CPU fp32
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> dict:
    # batch_size is fixed at 1 (Change 1) — not passed into the pipeline here;
    # pipeline batch_size is set at call time via run_batch.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_gpu = torch.cuda.is_available()
    print(f"  GPU available: {has_gpu}")

    # [Change 4] Always fp16 — no 4-bit quantization, no CPU fp32 fallback.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto" if has_gpu else "cpu",
        low_cpu_mem_usage=True,
    )

    # Clear the default max_length=20 so it doesn't conflict with max_new_tokens
    # at inference time (eliminates "Both max_new_tokens and max_length" warning).
    if hasattr(model, "generation_config") and model.generation_config.max_length == 20:
        model.generation_config.max_length = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,   # [Change 1] always 1
    )
    print(f"Model ready: {model_name}  (fp16, batch_size=1)")
    return {"pipeline": pipe, "model_name": model_name, "pad_token_id": tokenizer.pad_token_id}


# ---------------------------------------------------------------------------
# Single-response post-processing (shared between batch and retry paths)
# ---------------------------------------------------------------------------

def _parse_response(raw_text: str, claim: str = "") -> dict:
    """
    Convert raw model text into {"reasoning": str, "label": str}.

    Processing order:
      1. Truncate runaway repetitions (Qwen / R1 looping guard).
      2. Strip <think> blocks in all three forms:
           a. Properly closed <think>...</think>
           b. Orphaned </think> with no opening tag (seen in DeepSeek-R1)
           c. Unclosed <think> with no closing tag — the model ran out of
              tokens mid-thinking. Strip from <think> to end of string.
      3. Strip markdown fences if present.
      4. Parse JSON.
      5. _remove_implicit_from_step1: strip the calculator output from step1.
      6. _override_step2_by_criteria: enforce benchmark accuracy tolerance.
      7. _logical_reasoner: derive the final label.
      8. If JSON fails, fall back to _keyword_label on the cleaned text.
    """
    raw_text = _truncate_repetition(raw_text)

    # ── Strip reasoning blocks ─────────────────────────────────────────────
    # Case a: properly closed <think>...</think>
    raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    # Case b: orphaned </think> with no opening tag (DeepSeek-R1)
    raw_text = re.sub(r"^[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    # Case c: unclosed <think> — model ran out of tokens before </think>.
    #         Strip everything from <think> to end of string.
    if re.search(r"<think>", raw_text, re.IGNORECASE):
        raw_text = re.sub(r"<think>[\s\S]*$", "", raw_text, flags=re.IGNORECASE).strip()

    clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    # ── JSON extraction ────────────────────────────────────────────────────
    parsed = None

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        pass

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
        if not parsed.get("step1_entities"):
            for alt in ("entities", "step1", "input_entities", "extractive_claims"):
                if isinstance(parsed.get(alt), list) and parsed[alt]:
                    parsed["step1_entities"] = parsed[alt]
                    break

        if not parsed.get("step2_calculation"):
            for alt in ("calculation", "step2", "implicit_calculation", "step2_calc"):
                if isinstance(parsed.get(alt), dict) and parsed[alt]:
                    parsed["step2_calculation"] = parsed[alt]
                    break

        if not parsed.get("step1_entities") and not parsed.get("step2_calculation"):
            label = normalize_label(parsed.get("step3_label", ""))
            if label not in ("true", "partially true", "false"):
                label = _keyword_label(clean)
            return {"reasoning": clean, "label": label}

        parsed = _remove_implicit_from_step1(parsed, claim)
        parsed = _override_step2_by_criteria(parsed)

        label = _logical_reasoner(parsed)

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
# [Change 1 + 2] Batched inference — batch_size=1, do_sample=False, temp=0
# ---------------------------------------------------------------------------

def _build_messages(item: dict, ev_limit: int, no_think: bool = False) -> list:
    """
    Build the chat message list for one sample.

    no_think: when True, prepends '/no_think' to the user message.
              This instructs Qwen3 thinking models to skip the <think> block
              entirely and output the JSON directly, saving ~1500+ tokens.
              Has no effect on Llama or other non-thinking models.
    """
    user_msg = (
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        "Respond only with the JSON object."
    )
    if no_think:
        user_msg = "/no_think\n" + user_msg

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


def run_batch(model: dict, batch: list, ev_limit: int,
              max_tokens: int,
              no_think: bool = False) -> list:
    """
    Run inference on a list of items one sample at a time (batch_size=1).

    [Change 1] batch_size is always 1 — the pipeline is called per sample.
    [Change 2] Greedy decoding is pinned unconditionally:
               do_sample=False, temperature=0.  The temperature and
               repetition_penalty arguments have been removed from this
               function's signature so callers can't accidentally re-enable
               sampling.
    """
    from transformers import GenerationConfig
    pipe = model["pipeline"]
    # Build a GenerationConfig object once — greedy, no temperature/top_p/top_k.
    # Using GenerationConfig (rather than individual kwargs) avoids the deprecated
    # "Passing generation_config together with generation-related arguments" warning.
    # temperature is intentionally omitted: it is invalid (and warns) when
    # do_sample=False.
    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=model.get("pad_token_id") or pipe.tokenizer.pad_token_id,
    )

    results = []
    for item in batch:
        messages = _build_messages(item, ev_limit, no_think=no_think)
        out = pipe(
            messages,
            generation_config=gen_cfg,
            return_full_text=False,
        )
        raw = out[0]["generated_text"].strip()
        results.append(_parse_response(raw, item.get("claim", "")))

    return results


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
    # [Change 3] --seed argument
    parser.add_argument("--seed", type=int, default=42,
                        help=(
                            "Global RNG seed for Python random, NumPy, and PyTorch "
                            "(default: 42). Governs --random sample selection and any "
                            "internal stochastic ops. Set to a fixed value for "
                            "fully reproducible runs."
                        ))
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help=(
                            "Max output tokens per inference (default: 4096). "
                            "Llama outputs JSON directly without a reasoning block, "
                            "so 2048 is sufficient. For Qwen without --no-think, "
                            "increase to 8192 to fit the <think> block + JSON."
                        ))
    parser.add_argument("--start", type=int, default=1,
                        help="1-based index to start from (default: 1)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between samples (default: 0.0)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: auto-generated with timestamp)")
    parser.add_argument("--no-think", action="store_true",
                        help=(
                            "Prepend '/no_think' to user messages. "
                            "Disables the <think> reasoning block for Qwen3 thinking "
                            "models, so the full JSON fits within --max-tokens. "
                            "Has no effect on Llama or other non-thinking models."
                        ))
    args = parser.parse_args()

    if args.random and args.start != 1:
        print(f"Warning: --start {args.start} is ignored when --random is set.")

    if args.no_think:
        print("--no-think enabled: prepending '/no_think' to all user messages.")

    # [Change 3] Lock down all randomness immediately after arg parsing.
    _seed = args.seed
    random.seed(_seed)
    try:
        import numpy as np
        np.random.seed(_seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        # Deterministic cuDNN ops where possible (may slow some kernels).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
    print(f"Global seed set to {_seed}")

    # -- Load data ---------------------------------------------------------
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples (seed={_seed})")
    else:
        start_idx = args.start - 1
        subset = (all_data[start_idx:start_idx + n]
                  if args.samples != 0
                  else all_data[start_idx:])
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # -- Load model --------------------------------------------------------
    # [Change 1] batch_size argument removed — always 1 inside load_model.
    model = load_model(args.model)

    # -- Prepare output CSV ------------------------------------------------
    out_path = args.output or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "model_reasoning", "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # -- Evaluation loop (batch_size=1 internally) -------------------------
    results = []
    total   = len(subset)
    print(f"Starting evaluation ({total} samples, greedy decoding, fp16)...\n")

    # [Change 1] Loop in steps of 1; run_batch still accepts a list for
    # structural consistency but always receives a single-element list.
    for batch_start in range(0, total, 1):
        batch      = subset[batch_start : batch_start + 1]
        batch_idxs = [args.start + batch_start]

        print(f"[{batch_idxs[0]}/{args.start + total - 1}] Running...")

        try:
            # [Change 2] temperature and repetition_penalty removed from call.
            resps = run_batch(
                model, batch, args.ev_limit,
                args.max_tokens,
                no_think=args.no_think,
            )
        except Exception as e:
            print(f"  Error ({e}), retrying up to 3 times...")
            resps = []
            for item in batch:
                for attempt in range(3):
                    try:
                        msgs = _build_messages(item, args.ev_limit,
                                               no_think=args.no_think)
                        pipe = model["pipeline"]
                        # [Change 2] Same greedy pinning in the retry path.
                        from transformers import GenerationConfig
                        _retry_cfg = GenerationConfig(
                            max_new_tokens=args.max_tokens,
                            do_sample=False,
                            pad_token_id=pipe.tokenizer.pad_token_id,
                        )
                        raw = pipe(
                            msgs,
                            generation_config=_retry_cfg,
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

        # Process results for this sample
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

        if args.delay > 0 and batch_start + 1 < total:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()