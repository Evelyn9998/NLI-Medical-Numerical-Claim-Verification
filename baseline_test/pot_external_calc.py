"""
PoT (Program of Thought) + External Calculator Pipeline
========================================================
Step 1 (PoT):        LLM generates Python code to extract parameters from evidence.
                     Code is executed to produce exact parameter values (handles unit conversions).
Step 2 (External):   Pre-written Python calculator computes exact result (no LLM math).
Step 3 (Verdict):    LLM compares computed result vs claim → TRUE / FALSE / PARTIALLY TRUE.

Usage:
    python baseline_test/pot_external_calc.py --data data/train_300.csv
    python baseline_test/pot_external_calc.py --data data/train_300.csv --samples 20
    python baseline_test/pot_external_calc.py --data data/train_300.csv --samples 0
    python baseline_test/pot_external_calc.py --data data/train_300.csv --model meta-llama/Llama-3.1-8B-Instruct
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

# Add project root to path so we can import calculators
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculators import run_calculator

# ─────────────────────────────────────────────────────────────────────────────
# Calculator reference list
# ─────────────────────────────────────────────────────────────────────────────

_CALCULATOR_LIST = """
ID 2  — Creatinine Clearance (Cockcroft-Gault): age, weight(kg), creatinine(mg/dL), height(cm), sex
ID 3  — CKD-EPI GFR: age, creatinine(mg/dL), sex, [race]
ID 4  — CHA2DS2-VASc: age, sex, chf, hypertension, diabetes, stroke/tia, vascular_disease
ID 5  — Mean Arterial Pressure: sbp(mmHg), dbp(mmHg)
ID 6  — BMI: weight(kg), height(cm)
ID 7  — Calcium Correction for Hypoalbuminemia: calcium(mg/dL), albumin(g/dL)
ID 8  — Wells' PE: clinical signs of DVT, pe_top_diagnosis, heart_rate, immobilization, previous_dvt_pe, hemoptysis, malignancy
ID 9  — MDRD GFR: age, creatinine(mg/dL), sex, [race]
ID 10 — Ideal Body Weight: height(cm), sex
ID 11 — QTc Bazett: qt_interval(ms), heart_rate(bpm)
ID 13 — Estimated Due Date: last_menstrual_date(YYYY-MM-DD), [cycle_length=28]
ID 15 — Child-Pugh: albumin(g/dL), bilirubin(mg/dL), inr, ascites(none/slight/moderate), encephalopathy(none/grade1-2/grade3-4)
ID 16 — Wells' DVT: active_cancer, paralysis_paresis_plaster, bedridden_recently, major_surgery, localized_tenderness, entire_leg_swollen, calf_swelling, pitting_edema, collateral_veins, previous_dvt, alternative_diagnosis
ID 17 — RCRI: high_risk_surgery, ischemic_heart_disease, chf, cerebrovascular_disease, insulin, creatinine(mg/dL)
ID 18 — HEART Score: history(0-2), ecg(0-2), age, risk_factors(0-2), troponin(0-2)
ID 19 — FIB-4: age, ast(U/L), alt(U/L), platelets(×10³/µL)
ID 20 — Centor/McIsaac: age, temperature(°C), cough_absent, tonsillar_exudate, tender_anterior_cervical_lymph_nodes
ID 21 — GCS: best_eye_response, best_verbal_response, best_motor_response
ID 22 — Maintenance Fluids (Holliday-Segar): weight(kg)
ID 23 — MELD Na: bilirubin(mg/dL), creatinine(mg/dL), inr, sodium(mEq/L), [dialysis=0]
ID 24 — Steroid Conversion: input_steroid, dose(mg), target_steroid
ID 25 — HAS-BLED: hypertension, renal_disease, liver_disease, stroke, prior_bleeding, labile_inr, age, medication, alcohol
ID 26 — Sodium Correction for Hyperglycemia: sodium(mEq/L), glucose(mg/dL)
ID 27 — Glasgow-Blatchford: bun(mg/dL), hemoglobin(g/dL), sbp(mmHg), sex, [heart_rate, melena, syncope, hepatic_disease, cardiac_failure]
ID 28 — APACHE II: temperature(°C), map(mmHg), heart_rate, respiratory_rate, pao2, fio2, ph, sodium, potassium, creatinine, hematocrit, wbc, gcs, age, [acute_renal_failure, chronic_health, surgery_type]
ID 29 — PSI: age, sex, nursing_home, neoplastic_disease, liver_disease, chf, cerebrovascular_disease, renal_disease, altered_mental_status, respiratory_rate, sbp, temperature, heart_rate, ph, bun, sodium, glucose, hematocrit, pao2, pleural_effusion
ID 30 — Serum Osmolality: sodium(mEq/L), glucose(mg/dL), bun(mg/dL)
ID 31 — HOMA-IR: insulin(µIU/mL), glucose(mg/dL)
ID 32 — Charlson Comorbidity Index: age + various comorbidities (0/1 flags)
ID 33 — FeverPAIN: fever, absence_of_cough_or_coryza, symptom_onset_3_days, purulent_tonsils, severe_tonsil_inflammation
ID 36 — Caprini VTE: age, bmi, sex, various risk factors (0/1 flags)
ID 38 — Free Water Deficit: sodium(mEq/L), weight(kg), age, sex
ID 39 — Anion Gap: sodium(mEq/L), chloride(mEq/L), bicarbonate(mEq/L)
ID 40 — FENa: sodium(mEq/L), urine_sodium(mEq/L), creatinine(mg/dL), urine_creatinine(mg/dL)
ID 43 — SOFA: pao2, fio2, platelets, bilirubin, map, vasopressors, gcs, creatinine, [urine_output]
ID 44 — LDL (Friedewald): total_cholesterol, hdl, triglycerides (all mg/dL)
ID 45 — CURB-65: confusion, bun(mg/dL), respiratory_rate, sbp, dbp, age
ID 46 — Framingham Risk Score: age, sex, total_cholesterol, hdl, sbp, bp_treated(0/1), smoker(0/1)
ID 48 — PERC Rule: age, heart_rate, spo2(%), unilateral_leg_swelling, hemoptysis, recent_surgery, previous_dvt_pe, hormone_use
ID 49 — MME Calculator: [drug]_dose_per_day for each opioid (morphine, oxycodone, hydrocodone, etc.)
ID 51 — SIRS: temperature(°C), heart_rate, respiratory_rate, paco2, wbc(×10³/µL)
ID 56 — QTc Fridericia: qt_interval(ms), heart_rate(bpm)
ID 57 — QTc Framingham: qt_interval(ms), heart_rate(bpm)
ID 58 — QTc Hodges: qt_interval(ms), heart_rate(bpm)
ID 59 — QTc Rautaharju: qt_interval(ms), heart_rate(bpm)
ID 60 — Body Surface Area (Mosteller): height(cm), weight(kg)
ID 61 — Target Weight: bmi(kg/m²), height(cm)
ID 62 — Adjusted Body Weight: weight(kg), height(cm), sex
ID 63 — Delta Gap: sodium, chloride, bicarbonate (mEq/L)
ID 64 — Delta Ratio: sodium, chloride, bicarbonate (mEq/L)
ID 65 — Albumin Corrected Anion Gap: sodium, chloride, bicarbonate (mEq/L), albumin(g/dL)
ID 66 — Albumin Corrected Delta Gap: sodium, chloride, bicarbonate (mEq/L), albumin(g/dL)
ID 67 — Albumin Corrected Delta Ratio: sodium, chloride, bicarbonate (mEq/L), albumin(g/dL)
ID 68 — Estimated Date of Conception: last_menstrual_date(YYYY-MM-DD)
ID 69 — Estimated Gestational Age: last_menstrual_date(YYYY-MM-DD), current_date(YYYY-MM-DD)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

STEP1_SYSTEM = f"""You are a clinical data extraction assistant. Your job is to read a medical evidence note and write a short Python program that extracts the exact parameter values needed for a specific medical calculator.

Available calculators and their required parameters:
{_CALCULATOR_LIST}

Your output must be a Python code block that:
1. Reads values directly from the evidence (assign them as Python variables with comments showing where you found each value).
2. Performs any necessary unit conversions using arithmetic (do NOT use import statements):
   - lbs to kg: kg = lbs / 2.205
   - feet/inches to cm: cm = feet * 30.48 + inches * 2.54
   - °F to °C: c = (f - 32) * 5/9
3. For yes/no (binary) parameters: use 1 for yes/present, 0 for no/absent.
4. For scoring systems (Child-Pugh, HEART, GBS, PSI, Wells): extract RAW clinical input values — do NOT compute the final score.
5. Infer sex from pronouns (he/his → "male", she/her → "female") or words like "man", "woman".
6. If a value is truly not present in the evidence, set it to None.
7. Build a dict called `params` containing all required parameter names and their values.

IMPORTANT RULES:
- Do NOT use any import statements.
- Do NOT use input(), open(), or any file/network operations.
- The last line must assign the `params` dict.
- Only output the Python code block, nothing else.

Example output format:
```python
# Age: "68-year-old"
age = 68

# Weight: "weighs 180 lbs" → convert to kg
weight_lbs = 180
weight = weight_lbs / 2.205  # = 81.6 kg

# Height: "5 feet 10 inches" → convert to cm
height = 5 * 30.48 + 10 * 2.54  # = 177.8 cm

# Sex: "he" → male
sex = "male"

# Creatinine: "serum creatinine 1.2 mg/dL"
creatinine = 1.2

params = {{
    "age": age,
    "weight": round(weight, 1),
    "height": round(height, 1),
    "sex": sex,
    "creatinine": creatinine,
}}
```"""


STEP3_SYSTEM = """You are a medical fact-checking assistant.

You are given:
- A clinical evidence note
- A claim about a medical calculation (structured as: "Based on the patient's data where <entity_1> is <value_1>, <entity_2> is <value_2>, ..., <entity_N> is <value_N>.")
- The calculator name used
- The EXACT computed value produced by a verified Python calculator function
- The relevant entities extracted from the evidence (the ground-truth parameter values)

Your task: decide if the claim is TRUE, FALSE, or PARTIALLY TRUE by following these steps carefully.

---

STEP-BY-STEP VERIFICATION PROCEDURE:

Step 1 — Entity verification (all entities EXCEPT the last one):
  The claim lists entities in the form "<name> is <value>". Check each entity except the LAST one:
  - Compare the claim's value against the evidence note.
  - If an entity is NOT mentioned in the evidence, and the claim states it is "none" or "no", treat that entity as CORRECT.
  - If an entity is NOT mentioned in the evidence, and the claim states a non-zero or non-none value, treat that entity as INCORRECT.
  - Count how many input entities are correct and how many are incorrect.

Step 2 — Calculation verification (the last entity in the claim):
  The last entity in the claim is the computed result. Compare it against the EXACT computed value using these rules:
  - Rule-based calculators (scores, classifications): the claim value must EXACTLY match the computed value.
  - Equation-based calculators (lab tests, physical measurements, dosage conversions): the claim value must be within 5% of the computed value.
  - Date-based calculators: the claim date must EXACTLY match the computed date.

Step 3 — Assign label (read ALL three definitions carefully before choosing):
  - "true":           ALL input entities (Step 1) are correct AND the computed result (Step 2) is correct.
  - "partially true": ALL input entities (Step 1) are correct AND the computed result (Step 2) is WRONG.
                      Use this when the inputs match evidence perfectly but the final answer is off.
  - "false":          ONE OR MORE input entities (Step 1) are incorrect AND the computed result (Step 2) is also WRONG.
                      Use this ONLY when there is an input entity error — do NOT use "false" just because the result is wrong.

CRITICAL RULE: The label "false" requires an input entity error. If all inputs are correct, you MUST choose "true" or "partially true", never "false".

---

Show your work for each step, then output ONLY this JSON (no markdown fences):
{
  "reasoning": "<step-by-step verification: entity check, formula used, calculation process, comparison with claim>",
  "label": "<true|false|partially true>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_label(label: str) -> str:
    label = (label or "").lower().strip()
    if label == "true":          return "true"
    if label == "false":         return "false"
    if "partial" in label:       return "partially true"
    return label


def load_dataset(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Evidence") and row.get("Claim") and row.get("Label"):
                rows.append({
                    "calculator_id":     int(row.get("Calculator ID", 0) or 0),
                    "calculator_name":   row.get("Calculator Name", ""),
                    "evidence":          row["Evidence"],
                    "claim":             row["Claim"],
                    "label":             normalize_label(row["Label"]),
                    "explanation":       row.get("Explanation", "").strip(),
                    "ground_truth":      row.get("Ground Truth Answer", ""),
                    "lower_limit":       row.get("Lower Limit", ""),
                    "upper_limit":       row.get("Upper Limit", ""),
                    "relevant_entities": row.get("Relevant Entities", ""),
                })
    print(f"Loaded {len(rows)} samples from {csv_path}")
    return rows


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


def _chat(pipe, _model_name, messages, temperature, max_tokens):
    do_sample = temperature > 0.0
    result = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        return_full_text=False,
    )
    return result[0]["generated_text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: PoT extraction
# ─────────────────────────────────────────────────────────────────────────────

# Blocked built-ins for safe exec
_SAFE_GLOBALS = {"__builtins__": {"round": round, "abs": abs, "int": int,
                                   "float": float, "str": str, "None": None,
                                   "True": True, "False": False, "print": print}}


def _exec_code(code: str) -> dict:
    """Execute generated Python code and return the `params` dict."""
    local_vars = {}
    exec(code, _SAFE_GLOBALS.copy(), local_vars)
    params = local_vars.get("params", {})
    if not isinstance(params, dict):
        return {}
    # Remove None values so calculator knows they are missing
    return {k: v for k, v in params.items() if v is not None}


def step1_extract(client, model, item: dict,
                  temperature: float, max_tokens: int, ev_limit: int) -> dict:
    """
    Returns {"code": str, "parameters": dict, "exec_error": str}
    Uses PoT: LLM generates Python code, code is executed to get params.
    """
    hint = (f"The calculator for this case is: "
            f"ID {item['calculator_id']} — {item['calculator_name']}")

    user_msg = (
        f"{hint}\n\n"
        f"Evidence (clinical note):\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim (for context only — do not use claim values):\n"
        f"{item['claim']}\n\n"
        "Write Python code to extract all required parameter values from the evidence. "
        "Do NOT include the final calculated result in params — only extract raw input values from the evidence. "
        "Output ONLY the ```python ... ``` code block."
    )

    raw = _chat(client, model,
                [{"role": "system", "content": STEP1_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature, max_tokens)

    # Extract code block
    code_match = re.search(r"```python([\s\S]*?)```", raw, re.IGNORECASE)
    if not code_match:
        # Fallback: try bare code block
        code_match = re.search(r"```([\s\S]*?)```", raw)

    exec_error = ""
    params = {}

    if code_match:
        code = code_match.group(1).strip()
        try:
            params = _exec_code(code)
        except Exception as e:
            exec_error = f"exec error: {e}"
            # Try to salvage params via JSON fallback in raw text
            jm = re.search(r"\{[\s\S]*\}", raw)
            if jm:
                try:
                    params = json.loads(jm.group())
                except Exception:
                    pass
    else:
        exec_error = "no code block found"
        # Try JSON fallback
        jm = re.search(r"\{[\s\S]*\}", raw)
        if jm:
            try:
                params = json.loads(jm.group())
            except Exception:
                pass
        code = raw

    return {
        "code":       code_match.group(1).strip() if code_match else raw,
        "parameters": params,
        "exec_error": exec_error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verdict
# ─────────────────────────────────────────────────────────────────────────────

def step3_verdict(client, model, item: dict,
                  computed: dict,
                  temperature: float, max_tokens: int, ev_limit: int) -> dict:
    """Returns {"reasoning": str, "label": str}"""
    computed_str = (f"{computed['value']} {computed['unit']}".strip()
                    if computed.get("value") is not None
                    else f"[calculation error: {computed.get('note', 'unknown')}]")

    user_msg = (
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        f"Calculator used: {item['calculator_name']}\n"
        f"Computed result (exact, from verified Python function): {computed_str}\n\n"
        "Verify the claim step by step and output the JSON label."
    )

    raw = _chat(client, model,
                [{"role": "system", "content": STEP3_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature, max_tokens)

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(clean)
        return {
            "reasoning": parsed.get("reasoning", ""),
            "label":     normalize_label(parsed.get("label", "")),
        }
    except json.JSONDecodeError:
        lm = re.search(r'"label"\s*:\s*"([^"]+)"', clean, re.IGNORECASE)
        rm = re.search(r'"reasoning"\s*:\s*"([\s\S]+?)"(?:\s*[,}])', clean, re.IGNORECASE)
        label = normalize_label(lm.group(1)) if lm else _fallback_label(clean)
        return {
            "reasoning": rm.group(1) if rm else clean,
            "label":     label,
        }


def _fallback_label(text: str) -> str:
    t = text.lower()
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    if re.search(r'label["\s:]+false', t) or re.search(r'assign.*\bfalse\b', t):
        return "false"
    if re.search(r'label["\s:]+true', t) or re.search(r'assign.*\btrue\b', t):
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
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Retry wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"  [attempt {attempt+1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Waiting {wait}s...")
                time.sleep(wait)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

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
    by_label  = defaultdict(lambda: [0, 0])
    pred_dist = defaultdict(int)
    for r in results:
        by_label[r["true_label"]][1] += 1
        if r["match"]:
            by_label[r["true_label"]][0] += 1
        pred_dist[r["predicted_label"]] += 1

    print("\nAccuracy by true label:")
    for lbl in ["true", "false", "partially true"]:
        c, t = by_label[lbl]
        if t:
            print(f"  {lbl:<16} {c}/{t}  ({c/t*100:.0f}%)")

    print("\nPrediction distribution:")
    for lbl, cnt in sorted(pred_dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<16} {cnt}  ({cnt/n*100:.0f}%)")

    calc_errors = sum(1 for r in results if r.get("step2_error"))
    exec_errors = sum(1 for r in results if r.get("step1_exec_error"))
    if calc_errors:
        print(f"\nStep-2 calculator errors : {calc_errors}/{n}")
    if exec_errors:
        print(f"Step-1 exec errors       : {exec_errors}/{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PoT + External Calculator pipeline for medical NLI"
    )
    parser.add_argument("--data",       required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--samples",    type=int, default=10,
                        help="Number of samples (0 = all)")
    parser.add_argument("--random",     action="store_true",
                        help="Randomly sample")
    parser.add_argument("--start",      type=int, default=1,
                        help="1-based start index")
    parser.add_argument("--ev-limit",   type=int, default=3000,
                        help="Max evidence characters (default: 3000)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens-step1", type=int, default=1024,
                        help="Max tokens for Step 1 (default: 1024)")
    parser.add_argument("--max-tokens-step3", type=int, default=512,
                        help="Max tokens for Step 3 (default: 512)")
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Seconds between samples (default: 2.0)")
    parser.add_argument("--output",     default="",
                        help="Output CSV path (auto-generated if empty)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    all_data = load_dataset(args.data)
    n = len(all_data) if args.samples == 0 else min(args.samples, len(all_data))

    if args.random:
        subset = random.sample(all_data, n)
        print(f"Randomly selected {n} samples")
    else:
        s = args.start - 1
        subset = all_data[s:s + n] if args.samples != 0 else all_data[s:]
        print(f"Using samples {args.start} to {args.start + len(subset) - 1}")

    # ── Load model ────────────────────────────────────────────────────────
    client, model = load_model(args.model)

    # ── Prepare output CSV ────────────────────────────────────────────────
    out_path = args.output or (
        f"results/pot_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    fieldnames = [
        "index", "calculator_id", "calculator_name",
        "evidence", "claim", "true_label", "predicted_label", "match",
        "step1_code", "step1_extracted_params", "step1_exec_error",
        "step2_computed_value", "step2_unit", "step2_error",
        "step3_reasoning",
        "ground_truth", "lower_limit", "upper_limit",
        "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # ── Evaluation loop ───────────────────────────────────────────────────
    results = []
    total = len(subset)

    for i, item in enumerate(subset, start=args.start):
        print(f"[{i}/{args.start + total - 1}] "
              f"ID {item['calculator_id']}: {item['calculator_name']}")

        # ── Step 1: PoT extraction ────────────────────────────────────────
        s1 = _with_retry(
            step1_extract,
            client, model, item,
            args.temperature, args.max_tokens_step1, args.ev_limit
        )
        if s1 is None:
            s1 = {"code": "ERROR", "parameters": {}, "exec_error": "retry failed"}

        if s1["exec_error"]:
            print(f"  Step 1 exec error: {s1['exec_error']}")
        print(f"  Step 1 params: {list(s1['parameters'].keys())}")

        # ── Step 2: Python calculator ─────────────────────────────────────
        computed = run_calculator(item["calculator_id"], s1["parameters"])
        print(f"  Step 2 computed: {computed['value']} {computed['unit']}"
              + (f"  [!] {computed['note']}" if computed.get("note") else ""))

        # ── Step 3: Verdict ───────────────────────────────────────────────
        s3 = _with_retry(
            step3_verdict,
            client, model, item, computed,
            args.temperature, args.max_tokens_step3, args.ev_limit
        )
        if s3 is None:
            s3 = {"reasoning": "ERROR", "label": "unknown"}

        predicted = s3["label"]
        match = predicted == item["label"]
        print(f"  true={item['label']}  predicted={predicted}  match={match}")

        row = {
            "index":                  i,
            "calculator_id":          item["calculator_id"],
            "calculator_name":        item["calculator_name"],
            "evidence":               item["evidence"],
            "claim":                  item["claim"],
            "true_label":             item["label"],
            "predicted_label":        predicted,
            "match":                  "TRUE" if match else "FALSE",
            "step1_code":             s1["code"].replace("\n", " "),
            "step1_extracted_params": json.dumps(s1["parameters"], ensure_ascii=False),
            "step1_exec_error":       s1["exec_error"],
            "step2_computed_value":   str(computed.get("value", "")),
            "step2_unit":             computed.get("unit", ""),
            "step2_error":            computed.get("note", ""),
            "step3_reasoning":        s3["reasoning"].replace("\n", " "),
            "ground_truth":           item["ground_truth"],
            "lower_limit":            item["lower_limit"],
            "upper_limit":            item["upper_limit"],
            "dataset_explanation":    item["explanation"].replace("\n", " "),
        }
        results.append({**row, "match": match})

        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if i < args.start + total - 1:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
