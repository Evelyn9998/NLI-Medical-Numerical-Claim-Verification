"""
CoT + External Calculator Pipeline
===================================
Step 1 (CoT, free-form): LLM reasons freely → identifies formula + extracts parameters.
Step 2 (External Calculator): Pre-written Python function computes exact result (no LLM math).
Step 3 (Verdict): LLM compares computed result vs claim → TRUE / FALSE / PARTIALLY TRUE.

Usage:
    python baseline_test/cot_external_calc.py --data data/train_300.csv
    python baseline_test/cot_external_calc.py --data data/train_300.csv --samples 20
    python baseline_test/cot_external_calc.py --data data/train_300.csv --samples 0 --random
    python baseline_test/cot_external_calc.py --data data/train_300.csv --model meta-llama/Llama-3.1-8B-Instruct
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
from train_formula_classifier import FormulaClassifier, QTC_VARIANTS

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

# Calculator reference list shown to the model
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

STEP1_SYSTEM = f"""You are a clinical data extraction assistant. Your job is to read a medical evidence note and extract the exact parameter values needed for a specific medical calculator.

Available calculators and their required parameters:
{_CALCULATOR_LIST}

Instructions:
1. Think step by step. Reason freely about the patient's clinical data, units, and which values map to each parameter.
2. Be careful about unit conversions (e.g. lbs→kg, inches→cm, °F→°C).
3. For yes/no (binary) parameters, output 1 for yes/present and 0 for no/absent.
4. For GCS sub-scores, use the exact response description (e.g. "eye opening to verbal command").
5. For scoring systems (e.g. Child-Pugh, HEART, GBS, PSI, Wells), extract the RAW clinical input values — do NOT compute or output the final score yourself.
6. Sex/gender: infer from pronouns (he/his → male, she/her → female) or words like "man", "woman".
7. Weight/height: convert units if needed (lbs÷2.205=kg; feet/inches→cm: ft×30.48+in×2.54).
8. Binary comorbidities: if not mentioned or explicitly absent, output 0. If present or history of, output 1.

9. After your reasoning, output a JSON block between <extraction> tags with this structure:

<extraction>
{{
  "calculator_id": <int>,
  "calculator_name": "<name>",
  "parameters": {{
    "<param_name>": <value>,
    ...
  }}
}}
</extraction>

The JSON must be valid. Do not output anything after the closing </extraction> tag."""


STEP3_SYSTEM = """You are a medical fact-checking assistant.

You are given:
- A clinical evidence note
- A claim about a medical calculation (structured as: "Based on the patient's data where <entity_1> is <value_1>, <entity_2> is <value_2>, ..., <entity_N> is <value_N>.")
- The calculator name used
- The EXACT computed value produced by a verified Python calculator function

Your task: decide if the claim is TRUE, FALSE, or PARTIALLY TRUE by following these steps carefully.

---

STEP-BY-STEP VERIFICATION PROCEDURE:

Step 1 — Entity verification (all entities EXCEPT the last one):
  The claim lists entities in the form "<name> is <value>". Check each entity except the LAST one:
  - Compare the claim's value against the evidence note.
  - If an entity is NOT mentioned in the evidence, and the claim states it is "none" or "no", treat that entity as CORRECT.
  - If an entity is NOT mentioned in the evidence, and the claim states a non-zero or non-none value, treat that entity as INCORRECT.
  - An entity that is PARTIALLY CORRECT (some aspects right, some wrong) counts as INCORRECT.
  - Count: how many entities are CORRECT and how many are INCORRECT (including partially correct ones).

Step 2 — Calculation verification (the last entity in the claim):
  The last entity in the claim is the computed result for the named calculator. Compare it against the EXACT computed value:
  - IMPORTANT: The calculator name above tells you exactly which entity to verify. Only compare the entity that represents the output of that specific calculator — ignore any other numeric values in the claim.
  - Rule-based calculators (scores, classifications): the claim value must EXACTLY match the computed value.
  - Equation-based calculators (lab tests, physical measurements, dosage conversions): the claim value must be within 5% of the computed value.
  - Date-based calculators: the claim date must EXACTLY match the computed date.

Step 3 — Assign label (read ALL four definitions carefully before choosing):
  - "true":           ALL input entities (Step 1) are correct AND the computed result (Step 2) is correct.
  - "partially true": ALL input entities (Step 1) are correct AND the computed result (Step 2) is WRONG.
                      Use this when the inputs match evidence perfectly but the final answer is off.
  - "false":          ONE OR MORE input entities (Step 1) are incorrect, regardless of whether the result is correct or wrong.
                      Mixed entity case: if SOME entities are correct and OTHERS are incorrect, that still qualifies as "false" — having some correct entities does NOT prevent "false".
                      Even if the computed result happens to match the claim, input entity errors mean the label is "false".

CRITICAL RULES:
- "false" requires at least one incorrect input entity. If ALL inputs are correct, you MUST choose "true" or "partially true".
- Input entity errors always take priority: one or more incorrect entities → "false", no matter what the result is.
- You MUST end your response with the JSON object. Be concise in reasoning — for long entity lists, summarize rather than check each one individually.

---

Output ONLY this JSON (no markdown fences):
{
  "reasoning": "<concise verification: summarize entity correctness, state computed vs claim result, assign label>",
  "label": "<true|false|partially true>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_label(label: str) -> str:
    label = (label or "").lower().strip()
    if label == "true":            return "true"
    if label == "false":           return "false"
    if "partial" in label:         return "partially true"
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
                    "evidence":          row["Evidence"],
                    "claim":             row["Claim"],
                    "label":             normalize_label(row["Label"]),
                    "explanation":       row.get("Explanation", "").strip(),
                    "ground_truth":      row.get("Ground Truth Answer", ""),
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
# Step 1: CoT extraction
# ─────────────────────────────────────────────────────────────────────────────

def step1_extract(client, model, item: dict, classifier: "FormulaClassifier",
                  temperature: float, max_tokens: int, ev_limit: int) -> dict:
    """
    Returns {"reasoning": str, "calculator_id": int, "calculator_name": str, "parameters": dict}
    Uses FormulaClassifier to decide which calculator to apply from the claim text.
    """
    # Use the classifier to predict calculator name from the claim
    predicted_calc_name = classifier.predict(item["claim"])
    calc_id = _resolve_calculator_id(predicted_calc_name)

    hint = (f"The calculator for this case is: "
            f"ID {calc_id} — {predicted_calc_name}")

    user_msg = (
        f"{hint}\n\n"
        f"Evidence (clinical note):\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim (for context only — do not use claim values in extraction):\n"
        f"{item['claim']}\n\n"
        "Extract ALL required parameter values from the evidence note. "
        "Think step by step, then output the <extraction> JSON."
    )

    raw = _chat(client, model,
                [{"role": "system", "content": STEP1_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature, max_tokens)

    # ── Parse extraction block ────────────────────────────────────────────────
    match = re.search(r"<extraction>([\s\S]*?)</extraction>", raw, re.IGNORECASE)
    json_str = match.group(1).strip() if match else raw

    # Strip markdown fences if present
    json_str = re.sub(r"```(?:json)?|```", "", json_str).strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: try to grab the JSON object with regex
        m = re.search(r"\{[\s\S]*\}", json_str)
        try:
            parsed = json.loads(m.group()) if m else {}
        except Exception:
            parsed = {}

    params = parsed.get("parameters", parsed)  # fallback: top-level keys

    return {
        "reasoning":       raw,
        "calculator_id":   calc_id,
        "calculator_name": predicted_calc_name,
        "parameters":      params if isinstance(params, dict) else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calculator ID resolver  (classifier name → integer ID)
# ─────────────────────────────────────────────────────────────────────────────

_CALC_NAME_TO_ID: dict = {}

# Explicit aliases for common classifier output names that don't match _CALCULATOR_LIST
_CALC_ALIASES: dict = {
    # Wells' DVT (ID 16)
    "wells' criteria for dvt": 16,
    "wells criteria for dvt": 16,
    "wells' criteria for deep vein thrombosis": 16,
    "wells dvt criteria": 16,
    "wells dvt score": 16,
    # Wells' PE (ID 8)
    "wells' criteria for pulmonary embolism": 8,
    "wells criteria for pulmonary embolism": 8,
    "wells' pe criteria": 8,
    "wells' criteria for pe": 8,
    "wells pe score": 8,
    # Maintenance Fluids (ID 22)
    "maintenance fluids calculations": 22,
    "maintenance fluids calculation": 22,
    "maintenance fluid calculations": 22,
    "holliday-segar": 22,
    "holliday segar": 22,
    # Estimated Date of Conception (ID 68)
    "estimated of conception": 68,
    "estimated date of conception": 68,
    "estimated conception date": 68,
    "date of conception": 68,
    # Body Surface Area (ID 60)
    "body surface area calculator": 60,
    "body surface area (mosteller)": 60,
    "bsa (mosteller)": 60,
    "bsa calculator": 60,
    "body surface area": 60,
    # PSI (ID 29)
    "pneumonia severity index": 29,
    "psi score": 29,
    "psi: pneumonia severity index": 29,
    # CCI (ID 32)
    "charlson comorbidity index": 32,
    "cci score": 32,
}


def _build_name_to_id() -> dict:
    """Parse _CALCULATOR_LIST once and return a {lowercased_name: id} dict."""
    mapping: dict = {}
    for line in _CALCULATOR_LIST.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"ID\s+(\d+)\s+[—–-]+\s+(.+?)(?::|$)", line)
        if m:
            mapping[m.group(2).strip().lower()] = int(m.group(1))
    return mapping


def _resolve_calculator_id(calc_name: str) -> int:
    """Return the integer calculator ID for a given classifier output name."""
    global _CALC_NAME_TO_ID
    if not _CALC_NAME_TO_ID:
        _CALC_NAME_TO_ID = _build_name_to_id()

    key = calc_name.lower().strip()

    # 1. Exact match in calculator list
    if key in _CALC_NAME_TO_ID:
        return _CALC_NAME_TO_ID[key]

    # 2. Exact match in explicit aliases
    if key in _CALC_ALIASES:
        return _CALC_ALIASES[key]

    # 3. Partial match in explicit aliases
    for alias, cid in _CALC_ALIASES.items():
        if alias in key or key in alias:
            return cid

    # 4. Substring match in calculator list
    for name, cid in _CALC_NAME_TO_ID.items():
        if name in key or key in name:
            return cid

    # 5. Word-overlap: find list entry with most shared significant words
    key_words = set(re.findall(r'\w{3,}', key))
    best_cid, best_overlap = 0, 0
    for name, cid in _CALC_NAME_TO_ID.items():
        name_words = set(re.findall(r'\w{3,}', name))
        overlap = len(key_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cid = cid
    if best_overlap >= 3:
        return best_cid

    return 0  # unknown — run_calculator will fail gracefully




# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verdict
# ─────────────────────────────────────────────────────────────────────────────

def step3_verdict(client, model, item: dict,
                  computed: dict,
                  temperature: float, max_tokens: int, ev_limit: int) -> dict:
    """
    Returns {"reasoning": str, "label": str}
    """
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
        # Try extracting label field
        lm = re.search(r'"label"\s*:\s*"([^"]+)"', clean, re.IGNORECASE)
        rm = re.search(r'"reasoning"\s*:\s*"([\s\S]+?)"(?:\s*[,}])', clean, re.IGNORECASE)
        label = normalize_label(lm.group(1)) if lm else _fallback_label(clean)
        return {
            "reasoning": rm.group(1) if rm else clean,
            "label":     label,
        }


def _fallback_label(text: str) -> str:
    t = text.lower()
    # Must check "partially true" before "true" to avoid false positive
    if "partially true" in t or "partially_true" in t:
        return "partially true"
    # Look for label assignment patterns before generic word search
    if re.search(r'label["\s:]+false', t) or re.search(r'assign.*\bfalse\b', t):
        return "false"
    if re.search(r'label["\s:]+true', t) or re.search(r'assign.*\btrue\b', t):
        return "true"
    # Last resort: find the last occurrence of a label word (model tends to state label at the end)
    last_false = t.rfind('"false"')
    last_true  = t.rfind('"true"')
    if last_false > last_true:
        return "false"
    if last_true > last_false:
        return "true"
    # Plain word search
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
    by_label = defaultdict(lambda: [0, 0])
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

    # Step-2 error rate
    calc_errors = sum(1 for r in results if r.get("step2_error"))
    if calc_errors:
        print(f"\nStep-2 calculator errors: {calc_errors}/{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoT + External Calculator pipeline for medical NLI"
    )
    parser.add_argument("--data",       required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path (default: meta-llama/Llama-3.1-8B-Instruct)")
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
    parser.add_argument("--max-tokens-step3", type=int, default=1024,
                        help="Max tokens for Step 3 (default: 1024)")
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Seconds between samples (default: 2.0)")
    parser.add_argument("--output",     default="",
                        help="Output CSV path (auto-generated if empty)")
    parser.add_argument("--classifier", default="./formula_classifier",
                        help="Path to trained FormulaClassifier directory "
                             "(default: ./formula_classifier)")
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

    # ── Load formula classifier ───────────────────────────────────────────
    print(f"Loading formula classifier from: {args.classifier}")
    classifier = FormulaClassifier(args.classifier)
    print("Formula classifier ready.\n")

    # ── Load LLM ──────────────────────────────────────────────────────────
    client, model = load_model(args.model)

    # ── Prepare output CSV ────────────────────────────────────────────────
    out_path = args.output or (
        f"results/cot_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    fieldnames = [
        "index", "classifier_calculator_id", "classifier_calculator_name",
        "evidence", "claim", "true_label", "predicted_label", "match",
        "step1_reasoning", "step1_extracted_params",
        "step2_computed_value", "step2_unit", "step2_error",
        "step3_reasoning",
        "ground_truth",
        "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # ── Evaluation loop ───────────────────────────────────────────────────
    results = []
    total = len(subset)

    for i, item in enumerate(subset, start=args.start):
        print(f"[{i}/{args.start + total - 1}] claim: {item['claim'][:80]}...")

        # ── Step 1: CoT extraction (classifier decides calculator) ────────
        s1 = _with_retry(
            step1_extract,
            client, model, item, classifier,
            args.temperature, args.max_tokens_step1, args.ev_limit
        )
        if s1 is None:
            s1 = {"reasoning": "ERROR", "calculator_id": 0,
                  "calculator_name": "unknown", "parameters": {}}

        print(f"  Classifier → ID {s1['calculator_id']}: {s1['calculator_name']}")
        print(f"  Step 1 params: {list(s1['parameters'].keys())}")

        # Inject classifier-resolved name into item so step3 can reference it
        item_with_calc = {**item,
                          "calculator_name": s1["calculator_name"],
                          "calculator_id":   s1["calculator_id"]}

        # ── Step 2: Python calculator ─────────────────────────────────────
        computed = run_calculator(s1["calculator_id"], s1["parameters"])
        print(f"  Step 2 computed: {computed['value']} {computed['unit']}"
              + (f"  [!] {computed['note']}" if computed.get("note") else ""))

        # ── Step 3: Verdict ───────────────────────────────────────────────
        s3 = _with_retry(
            step3_verdict,
            client, model, item_with_calc, computed,
            args.temperature, args.max_tokens_step3, args.ev_limit
        )
        if s3 is None:
            s3 = {"reasoning": "ERROR", "label": "unknown"}

        predicted = s3["label"]
        match = predicted == item["label"]
        print(f"  true={item['label']}  predicted={predicted}  match={match}")

        row = {
            "index":                      i,
            "classifier_calculator_id":   s1["calculator_id"],
            "classifier_calculator_name": s1["calculator_name"],
            "evidence":                   item["evidence"],
            "claim":                      item["claim"],
            "true_label":                 item["label"],
            "predicted_label":            predicted,
            "match":                      "TRUE" if match else "FALSE",
            "step1_reasoning":            s1["reasoning"].replace("\n", " "),
            "step1_extracted_params":     json.dumps(s1["parameters"],
                                                     ensure_ascii=False),
            "step2_computed_value":       str(computed.get("value", "")),
            "step2_unit":                 computed.get("unit", ""),
            "step2_error":                computed.get("note", ""),
            "step3_reasoning":            s3["reasoning"].replace("\n", " "),
            "ground_truth":               item["ground_truth"],
            "dataset_explanation":        item["explanation"].replace("\n", " "),
        }
        results.append({**row, "match": match})

        # Write live
        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if i < args.start + total - 1:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()