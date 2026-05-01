"""
CoT + Classifier + External Calculator Pipeline
================================================
Step 0 (Classifier):  FormulaClassifier predicts calculator name from the claim text.
Step 1 (CoT):         LLM reasons freely → identifies formula + extracts parameters.
Step 2 (External):    Pre-written Python calculator computes exact result (no LLM math).
Step 3 (Verdict):     Structured JSON reasoning → label derived by Logical Reasoner.
                      FACT_1 = all input entities correct
                      FACT_2 = computed result matches claim
                      FACT_1+FACT_2=true → "true"
                      FACT_1=true, FACT_2=false → "partially true"
                      FACT_1=false → "false"

Usage:
    python baseline_test/cot_classifier_external_calc.py --data data/train_300.csv
    python baseline_test/cot_classifier_external_calc.py --data data/train_300.csv --samples 20
    python baseline_test/cot_classifier_external_calc.py --data data/train_300.csv --samples 0 --random
    python baseline_test/cot_classifier_external_calc.py --data data/train_300.csv --model meta-llama/Llama-3.1-8B-Instruct
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
from pathlib import Path

# Add project root and baseline_test/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # baseline_test/ (searched first)
from calculators import run_calculator
from baseline_test.train_formula_classifier_cot_gpt import FormulaClassifier, QTC_VARIANTS

# ─────────────────────────────────────────────────────────────────────────────
# Calculator reference list
# ─────────────────────────────────────────────────────────────────────────────

_CALCULATOR_LIST = """
ID 2  — Creatinine Clearance (Cockcroft-Gault Equation): age, creatinine, height, sex, weight
ID 3  — CKD-EPI Equations for Glomerular Filtration Rate: age, creatinine, sex
ID 4  — CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk: Congestive Heart Failure, Diabetes history, Hypertension history, Stroke, Thromboembolism history, Transient Ischemic Attacks History, Vascular disease history, age, sex
ID 5  — Mean Arterial Pressure (MAP): Diastolic Blood Pressure, Systolic Blood Pressure
ID 6  — Body Mass Index (BMI): height, weight
ID 7  — Calcium Correction for Hypoalbuminemia: Albumin, Calcium
ID 8  — Wells' Criteria for Pulmonary Embolism: Clinical signs and symptoms of Deep Vein Thrombosis, Heart Rate or Pulse, Hemoptysis, Immobilization for at least 3 days, Malignancy with treatment within 6 months or palliative, Previously Documented Pulmonary Embolism, Previously documented Deep Vein Thrombosis, Pulmonary Embolism is #1 diagnosis OR equally likely, Surgery in the previous 4 weeks
ID 9  — MDRD GFR Equation: Race, age, creatinine, sex
ID 10 — Ideal Body Weight: height, sex
ID 11 — QTc Bazett Calculator: Heart Rate or Pulse, QT Interval
ID 13 — Estimated Due Date: Last menstrual date, cycle length
ID 15 — Child-Pugh Score for Cirrhosis Mortality: Albumin, Ascites, Bilirubin, Encephalopathy, international normalized ratio
ID 16 — Wells' Criteria for DVT: Active cancer, Alternative diagnosis to Deep Vein Thrombosis as likely or more likely, Bedridden recently >3 days, Calf swelling >3 centimeters compared to the other leg, Collateral (nonvaricose) superficial veins present, Entire Leg Swollen, Localized tenderness along the deep venous system, Major surgery within 12 weeks, Paralysis, paresis, or recent plaster immobilization of the lower extremity, Pitting edema, confined to symptomatic leg, Previously documented Deep Vein Thrombosis
ID 17 — Revised Cardiac Risk Index for Pre-Operative Risk: Congestive Heart Failure criteria for the Cardiac Risk Index rule, Elevated-risk surgery, History of cerebrovascular disease, History of ischemic heart disease, Pre-operative creatinine, Pre-operative treatment with insulin
ID 18 — HEART Score for Major Cardiac Events: Diabetes mellitus, Electrocardiogram Test, Hypertension history, Initial troponin, Suspicion History, Transient Ischemic Attacks History, age, atherosclerotic disease, hypercholesterolemia, obesity, parent or sibling with Cardiovascular disease before age 65, smoking
ID 19 — Fibrosis-4 (FIB-4) Index for Liver Fibrosis: Alanine aminotransferase, Aspartate aminotransferase, Platelet count, age
ID 20 — Centor Score (Modified/McIsaac) for Strep Pharyngitis: Cough Absent, Exudate or swelling on tonsils, Temperature, Tender/swollen anterior cervical lymph nodes, age
ID 21 — Glasgow Coma Score (GCS): Best eye response, Best motor response, Best verbal response
ID 22 — Maintenance Fluids Calculations: weight
ID 23 — MELD Na (UNOS/OPTN): Bilirubin, Continuous veno-venous hemodialysis for ≥4 hours in the past week, Dialysis at least twice in the past week, Sodium, creatinine, international normalized ratio
ID 24 — Steroid Conversion Calculator: input steroid, target steroid
ID 25 — HAS-BLED Score for Major Bleeding Risk: Hypertension, Labile international normalized ratio, Liver disease criteria for the HAS-BLED rule, Medication usage predisposing to bleeding, Number of Alcoholic Drinks Per Week, Prior major bleeding or predisposition to bleeding, Renal disease criteria for the HAS-BLED rule, Stroke, age
ID 26 — Sodium Correction for Hyperglycemia: Glucose, Sodium
ID 27 — Glasgow-Blatchford Bleeding Score (GBS): Blood Urea Nitrogen (BUN), Cardiac Failure Present, Heart Rate or Pulse, Hemoglobin, Hepatic disease history, Melena Present, Recent Syncope, Systolic Blood Pressure, sex
ID 28 — APACHE II Score: A-a gradient, Acute renal failure, Chronic renal failure, Diastolic Blood Pressure, FiO2, Glasgow Coma Score, Heart Rate or Pulse, Hematocrit, History of severe organ failure or immunocompromise, PaO2, Potassium, Sodium, Surgery Type, Systolic Blood Pressure, Temperature, White blood cell count, age, creatinine, pH, respiratory rate
ID 29 — PSI Score: Pneumonia Severity Index for CAP: Altered mental status, Blood Urea Nitrogen (BUN), Cerebrovascular disease history, Congestive Heart Failure, Glucose, Heart Rate or Pulse, Hematocrit, Liver disease history, Neoplastic disease, Nursing home resident, Partial pressure of oxygen, Pleural effusion on x-ray, Renal disease history, Sodium, Systolic Blood Pressure, Temperature, age, pH, respiratory rate, sex
ID 30 — Serum Osmolality: Blood Urea Nitrogen (BUN), Glucose, Sodium
ID 31 — HOMA-IR (Homeostatic Model Assessment for Insulin Resistance): Glucose, Insulin
ID 32 — Charlson Comorbidity Index (CCI): AIDS, Cerebrovascular Accident, Chronic Pulmonary Disease, Congestive Heart Failure, Connective tissue disease, Dementia, Diabetes mellitus, Hemiplegia, Leukemia, Liver disease severity, Lymphoma, Moderate to severe Chronic Kidney Disease, Myocardial infarction, Peptic ulcer disease, Peripheral vascular disease, Solid tumor, Transient Ischemic Attacks History, age
ID 33 — FeverPAIN Score for Strep Pharyngitis: Absence of cough or coryza, Fever in past 24 hours, Purulent tonsils, Severe tonsil inflammation, Symptom onset <=3 days
ID 36 — Caprini Score for Venous Thromboembolism (2005): Acute Myocardial infarction, Acute spinal cord injury causing paralysis in the last month, Body Mass Index (BMI), Chronic Obstructive Pulmonary Disease, Congestive Heart Failure in the last month, Current central venous access, Current swollen legs, Elevated anticardiolipin antibody, Elevated serum homocysteine, Family history of thrombosis, Heparin-induced thrombocytopenia, Hip, pelvis, or leg fracture in the last month, History of inflammatory bowel disease, Immobilizing plaster cast in the last month, Major Surgery in the last month, Mobility, Multiple trauma in the last month, Other congenital or acquired thrombophilia, Pneumonia in the last month, Positive Factor V Leiden, Positive lupus anticoagulant, Positive prothrombin 20210A, Present or previous malignancy, Previously Documented Pulmonary Embolism, Previously documented Deep Vein Thrombosis, Sepsis in the last month, Stroke in the last month, Surgery Type, Varicose veins, age, sex
ID 38 — Free Water Deficit: Sodium, age, sex, weight
ID 39 — Anion Gap: Bicarbonate, Chloride, Sodium
ID 40 — Fractional Excretion of Sodium (FENa): Sodium, Urine creatinine, Urine sodium, creatinine
ID 43 — Sequential Organ Failure Assessment (SOFA) Score: Bilirubin, Continuous positive airway pressure, DOBUTamine, DOPamine, Diastolic Blood Pressure, FiO2, Glasgow Coma Score, Hypotension, On mechanical ventilation, PaO2, Platelet count, Systolic Blood Pressure, Urine Output, creatinine
ID 44 — LDL Calculated: Total cholesterol, Triglycerides, high-density lipoprotein cholesterol
ID 45 — CURB-65 Score for Pneumonia Severity: Blood Urea Nitrogen (BUN), Confusion, Diastolic Blood Pressure, Systolic Blood Pressure, age, respiratory rate
ID 46 — Framingham Risk Score for Hard Coronary Heart Disease: Blood pressure being treated with medicines, Smoker, Systolic Blood Pressure, Total cholesterol, age, high-density lipoprotein cholesterol, sex
ID 48 — PERC Rule for Pulmonary Embolism: Heart Rate or Pulse, Hemoptysis, Hormone use, O2 saturation percentage, Previously Documented Pulmonary Embolism, Previously documented Deep Vein Thrombosis, Recent surgery or trauma, Unilateral Leg Swelling, age
ID 49 — Morphine Milligram Equivalents (MME) Calculator: Codeine Dose, Codeine Dose Per Day, FentaNYL buccal Dose, FentaNYL buccal Dose Per Day, HYDROcodone Dose, HYDROcodone Dose Per Day, HYDROmorphone Dose, HYDROmorphone Dose Per Day, Methadone Dose, Methadone Dose Per Day, Morphine Dose, Morphine Dose Per Day, OxyCODONE Dose, OxyCODONE Dose Per Day, OxyMORphone Dose, OxyMORphone Dose Per Day, Tapentadol Dose, Tapentadol Dose Per Day, TraMADol Dose, TraMADol Dose Per Day
ID 51 — SIRS Criteria: Heart Rate or Pulse, PaCO2, Temperature, White blood cell count, respiratory rate
ID 56 — QTc Fridericia Calculator: Heart Rate or Pulse, QT Interval
ID 57 — QTc Framingham Calculator: Heart Rate or Pulse, QT Interval
ID 58 — QTc Hodges Calculator: Heart Rate or Pulse, QT Interval
ID 59 — QTc Rautaharju Calculator: Heart Rate or Pulse, QT Interval
ID 60 — Body Surface Area Calculator: height, weight
ID 61 — Target weight: Body Mass Index (BMI), height
ID 62 — Adjusted Body Weight: height, sex, weight
ID 63 — Delta Gap: Bicarbonate, Chloride, Sodium
ID 64 — Delta Ratio: Bicarbonate, Chloride, Sodium
ID 65 — Albumin Corrected Anion Gap: Albumin, Bicarbonate, Chloride, Sodium
ID 66 — Albumin Corrected Delta Gap: Albumin, Bicarbonate, Chloride, Sodium
ID 67 — Albumin Corrected Delta Ratio: Albumin, Bicarbonate, Chloride, Sodium
ID 68 — Estimated Date of Conception: Last menstrual date
ID 69 — Estimated Gestational Age: Current Date, Last menstrual date
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

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


# Step 3 system prompt — structured JSON with three explicit fields for logical reasoner
STEP3_SYSTEM = """You are a medical fact-checking assistant.

You are given:
- A clinical evidence note
- A claim about a medical calculation (structured as: "Based on the patient's data where <entity_1> is <value_1>, ..., <entity_N> is <value_N>.")
- The calculator name used
- The EXACT computed value produced by a verified Python calculator function

Your task: verify the claim by working through three explicit steps and returning structured JSON.

---

STEP 1 — Entity verification (all entities EXCEPT the last one):
For each input entity in the claim:
  - Compare the claim value against the evidence note.
  - correct: true  → value matches evidence, OR entity absent from evidence AND claim says "none"/"no".
  - correct: false → value differs from evidence, OR entity absent from evidence AND claim gives a specific non-zero value.

STEP 2 — Calculation verification (the LAST entity in the claim):
  Compare the claim's last entity value against the EXACT computed value provided.
  - Rule-based calculators (scores): must EXACTLY match → correct: true/false.
  - Equation-based calculators (labs, measurements): must be within 5% → correct: true/false.
  - Date-based calculators: must EXACTLY match → correct: true/false.

STEP 3 — Label (derived automatically by logical reasoner — still state it explicitly):
  - "true":           ALL step1 entities correct=true  AND  step2 correct=true
  - "partially true": ALL step1 entities correct=true  AND  step2 correct=false
  - "false":          ANY step1 entity correct=false (regardless of step2)

---

Respond ONLY with this exact JSON object (no other text, no markdown fences):
{
  "step1_entities": [
    {
      "name": "<entity name>",
      "claim_value": "<value stated in claim>",
      "evidence_value": "<value found in evidence, or 'not found'>",
      "correct": true or false
    }
  ],
  "step2_calculation": {
    "computed_value": "<exact value from verified calculator>",
    "claimed_value": "<last entity value from claim>",
    "match_type": "<exact|within_5pct|date_exact>",
    "correct": true or false
  },
  "step3_label": "<true|partially true|false>"
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


def _truncate_repetition(text: str, phrase_min_len: int = 20, threshold: int = 3) -> str:
    """
    Detect and truncate Qwen-style repetition loops (e.g. "All rights reserved." repeated).
    If a phrase of phrase_min_len+ chars repeats threshold+ times, cut before it starts.
    """
    pattern = re.compile(
        r'(.{' + str(phrase_min_len) + r',}?)\1{' + str(threshold - 1) + r',}',
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        return text[:m.start()].strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Logical Reasoner
# ─────────────────────────────────────────────────────────────────────────────
#
#  FACT_1 = all step1_entities have correct=True
#  FACT_2 = step2_calculation correct=True
#
#  FACT_1=True  AND FACT_2=True   →  "true"
#  FACT_1=True  AND FACT_2=False  →  "partially true"
#  FACT_1=False (any entity)      →  "false"
#
# ─────────────────────────────────────────────────────────────────────────────

def _logical_reasoner(parsed: dict) -> str:
    entities = parsed.get("step1_entities", [])
    calc     = parsed.get("step2_calculation", {})

    fact1 = bool(entities) and all(bool(e.get("correct", False)) for e in entities)
    fact2 = bool(calc.get("correct", False))

    if fact1 and fact2:
        return "true"
    elif fact1 and not fact2:
        return "partially true"
    else:
        return "false"


def _keyword_label(text: str) -> str:
    """
    Last-resort label extraction from free text.
    Priority: partially true > true > false.
    Defaults to "false" — never returns "unknown".
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
    return "false"   # safe default — never "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / model loading
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI model detection
# ─────────────────────────────────────────────────────────────────────────────

_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")
_OPENAI_MODEL_NAMES = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
}

def _is_openai_model(model_name: str) -> bool:
    name = model_name.lower().strip()
    if name in _OPENAI_MODEL_NAMES:
        return True
    return any(name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def load_model(model_name: str):
    if _is_openai_model(model_name):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)

        client = OpenAI(api_key=api_key)
        print(f"OpenAI client ready for model: {model_name}")
        return client, model_name

    # ── HuggingFace path ──────────────────────────────────────────────────────
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


def _chat(client_or_pipe, model_name, messages, temperature, max_tokens):
    # ── OpenAI path ───────────────────────────────────────────────────────────
    if _is_openai_model(model_name):
        kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        # o1/o3/o4 reasoning models do not support temperature
        if not any(model_name.lower().startswith(p) for p in ("o1", "o3", "o4")):
            kwargs["temperature"] = temperature
        response = client_or_pipe.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    # ── HuggingFace / pipeline path ───────────────────────────────────────────
    pipe = client_or_pipe
    do_sample = temperature > 0.0
    result = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        repetition_penalty=1.4,
        return_full_text=False,
    )
    raw = result[0]["generated_text"].strip()
    return _truncate_repetition(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Calculator ID resolver  (classifier name → integer ID)
# ─────────────────────────────────────────────────────────────────────────────

_CALC_NAME_TO_ID: dict = {}

_CLASSIFIER_ALIASES: dict = {
    # Wells' DVT (ID 16)
    "wells' criteria for dvt":                    16,
    "wells criteria for dvt":                     16,
    "wells' criteria for deep vein thrombosis":   16,
    "wells dvt criteria":                         16,
    "wells dvt score":                            16,
    "wells dvt":                                  16,
    # Wells' PE (ID 8)
    "wells' criteria for pulmonary embolism":     8,
    "wells criteria for pulmonary embolism":      8,
    "wells' criteria for pe":                     8,
    "wells' pe criteria":                         8,
    "wells pe score":                             8,
    "wells pe":                                   8,
    # Maintenance Fluids (ID 22)
    "maintenance fluids calculations":            22,
    "maintenance fluids calculation":             22,
    "maintenance fluid calculations":             22,
    "holliday-segar":                             22,
    "holliday segar":                             22,
    # Centor / McIsaac (ID 20)
    "centor score (modified/mcisaac)":            20,
    "centor score":                               20,
    "modified centor score":                      20,
    "mcisaac score":                              20,
    "centor/mcisaac score":                       20,
    # CCI (ID 32)
    "charlson comorbidity index":                 32,
    "cci score":                                  32,
    "cci":                                        32,
    # PSI (ID 29)
    "psi score":                                  29,
    "pneumonia severity index":                   29,
    "psi: pneumonia severity index":              29,
    # HEART (ID 18)
    "heart score":                                18,
    # Body Surface Area (ID 60)
    "body surface area":                          60,
    "body surface area calculator":               60,
    "body surface area (mosteller)":              60,
    "bsa (mosteller)":                            60,
    "bsa calculator":                             60,
    # Estimated Date of Conception (ID 68)
    "estimated date of conception":               68,
    "estimated of conception":                    68,
    "estimated conception date":                  68,
    "date of conception":                         68,
}


def _build_name_to_id() -> dict:
    mapping: dict = {}
    for line in _CALCULATOR_LIST.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"ID\s+(\d+)\s+[—–-]+\s+(.+?)(?::|$)", line)
        if m:
            mapping[m.group(2).strip().lower()] = int(m.group(1))
    return mapping


_MATCH_STOP = frozenset({
    'the', 'a', 'an', 'for', 'of', 'and', 'or', 'score', 'index',
    'calculation', 'calculations', 'calculator', 'criteria', 'modified',
    'based', 's', 'by', 'rule',
})


def _resolve_calculator_id(calc_name: str) -> int:
    global _CALC_NAME_TO_ID
    if not _CALC_NAME_TO_ID:
        _CALC_NAME_TO_ID = _build_name_to_id()

    key = calc_name.lower().strip()

    if key in _CALC_NAME_TO_ID:
        return _CALC_NAME_TO_ID[key]
    if key in _CLASSIFIER_ALIASES:
        return _CLASSIFIER_ALIASES[key]
    for name, cid in _CALC_NAME_TO_ID.items():
        if name in key or key in name:
            return cid

    key_words = set(re.findall(r'[a-z0-9]+', key)) - _MATCH_STOP
    best_cid, best_overlap = 0, 0
    for name, cid in _CALC_NAME_TO_ID.items():
        name_words = set(re.findall(r'[a-z0-9]+', name)) - _MATCH_STOP
        overlap = len(key_words & name_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cid = cid
    if best_overlap >= 3:
        return best_cid

    return 0  # unknown — run_calculator will fail gracefully


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: CoT extraction
# ─────────────────────────────────────────────────────────────────────────────

def step1_extract(client, model, item: dict, classifier: "FormulaClassifier",
                  temperature: float, max_tokens: int, ev_limit: int) -> dict:
    """
    Returns {"reasoning": str, "calculator_id": int, "calculator_name": str, "parameters": dict}
    Uses FormulaClassifier to decide which calculator to apply from the claim text.
    """
    predicted_calc_name = classifier.predict(item["claim"])
    calc_id = _resolve_calculator_id(predicted_calc_name)

    hint = f"The calculator for this case is: ID {calc_id} — {predicted_calc_name}"

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

    match = re.search(r"<extraction>([\s\S]*?)</extraction>", raw, re.IGNORECASE)
    json_str = match.group(1).strip() if match else raw
    json_str = re.sub(r"```(?:json)?|```", "", json_str).strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", json_str)
        try:
            parsed = json.loads(m.group()) if m else {}
        except Exception:
            parsed = {}

    params = parsed.get("parameters", parsed)

    return {
        "reasoning":       raw,
        "calculator_id":   calc_id,
        "calculator_name": predicted_calc_name,
        "parameters":      params if isinstance(params, dict) else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verdict — structured JSON → Logical Reasoner
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
        "Respond only with the JSON object."
    )

    raw = _chat(client, model,
                [{"role": "system", "content": STEP3_SYSTEM},
                 {"role": "user",   "content": user_msg}],
                temperature, max_tokens)

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # ── Primary path: structured JSON → logical reasoner ──────────────────
    try:
        parsed = json.loads(clean)
        label  = _logical_reasoner(parsed)

        # If step1_entities was empty, fall back to model's step3_label
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
        )
        return {"reasoning": reasoning_out, "label": label}

    # ── Fallback path: JSON failed — scan truncated text for keywords ──────
    except json.JSONDecodeError:
        return {
            "reasoning": clean,
            "label":     _keyword_label(clean),
        }


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
    if calc_errors:
        print(f"\nStep-2 calculator errors: {calc_errors}/{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoT + Classifier + External Calculator pipeline for medical NLI"
    )
    parser.add_argument("--data",       required=True,  help="Path to CSV dataset")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name: OpenAI (e.g. gpt-4o-mini) or HuggingFace path (default: gpt-4o-mini)")
    parser.add_argument("--samples",    type=int, default=10,
                        help="Number of samples (0 = all)")
    parser.add_argument("--random",     action="store_true", help="Randomly sample")
    parser.add_argument("--start",      type=int, default=1, help="1-based start index")
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
    parser.add_argument("--classifier", default="baseline_test/formula_classifier",
                        help="Path to trained FormulaClassifier directory "
                             "(default: baseline_test/formula_classifier)")
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
    classifier = FormulaClassifier(str(Path(args.classifier).resolve()))
    print("Formula classifier ready.\n")

    # ── Load LLM ──────────────────────────────────────────────────────────
    client, model = load_model(args.model)

    # ── Prepare output CSV ────────────────────────────────────────────────
    out_path = args.output or (
        f"results/cot_classifier_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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

        # ── Step 1: CoT extraction ────────────────────────────────────────
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

        item_with_calc = {**item,
                          "calculator_name": s1["calculator_name"],
                          "calculator_id":   s1["calculator_id"]}

        # ── Step 2: Python calculator ─────────────────────────────────────
        computed = run_calculator(s1["calculator_id"], s1["parameters"])
        print(f"  Step 2 computed: {computed['value']} {computed['unit']}"
              + (f"  [!] {computed['note']}" if computed.get("note") else ""))

        # ── Step 3: Verdict via logical reasoner ──────────────────────────
        s3 = _with_retry(
            step3_verdict,
            client, model, item_with_calc, computed,
            args.temperature, args.max_tokens_step3, args.ev_limit
        )
        if s3 is None:
            s3 = {"reasoning": "ERROR", "label": "false"}   # never "unknown"

        predicted = normalize_label(s3["label"])
        # Ensure predicted is always a valid label
        if predicted not in ("true", "partially true", "false"):
            predicted = "false"

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
            "step1_extracted_params":     json.dumps(s1["parameters"], ensure_ascii=False),
            "step2_computed_value":       str(computed.get("value", "")),
            "step2_unit":                 computed.get("unit", ""),
            "step2_error":                computed.get("note", ""),
            "step3_reasoning":            s3["reasoning"].replace("\n", " "),
            "ground_truth":               item["ground_truth"],
            "dataset_explanation":        item["explanation"].replace("\n", " "),
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
