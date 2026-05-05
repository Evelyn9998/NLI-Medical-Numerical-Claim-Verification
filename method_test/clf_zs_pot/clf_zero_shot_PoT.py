"""
LLM Fact-Check Evaluator — Classifier-guided Zero-shot Program of Thought (PoT)
================================================================================
Step 0 (Classifier): FormulaClassifier predicts the calculator name from the
                     claim text.  The prediction is injected as a hint into the
                     user message so the LLM knows which formula to apply.
Step 1-3 (PoT):      The LLM then writes a self-contained Python program that
                     works through STEP 1 → STEP 2 → STEP 3 and prints one JSON
                     object.  The host executes the program, captures its stdout,
                     and applies the same post-processing pipeline as the original
                     zero_shot_PoT.py — no external calculator is used.

Two system-prompt variants are kept:
  SYSTEM_PROMPT_WITH_CLF  — tells the LLM the calculator was pre-identified.
  SYSTEM_PROMPT_NO_CLF    — original wording; LLM self-identifies the calculator.
The correct one is selected per-sample based on whether a hint is available.
When classifier=None or loading fails the LLM operates exactly as it did in the
original zero_shot_PoT.py.

Host-side pipeline (unchanged from zero_shot_PoT.py):
  _remove_implicit_from_step1  →  strip the calculator output from step1
  _override_step2_by_criteria  →  enforce MedCalc-Bench accuracy tolerances
  _logical_reasoner            →  derive the final label from step1 + step2

Benchmark accuracy criteria (MedCalc-Bench §3.1, Table 2):
  - Rule-based calculators  (integer scores)  : exact match
  - Equation-based calculators (decimal output): within 5 %
  - Date-based calculators  (calendar date)   : exact match

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match |
  calculator_selected | model_reasoning | pot_code | dataset_explanation

Install dependencies:
    pip install transformers torch

Usage:
    python clf_zero_shot_PoT.py --data "mrt_claim_cleaned.csv" \\
        --model "$HOME/models/DeepSeek-R1" \\
        --classifier "./formula_classifier"

    python clf_zero_shot_PoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct"

    python clf_zero_shot_PoT.py --data "train_300.csv" \\
        --model "$HOME/models/Qwen2.5-14B-Instruct" --samples 20 --random

    python clf_zero_shot_PoT.py --data "train_300.csv" \\
        --model "$HOME/models/Qwen2.5-14B-Instruct" --no-think

    python clf_zero_shot_PoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0

    python clf_zero_shot_PoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct" --seed 42
"""

import argparse
import csv
import io
import json
import os
import re
import random
import sys
import time
import traceback
from datetime import datetime

# FormulaClassifier lives next to the original classifier script.
# Adjust this path if your project layout differs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_formula_classifier import FormulaClassifier, QTC_VARIANTS  # noqa: E402


# ---------------------------------------------------------------------------
# Calculator reference list  (copied from cot_classifier_external_calc.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Calculator name → ID resolver  (copied from cot_classifier_external_calc.py)
# ---------------------------------------------------------------------------

_CALC_NAME_TO_ID: dict = {}

_CALC_ALIASES: dict = {
    "wells' criteria for dvt": 16, "wells criteria for dvt": 16,
    "wells' criteria for deep vein thrombosis": 16, "wells dvt criteria": 16,
    "wells dvt score": 16, "wells dvt": 16,
    "wells' criteria for pulmonary embolism": 8, "wells criteria for pulmonary embolism": 8,
    "wells' criteria for pe": 8, "wells' pe criteria": 8, "wells pe score": 8, "wells pe": 8,
    "maintenance fluids calculations": 22, "maintenance fluids calculation": 22,
    "maintenance fluid calculations": 22, "holliday-segar": 22, "holliday segar": 22,
    "centor score (modified/mcisaac)": 20, "centor score": 20, "modified centor score": 20,
    "mcisaac score": 20, "centor/mcisaac score": 20,
    "charlson comorbidity index": 32, "cci score": 32, "cci": 32,
    "psi score": 29, "pneumonia severity index": 29, "psi: pneumonia severity index": 29,
    "heart score": 18,
    "body surface area": 60, "body surface area calculator": 60,
    "body surface area (mosteller)": 60, "bsa (mosteller)": 60, "bsa calculator": 60,
    "estimated date of conception": 68, "estimated of conception": 68,
    "estimated conception date": 68, "date of conception": 68,
}

_MATCH_STOP = frozenset({
    'the', 'a', 'an', 'for', 'of', 'and', 'or', 'score', 'index',
    'calculation', 'calculations', 'calculator', 'criteria', 'modified',
    'based', 's', 'by', 'rule',
})


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


def _resolve_calculator_id(calc_name: str) -> int:
    global _CALC_NAME_TO_ID
    if not _CALC_NAME_TO_ID:
        _CALC_NAME_TO_ID = _build_name_to_id()
    key = calc_name.lower().strip()
    if key in _CALC_NAME_TO_ID:
        return _CALC_NAME_TO_ID[key]
    if key in _CALC_ALIASES:
        return _CALC_ALIASES[key]
    for alias, cid in _CALC_ALIASES.items():
        if alias in key or key in alias:
            return cid
    for name, cid in _CALC_NAME_TO_ID.items():
        if name in key or key in name:
            return cid
    key_words = set(re.findall(r'[a-z0-9]+', key)) - _MATCH_STOP
    best_cid, best_overlap = 0, 0
    for name, cid in _CALC_NAME_TO_ID.items():
        name_words = set(re.findall(r'[a-z0-9]+', name)) - _MATCH_STOP
        overlap = len(key_words & name_words)
        if overlap > best_overlap:
            best_overlap, best_cid = overlap, cid
    if best_overlap >= 3:
        return best_cid
    return 0


# ---------------------------------------------------------------------------
# System prompts — instructs the LLM to emit executable Python (PoT)
#
# Two variants:
#   SYSTEM_PROMPT_WITH_CLF  — used when FormulaClassifier provided a hint.
#                             Tells the LLM the calculator was pre-identified.
#   SYSTEM_PROMPT_NO_CLF    — used when no classifier is available (identical
#                             to the original zero_shot_PoT.py prompt).
#                             The LLM self-identifies the calculator in STEP 2.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BODY = """Given clinical evidence and a clinical claim, write a self-contained Python
program that determines whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

The program must work through STEP 1 → STEP 2 → STEP 3, then print ONE JSON
object to stdout and nothing else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO SPLIT THE CLAIM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Split the claim on every comma ( , ).  Each segment is one entity.
  • Every segment EXCEPT the last  →  INPUT parameters  →  handled in STEP 1.
  • The LAST segment               →  final computed result  →  STEP 2 only.

EXAMPLE:
  Claim: "...the Calcium is 7.2 mg/dL, the Albumin is 3.2 g/dL,
           the Corrected Calcium Concentration is 7.84 mg/dL."
  Segments:  Calcium | Albumin | Corrected Calcium Concentration  ← last = STEP 2
  STEP 1 entities:   Calcium (7.2 mg/dL), Albumin (3.2 g/dL)
  STEP 2 calculation: Corrected Calcium Concentration = 7.84 mg/dL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Verify each INPUT entity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every INPUT segment (all except the last), look it up in the evidence and
set correct = True / False.

Special cases:
  • Positively-framed condition (e.g. "Liver disease criteria for HAS-BLED"):
      must be present in the evidence → False if not found.
  • Negatively-framed condition (e.g. "Cough Absent"):
      describes absence → True if the condition is not mentioned in the evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Calculate using evidence values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
a) Name the formula and write its expression as a Python comment.
b) Assign each input variable FROM THE EVIDENCE (the claim may be wrong).
c) Compute the result in Python and store it in `correct_result`.
d) Store the value stated in the claim's last segment in `claimed_result`.
e) Pick calculator_type: "rule-based" | "equation-based" | "date-based"
f) Apply the right accuracy criterion IN PYTHON:
     rule-based  / date-based  → exact match  (int or string equality)
     equation-based            → within 5 %   abs(correct - claimed)/abs(correct) ≤ 0.05

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Derive the label IN PYTHON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "true"          : ALL step1 entities correct=True  AND  step2 calc correct=True
  "partially true": ALL step1 entities correct=True  AND  step2 calc correct=False
  "false"         : ANY step1 entity has correct=False  (regardless of step2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your program must end with exactly one print(json.dumps({...})) call that
produces this JSON schema:

{
  "step1_entities": [
    {
      "name":           "<INPUT parameter name — never the calculator output>",
      "claim_value":    "<value stated in the claim>",
      "evidence_value": "<value from evidence, or 'not found'>",
      "correct":        true or false
    }
  ],
  "step2_calculation": {
    "calculator_type": "<rule-based | equation-based | date-based>",
    "formula":         "<formula name and expression>",
    "substitution":    "<formula with evidence values substituted>",
    "steps":           "<arithmetic shown as a Python expression or string>",
    "correct_result":  "<your computed answer, with units>",
    "claimed_result":  "<the last claim segment value, with units>",
    "correct":         true or false
  },
  "step3_label": "<true | partially true | false>"
}

Rules:
  • Import only standard-library modules (json, math, re, datetime, etc.).
  • Do NOT import numpy, pandas, or any third-party library.
  • Do NOT read files, make network calls, or use input().
  • Output ONLY the Python code — no prose, no markdown fences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMPLATE (adapt to the actual claim and evidence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json

# ── STEP 1: extract and verify every INPUT entity from the evidence ─────────
entities = []

# Segment 1 — "the Calcium is 7.2 mg/dL"
claim_calcium    = 7.2        # from claim
evidence_calcium = 7.2        # from evidence
correct_calcium  = (claim_calcium == evidence_calcium)
entities.append({
    "name": "Calcium", "claim_value": "7.2 mg/dL",
    "evidence_value": "7.2 mg/dL", "correct": correct_calcium
})

# Segment 2 — "the Albumin is 3.2 g/dL"
claim_albumin    = 3.2
evidence_albumin = 3.2
correct_albumin  = (claim_albumin == evidence_albumin)
entities.append({
    "name": "Albumin", "claim_value": "3.2 g/dL",
    "evidence_value": "3.2 g/dL", "correct": correct_albumin
})

# ── STEP 2: calculate using evidence values ─────────────────────────────────
# Formula: Corrected Calcium = Calcium + 0.8 × (4.0 − Albumin)
calculator_type = "equation-based"
formula         = "Corrected Calcium (mg/dL) = Calcium + 0.8 * (4.0 - Albumin)"
substitution    = f"{evidence_calcium} + 0.8 * (4.0 - {evidence_albumin})"

correct_result_val = evidence_calcium + 0.8 * (4.0 - evidence_albumin)
claimed_result_val = 7.84     # last claim segment

# Accuracy criterion — equation-based → within 5 %
if correct_result_val != 0:
    calc_correct = abs(correct_result_val - claimed_result_val) / abs(correct_result_val) <= 0.05
else:
    calc_correct = (claimed_result_val == 0)

calculation = {
    "calculator_type": calculator_type,
    "formula":         formula,
    "substitution":    substitution,
    "steps":           f"{substitution} = {correct_result_val:.4f}",
    "correct_result":  f"{correct_result_val:.2f} mg/dL",
    "claimed_result":  f"{claimed_result_val} mg/dL",
    "correct":         calc_correct,
}

# ── STEP 3: derive label ────────────────────────────────────────────────────
all_step1_correct = all(e["correct"] for e in entities)
if all_step1_correct and calc_correct:
    label = "true"
elif all_step1_correct and not calc_correct:
    label = "partially true"
else:
    label = "false"

print(json.dumps({
    "step1_entities":    entities,
    "step2_calculation": calculation,
    "step3_label":       label,
}))"""

SYSTEM_PROMPT_WITH_CLF = (
    "You are a medical fact-checking assistant.\n"
    "The calculator to use has been identified for you and is stated at the top of the "
    "user message. Use it in STEP 2 of your program.\n\n"
    + _SYSTEM_PROMPT_BODY
)

SYSTEM_PROMPT_NO_CLF = (
    "You are a medical fact-checking assistant.\n"
    + _SYSTEM_PROMPT_BODY
)


# ---------------------------------------------------------------------------
# Benchmark accuracy criteria — mirrors MedCalc-Bench §3.1 Table 2
# (used by the host-side override as a safety net on top of the LLM's code)
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
    """Re-evaluate step2 correct flag using MedCalc-Bench criteria."""
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

    else:  # tolerance
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
    """Strip the calculator's own output from step1_entities if it slipped in."""
    entities = parsed.get("step1_entities", [])
    if not entities:
        return parsed

    output_names: set = set()

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

    def _is_implicit(entity_name: str) -> bool:
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
# Logical Reasoner (host-side; mirrors the PoT label logic as a safety net)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PoT-specific: extract Python code & execute it
# ---------------------------------------------------------------------------

def _extract_pot_code(raw_text: str) -> str:
    """
    Pull the Python program out of the raw model output.

    Handles:
      1. Properly closed markdown fences (```python ... ```)
      2. Unclosed fences
      3. Raw code with no fences (anything after stripping think blocks)
    """
    # Strip think blocks (DeepSeek-R1 / Qwen3 style)
    raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    raw_text = re.sub(r"^[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    if re.search(r"<think>", raw_text, re.IGNORECASE):
        raw_text = re.sub(r"<think>[\s\S]*$", "", raw_text, flags=re.IGNORECASE).strip()

    raw_text = _truncate_repetition(raw_text)

    # Try to pull code from a fenced block
    fence_match = re.search(
        r"```(?:python)?\s*\n([\s\S]*?)```",
        raw_text,
        re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Unclosed fence — take everything after the opening ```
    unclosed = re.search(r"```(?:python)?\s*\n([\s\S]+)$", raw_text, re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()

    # No fences — return whatever is left (the model may have output raw code)
    return raw_text.strip()


_ALLOWED_IMPORTS = re.compile(
    r"^\s*import\s+(json|math|re|datetime|collections|functools|itertools|operator"
    r"|statistics|decimal|fractions|calendar|string|textwrap|enum|typing"
    r"|unicodedata|locale|numbers|cmath|random)\b",
    re.MULTILINE,
)
_FORBIDDEN_PATTERNS = re.compile(
    r"\b(open|exec|eval|compile|__import__|importlib|subprocess|os\.|sys\."
    r"|socket|urllib|requests|http|shutil|glob|pathlib|ctypes|threading"
    r"|multiprocessing|signal|resource|pty|atexit|builtins)\b",
    re.IGNORECASE,
)


def _is_safe_code(code: str) -> tuple:
    """
    Lightweight safety check before exec():
      - Reject any code that references dangerous builtins or I/O.
      - Allow only a curated set of standard-library imports.
    Returns (is_safe, reason).
    """
    import_lines = re.findall(r"^\s*import\s+\S+|^\s*from\s+\S+\s+import", code, re.MULTILINE)
    for line in import_lines:
        if not _ALLOWED_IMPORTS.match(line.strip() + "\n"):
            return False, f"Disallowed import: {line.strip()}"

    if _FORBIDDEN_PATTERNS.search(code):
        return False, "Forbidden built-in or module reference detected."

    return True, ""


def _execute_pot_code(code: str) -> dict | None:
    """
    Execute the LLM-generated Python program in an isolated namespace.
    Captures stdout, parses the printed JSON, and returns the parsed dict.
    Returns None on any error.
    """
    safe, reason = _is_safe_code(code)
    if not safe:
        print(f"  [PoT] Code rejected by safety check: {reason}")
        return None

    namespace: dict = {"__builtins__": __builtins__}
    captured = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = captured
        exec(compile(code, "<pot_program>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        sys.stdout = old_stdout
        print(f"  [PoT] Execution error: {exc}")
        traceback.print_exc()
        return None
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue().strip()
    if not output:
        print("  [PoT] Program produced no output.")
        return None

    # The last line should be valid JSON; try progressively shorter suffixes
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # Fallback: try the entire captured output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print(f"  [PoT] Could not parse JSON from output:\n{output[:300]}")
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _keyword_label(text: str) -> str:
    """Last-resort label extraction when PoT execution fails entirely."""
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
    return "false"


def _parse_response(raw_text: str, claim: str = "") -> dict:
    """
    PoT response pipeline:

      1. Extract Python code from the raw model output (strip think blocks,
         markdown fences, etc.).
      2. Safety-check the code.
      3. Execute the code; capture the printed JSON.
      4. _remove_implicit_from_step1  — strip the calculator output from step1.
      5. _override_step2_by_criteria  — enforce benchmark accuracy tolerance.
      6. _logical_reasoner            — derive the final label.
      7. Fallback: keyword scan if execution fails.

    Returns {"reasoning": str, "label": str, "pot_code": str}.
    Note: "calculator_selected" is attached by the caller (run_batch / retry path).
    """
    code   = _extract_pot_code(raw_text)
    parsed = _execute_pot_code(code) if code else None

    if parsed is not None:
        # Normalise alternative key names (robustness against slight schema drift)
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

        # If neither block is present, fall back on the raw label from the JSON
        if not parsed.get("step1_entities") and not parsed.get("step2_calculation"):
            label = normalize_label(parsed.get("step3_label", ""))
            if label not in ("true", "partially true", "false"):
                label = _keyword_label(code)
            return {"reasoning": code, "label": label, "pot_code": code}

        parsed = _remove_implicit_from_step1(parsed, claim)
        parsed = _override_step2_by_criteria(parsed)

        label = _logical_reasoner(parsed)

        # Prefer the Python-derived label; fall back to the JSON label only when
        # step1_entities is empty (e.g. the claim has no comma-separated inputs).
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
        return {"reasoning": reasoning_out, "label": label, "pot_code": code}

    # ── Execution failed — keyword scan over the raw model output ──────────
    return {
        "reasoning": raw_text[:2000],
        "label":     _keyword_label(raw_text),
        "pot_code":  code,
    }


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
# Model loading — always fp16, no 4-bit quantisation, no CPU fp32
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_gpu = torch.cuda.is_available()
    print(f"  GPU available: {has_gpu}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto" if has_gpu else "cpu",
        low_cpu_mem_usage=True,
    )

    if hasattr(model, "generation_config") and model.generation_config.max_length == 20:
        model.generation_config.max_length = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
    )
    print(f"Model ready: {model_name}  (fp16, batch_size=1)")
    return {"pipeline": pipe, "model_name": model_name, "pad_token_id": tokenizer.pad_token_id}


# ---------------------------------------------------------------------------
# Classifier helper
# ---------------------------------------------------------------------------

def _classify_claim(classifier: "FormulaClassifier", claim: str) -> str:
    """
    Run the FormulaClassifier on a single claim and return the predicted
    calculator name.  Returns an empty string if classifier is None.
    """
    if classifier is None:
        return ""
    try:
        return classifier.predict(claim)
    except Exception as e:
        print(f"  [classifier warning] {e}")
        return ""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(item: dict, ev_limit: int,
                    no_think: bool = False,
                    calculator_hint: str = "") -> list:
    """
    Build the chat message list for one sample.

    calculator_hint : when non-empty, prepended to the user message so the LLM
                      knows which calculator to apply in STEP 2 of its program.
                      Also selects SYSTEM_PROMPT_WITH_CLF over SYSTEM_PROMPT_NO_CLF.
    no_think        : prepends '/no_think' for Qwen3 thinking models to skip the
                      <think> block and output the program directly.
    """
    hint_line = (
        f"Calculator identified by classifier: {calculator_hint}\n\n"
        if calculator_hint else ""
    )
    user_msg = (
        f"{hint_line}"
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        "Output ONLY the Python program. No prose, no markdown fences."
    )
    if no_think:
        user_msg = "/no_think\n" + user_msg

    system_prompt = SYSTEM_PROMPT_WITH_CLF if calculator_hint else SYSTEM_PROMPT_NO_CLF

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Batched inference — batch_size=1, greedy decoding
# ---------------------------------------------------------------------------

def run_batch(
    model: dict,
    batch: list,
    ev_limit: int,
    max_tokens: int,
    no_think: bool = False,
    classifier: "FormulaClassifier" = None,
) -> list:
    """
    Run inference on a list of items one sample at a time (batch_size=1).
    Greedy decoding is pinned unconditionally (do_sample=False).

    Returns a list of dicts:
      {"reasoning": str, "label": str, "pot_code": str, "calculator_selected": str}
    """
    from transformers import GenerationConfig

    pipe = model["pipeline"]
    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=model.get("pad_token_id") or pipe.tokenizer.pad_token_id,
    )

    results = []
    for item in batch:
        # Step 0: run classifier to identify the calculator
        calc_name = _classify_claim(classifier, item["claim"])
        messages  = _build_messages(item, ev_limit,
                                    no_think=no_think,
                                    calculator_hint=calc_name)

        out = pipe(
            messages,
            generation_config=gen_cfg,
            return_full_text=False,
        )
        raw  = out[0]["generated_text"].strip()
        resp = _parse_response(raw, item.get("claim", ""))
        resp["calculator_selected"] = calc_name
        results.append(resp)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    n = len(results)
    if n == 0:
        return

    correct = sum(1 for r in results if r["match"])
    exec_ok = sum(1 for r in results if r.get("exec_ok", False))
    print(f"\n=== Summary ===")
    print(f"Total         : {n}")
    print(f"Correct       : {correct}")
    print(f"Wrong         : {n - correct}")
    print(f"Accuracy      : {correct / n * 100:.1f}%")
    print(f"PoT exec OK   : {exec_ok}/{n}  ({exec_ok/n*100:.0f}%)")

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
            "HuggingFace evaluator — Classifier-guided Zero-shot Fact-Check with "
            "Program of Thought (PoT) + Logical Reasoner.  The FormulaClassifier "
            "identifies the calculator; the LLM then writes Python code that works "
            "through STEP 1→2→3 by itself.  The host executes the code and derives "
            "the final label.  Compatible with DeepSeek-R1, Llama-3, and Qwen.  "
            "When --classifier is omitted or loading fails, the LLM self-identifies "
            "the calculator exactly as in zero_shot_PoT.py."
        )
    )
    parser.add_argument("--data", required=True,
                        help="Path to CSV dataset, e.g. mrt_claim_cleaned.csv")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--classifier", default="./formula_classifier",
                        help=(
                            "Path to the trained FormulaClassifier directory "
                            "(default: ./formula_classifier). "
                            "Pass an empty string to run without a classifier."
                        ))
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to evaluate (default: 10; 0 = all)")
    parser.add_argument("--random", action="store_true",
                        help="Randomly sample instead of taking from the start")
    parser.add_argument("--ev-limit", type=int, default=2000,
                        help="Max characters of Evidence passed to model (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help=(
                            "Global RNG seed for Python random, NumPy, and PyTorch "
                            "(default: 42).  Governs --random sample selection and "
                            "any internal stochastic ops."
                        ))
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help=(
                            "Max output tokens per inference (default: 4096). "
                            "PoT programs are typically shorter than CoT chains, "
                            "so 2048 is usually sufficient.  Increase to 8192 for "
                            "Qwen thinking models without --no-think."
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
                            "Disables the <think> block for Qwen3 thinking models. "
                            "Has no effect on Llama or other non-thinking models."
                        ))
    args = parser.parse_args()

    if args.random and args.start != 1:
        print(f"Warning: --start {args.start} is ignored when --random is set.")
    if args.no_think:
        print("--no-think enabled: prepending '/no_think' to all user messages.")

    # Lock down randomness
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
    print(f"Global seed set to {_seed}")

    # -- Load classifier ---------------------------------------------------
    classifier = None
    if args.classifier:
        print(f"Loading formula classifier from: {args.classifier}")
        try:
            classifier = FormulaClassifier(args.classifier)
            print("Formula classifier ready.\n")
        except Exception as e:
            print(f"Warning: could not load classifier ({e}). "
                  "Proceeding without calculator hint.\n")
    else:
        print("No classifier path provided — running without calculator hint.\n")

    # -- Load data -----------------------------------------------------------
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

    # -- Load model ----------------------------------------------------------
    model = load_model(args.model)

    # -- Prepare output CSV --------------------------------------------------
    out_path = args.output or f"results_clf_pot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "calculator_selected", "model_reasoning", "pot_code",
        "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # -- Evaluation loop -----------------------------------------------------
    results = []
    total   = len(subset)
    print(f"Starting PoT evaluation ({total} samples, greedy decoding, fp16)...\n")

    for batch_start in range(0, total, 1):
        batch      = subset[batch_start:batch_start + 1]
        batch_idxs = [args.start + batch_start]

        print(f"[{batch_idxs[0]}/{args.start + total - 1}] Running...")

        try:
            resps = run_batch(
                model, batch, args.ev_limit,
                args.max_tokens,
                no_think=args.no_think,
                classifier=classifier,
            )
        except Exception as e:
            print(f"  Error ({e}), retrying up to 3 times...")
            resps = []
            for item in batch:
                # Classify once before the retry loop so all attempts reuse the
                # same classifier output.
                calc_name = _classify_claim(classifier, item["claim"])
                print(f"  Classifier selected: {calc_name or '(none)'}")

                for attempt in range(3):
                    try:
                        msgs = _build_messages(item, args.ev_limit,
                                               no_think=args.no_think,
                                               calculator_hint=calc_name)
                        pipe = model["pipeline"]
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
                        resp = _parse_response(raw, item.get("claim", ""))
                        resp["calculator_selected"] = calc_name
                        resps.append(resp)
                        break
                    except Exception as e2:
                        print(f"    [attempt {attempt+1}/3] Error: {e2}")
                        if attempt < 2:
                            time.sleep(30 * (attempt + 1))
                        else:
                            resps.append({
                                "reasoning":           f"Error: {e2}",
                                "label":               "false",
                                "pot_code":            "",
                                "calculator_selected": calc_name,
                            })

        # Write results for this sample
        with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for item, idx, resp in zip(batch, batch_idxs, resps):
                predicted = normalize_label(resp["label"])
                if predicted not in ("true", "partially true", "false"):
                    predicted = "false"

                match         = predicted == item["label"]
                exec_ok       = bool(resp.get("pot_code", ""))
                calc_selected = resp.get("calculator_selected", "")
                print(f"  [{idx}] calculator={calc_selected or '(none)'}  "
                      f"true={item['label']}  predicted={predicted}  match={match}")

                row = {
                    "index":       idx,
                    "evidence":    item["evidence"],
                    "claim":       item["claim"],
                    "true_label":  item["label"],
                    "predicted":   predicted,
                    "match":       match,
                    "exec_ok":     exec_ok,
                    "calc":        calc_selected,
                    "reasoning":   resp["reasoning"],
                    "pot_code":    resp.get("pot_code", ""),
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
                    "calculator_selected": row["calc"],
                    "model_reasoning":     row["reasoning"].replace("\n", " "),
                    "pot_code":            row["pot_code"].replace("\n", "\\n"),
                    "dataset_explanation": row["explanation"].replace("\n", " "),
                })

        if args.delay > 0 and batch_start + 1 < total:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
