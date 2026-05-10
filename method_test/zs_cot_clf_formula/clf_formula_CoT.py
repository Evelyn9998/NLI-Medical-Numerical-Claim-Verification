"""
LLM Fact-Check Evaluator — Classifier-guided Formula-lookup Chain of Thought (CoT)
====================================================================================
Step 0 (Classifier): FormulaClassifier predicts the calculator name from the
                     claim text.

Step 0b (Formula lookup): When the classifier returns a non-null prediction,
                     the canonical formula for that calculator is looked up
                     from formula_cleaned.json (Calculator ID → Formula text).
                     The formula is injected verbatim into the user message as
                     a hard constraint so the LLM cannot substitute an
                     alternative formula variant.  A unit-conversion reminder
                     is appended so values are converted before substitution.

Step 1-3 (CoT):      The LLM reasons step-by-step through STEP 1 → STEP 2 →
                     STEP 3 in natural language and outputs ONE JSON object
                     directly.  The host applies the same post-processing
                     pipeline as the PoT variant.

formula_cleaned.json is produced from formula_new.json with the Question field
removed — only Calculator ID and Formula are kept.

Host-side pipeline:
  _remove_implicit_from_step1  →  strip the calculator output from step1
  _override_step2_by_criteria  →  enforce MedCalc-Bench accuracy tolerances
  _logical_reasoner            →  derive the final label from step1 + step2

Benchmark accuracy criteria (MedCalc-Bench §3.1, Table 2):
  - Rule-based calculators  (integer scores)  : exact match
  - Equation-based calculators (decimal output): within 5 %
  - Date-based calculators  (calendar date)   : exact match

Output CSV columns:
  index | evidence | claim | true_label | predicted_label | match |
  calculator_selected | model_reasoning | dataset_explanation

Usage:
    python clf_formula_CoT.py --data "mrt_claim_cleaned.csv" \\
        --model "$HOME/models/DeepSeek-R1" \\
        --classifier "./formula_classifier" \\
        --formula-data "formula_cleaned.json"

    python clf_formula_CoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct"

    python clf_formula_CoT.py --data "train_300.csv" \\
        --model "$HOME/models/Qwen2.5-14B-Instruct" --samples 20 --random

    python clf_formula_CoT.py --data "train_300.csv" \\
        --model "$HOME/models/Qwen2.5-14B-Instruct" --no-think

    python clf_formula_CoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct" --samples 0

    python clf_formula_CoT.py --data "train_300.csv" \\
        --model "$HOME/models/Llama-3.1-8B-Instruct" --seed 42
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
# Formula lookup — maps Calculator ID → canonical formula text
# ---------------------------------------------------------------------------

class FormulaLoader:
    """
    Loads formula_cleaned.json at construction time and indexes entries by
    Calculator ID for O(1) lookup at inference time.

    formula_cleaned.json entries have three keys:
        "Calculator ID"  (string or int)
        "Type"           (string — "Rule-Based" or "Equation-Based",
                          as defined in MedCalc-Bench Table 9)
        "Formula"        (string — the canonical formula description)

    The "Question" field from formula_new.json has been deliberately removed.
    """

    def __init__(self, json_path: str) -> None:
        self._formulas: dict = {}   # {calc_id (int): formula_text (str)}
        self._types:    dict = {}   # {calc_id (int): type_str (str)}

        if not json_path or not os.path.exists(json_path):
            print(f"[FormulaLoader] File not found: {json_path!r}. "
                  "Formula injection will not be used.")
            return

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            try:
                calc_id = int(entry["Calculator ID"])
            except (KeyError, ValueError):
                continue
            formula = (entry.get("Formula") or "").strip()
            calc_type = (entry.get("Type") or "").strip()
            if formula:
                self._formulas[calc_id] = formula
            if calc_type:
                self._types[calc_id] = calc_type

        print(f"[FormulaLoader] Loaded {len(self._formulas)} formulas "
              f"({sum(1 for t in self._types.values() if t == 'Rule-Based')} Rule-Based, "
              f"{sum(1 for t in self._types.values() if t == 'Equation-Based')} Equation-Based) "
              f"from {json_path!r}.")

    def get_formula(self, calc_id: int) -> str:
        """Return the formula text for *calc_id*, or '' if not found."""
        return self._formulas.get(calc_id, "")

    def get_type(self, calc_id: int) -> str:
        """Return 'Rule-Based', 'Equation-Based', or '' if not found."""
        return self._types.get(calc_id, "")

    def is_rule_based(self, calc_id: int) -> bool:
        """Return True when the calculator type is 'Rule-Based'."""
        return self._types.get(calc_id, "") == "Rule-Based"

    def is_empty(self) -> bool:
        return not self._formulas


# ---------------------------------------------------------------------------
# System prompts — instructs the LLM to reason step-by-step (CoT) and emit
# a single JSON object.
#
# Two variants:
#   SYSTEM_PROMPT_WITH_CLF  — used when FormulaClassifier provided a hint.
#                             Tells the LLM the calculator was pre-identified.
#   SYSTEM_PROMPT_NO_CLF    — used when no classifier is available (identical
#                             to the original zero_shot_CoT.py prompt).
#                             The LLM self-identifies the calculator in STEP 2.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BODY = """Given clinical evidence and a clinical claim, reason step by step to
determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.

Work through STEP 1 → STEP 2 → STEP 3, then output ONE JSON object and nothing else.

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
set correct = true / false.

Special cases:
  • Positively-framed condition (e.g. "Liver disease criteria for HAS-BLED"):
      must be present in the evidence → false if not found.
  • Negatively-framed condition (e.g. "Cough Absent"):
      describes absence → true if the condition is not mentioned in the evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Calculate using evidence values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
a) Name the formula and write its expression.
b) Identify each input variable FROM THE EVIDENCE (the claim may be wrong).
c) Show the arithmetic step by step and compute the result.
d) Note the value stated in the claim's last segment as claimed_result.
e) Pick calculator_type: "rule-based" | "equation-based" | "date-based"
f) Apply the right accuracy criterion:
     rule-based  / date-based  → exact match  (integer or string equality)
     equation-based            → within 5 %   |correct − claimed| / |correct| ≤ 0.05

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Derive the label
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "true"          : ALL step1 entities correct=true  AND  step2 calc correct=true
  "partially true": ALL step1 entities correct=true  AND  step2 calc correct=false
  "false"         : ANY step1 entity has correct=false  (regardless of step2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY the following JSON object — no prose before or after it:

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
    "steps":           "<arithmetic shown step by step>",
    "correct_result":  "<your computed answer, with units>",
    "claimed_result":  "<the last claim segment value, with units>",
    "correct":         true or false
  },
  "step3_label": "<true | partially true | false>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE OUTPUT (adapt to the actual claim and evidence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "step1_entities": [
    {
      "name": "Calcium",
      "claim_value": "7.2 mg/dL",
      "evidence_value": "7.2 mg/dL",
      "correct": true
    },
    {
      "name": "Albumin",
      "claim_value": "3.2 g/dL",
      "evidence_value": "3.2 g/dL",
      "correct": true
    }
  ],
  "step2_calculation": {
    "calculator_type": "equation-based",
    "formula": "Corrected Calcium (mg/dL) = Calcium + 0.8 * (4.0 - Albumin)",
    "substitution": "7.2 + 0.8 * (4.0 - 3.2)",
    "steps": "7.2 + 0.8 * 0.8 = 7.2 + 0.64 = 7.84",
    "correct_result": "7.84 mg/dL",
    "claimed_result": "7.84 mg/dL",
    "correct": true
  },
  "step3_label": "true"
}"""

SYSTEM_PROMPT_WITH_CLF = (
    "You are a medical fact-checking assistant.\n"
    "The calculator to use has been identified for you and is stated at the top of the "
    "user message. Use it in STEP 2 of your reasoning.\n\n"
    + _SYSTEM_PROMPT_BODY
)

SYSTEM_PROMPT_NO_CLF = (
    "You are a medical fact-checking assistant.\n"
    + _SYSTEM_PROMPT_BODY
)


# ---------------------------------------------------------------------------
# Benchmark accuracy criteria — mirrors MedCalc-Bench §3.1 Table 2
# (used by the host-side override as a safety net on top of the LLM's reasoning)
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
# Logical Reasoner (host-side; mirrors the CoT label logic as a safety net)
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
# CoT-specific: extract JSON from the model's natural-language output
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> dict | None:
    """
    Extract the first valid JSON object from the raw model output.

    Handles:
      1. JSON inside a markdown fence (```json ... ``` or ``` ... ```)
      2. Raw JSON object embedded in prose
      3. The entire output being a bare JSON object
    """
    # Try a properly closed markdown fence first
    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find the first '{' and walk to its matching '}'
    brace_start = text.find('{')
    if brace_start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    break

    # Last resort: try the whole remaining text as JSON
    try:
        return json.loads(text[brace_start:].strip())
    except json.JSONDecodeError:
        return None


def _validate_rule_based_score(parsed: dict, formula_text: str = "",
                               calc_type: str = "") -> dict:
    """
    Post-reasoning sanity check for rule-based calculators.

    Uses *calc_type* (from FormulaLoader.get_type()) as the authoritative
    source to determine whether the calculator is rule-based.  Falls back to
    text heuristics on *formula_text* when calc_type is unavailable.

    Problem: the LLM sometimes assigns 1 to every criterion without evaluating
    threshold conditions, producing a substitution like ``1 + 1 + 1 + 1 + 1``
    regardless of evidence.

    Detection: the substitution string is composed entirely of bare integer /
    float literals — no variable names, no comparison operators, no function
    calls.  This means no evidence value was actually used in the scoring.

    Action: when the substitution is all-literals AND the computed score does
    not match the claimed score, mark ``step2_calculation.correct = False``
    so ``_logical_reasoner`` returns "partially true" rather than "true".
    If the computed score *does* match the claimed score, leave ``correct``
    untouched — the answer happens to be right even if the reasoning was wrong.
    """
    if not _is_rule_based_formula(formula_text, calc_type):
        return parsed

    calc = parsed.get("step2_calculation", {})
    substitution = str(calc.get("substitution", "")).strip()
    if not substitution:
        return parsed

    tokens = re.split(r'[\s\+\-\*\/\(\)\,]+', substitution)
    tokens = [t.strip() for t in tokens if t.strip()]

    if not tokens:
        return parsed

    def _is_numeric_literal(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    all_literals = all(_is_numeric_literal(t) for t in tokens)
    if not all_literals:
        return parsed

    cr_str  = str(calc.get("correct_result",  "")).strip()
    clm_str = str(calc.get("claimed_result", "")).strip()

    cr_m  = re.search(r'[\-\d.]+', cr_str)
    clm_m = re.search(r'[\-\d.]+', clm_str)

    if cr_m and clm_m:
        try:
            cr_val  = float(cr_m.group())
            clm_val = float(clm_m.group())
            scores_match = (cr_val == clm_val)
        except ValueError:
            scores_match = False
    else:
        scores_match = (cr_str == clm_str)

    if not scores_match:
        print(f"  [rule-based validator] Substitution '{substitution}' "
              f"contains only literals — LLM did not evaluate thresholds. "
              f"Marking step2 correct=False.")
        calc["correct"] = False
    else:
        print(f"  [rule-based validator] Substitution '{substitution}' "
              f"contains only literals but score matches claimed value "
              f"({cr_val} == {clm_val}) — keeping correct=True.")

    return parsed


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _keyword_label(text: str) -> str:
    """Last-resort label extraction when JSON parsing fails entirely."""
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


def _parse_response(raw_text: str, claim: str = "",
                    formula_text: str = "",
                    calc_type: str = "") -> dict:
    """
    CoT response pipeline:

      1. Strip think blocks (DeepSeek-R1 / Qwen3 style) and truncate repetition.
      2. Extract the JSON object from the model's natural-language output.
      3. _remove_implicit_from_step1  — strip the calculator output from step1.
      4. _validate_rule_based_score   — catch all-literal substitutions.
      5. _override_step2_by_criteria  — enforce benchmark accuracy tolerance.
      6. _logical_reasoner            — derive the final label.
      7. Fallback: keyword scan if JSON extraction fails.

    Returns {"reasoning": str, "label": str}.
    Note: "calculator_selected" is attached by the caller (run_batch / retry path).
    """
    # Strip think blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    if re.search(r"<think>", text, re.IGNORECASE):
        text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE).strip()

    text = _truncate_repetition(text)

    parsed = _extract_json_from_text(text)

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
                label = _keyword_label(text)
            return {"reasoning": text[:2000], "label": label}

        parsed = _remove_implicit_from_step1(parsed, claim)
        parsed = _validate_rule_based_score(parsed, formula_text, calc_type)
        parsed = _override_step2_by_criteria(parsed)

        label = _logical_reasoner(parsed)

        # Prefer the host-derived label; fall back to the JSON label only when
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
        return {"reasoning": reasoning_out, "label": label}

    # ── JSON extraction failed — keyword scan over the raw model output ─────
    return {
        "reasoning": raw_text[:2000],
        "label":     _keyword_label(raw_text),
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
# Rule-based formula helpers
# ---------------------------------------------------------------------------

def _is_rule_based_formula(formula_text: str = "", calc_type: str = "") -> bool:
    """
    Return True when the calculator is rule-based.

    Priority:
    1. Use *calc_type* when it is set (looked up from FormulaLoader.get_type()).
       This is the authoritative source — taken directly from MedCalc-Bench
       Table 9 via the "Type" field added to formula_cleaned.json.
    2. Fall back to text heuristics on *formula_text* only when calc_type is
       unavailable (e.g. the loader was not provided or the ID was not found).
       Heuristic: presence of "point" or "criteria" in the formula text.
    """
    if calc_type:
        return calc_type.strip().lower() == "rule-based"
    # Legacy heuristic fallback
    t = formula_text.lower()
    return "point" in t or "criteria" in t


_RULE_BASED_SCORING_HINT = """\
MANDATORY RULE-BASED SCORING PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is a rule-based (criteria-driven) calculator. EVERY criterion MUST be \
evaluated by checking a threshold condition against the actual evidence value. \
Never assign points without checking the threshold.

Use one of these three patterns for each criterion:

  Pattern A — binary (Yes/No, Present/Absent):
      Check whether the condition is met, then assign the weight.
      e.g.: confusion_points = 1 if confusion is present, else 0
            dvt_signs_points = 3 if DVT signs are present, else 0

  Pattern B — tiered range (multiple score levels):
      Check the evidence value against each threshold in descending order.
      e.g. for Age in CHA₂DS₂-VASc:
        if age ≥ 75 → 2 points
        elif age ≥ 65 → 1 point
        else → 0 points

  Pattern C — direct numeric value (only when the formula says "enter X in years/units"):
      Use the raw evidence value directly.
      e.g. PSI age criterion: age_points = patient_age

FORBIDDEN patterns:
  Assigning a literal 1 to every criterion without checking its condition.
  Using a raw evidence value for a binary (Yes/No) criterion.
  A substitution field that contains only bare integers (e.g. "1 + 1 + 1 + 1").

The substitution field must show the actual threshold-checked expressions \
or condition outcomes — NOT bare integers like "1 + 1 + 1 + 1 + 1".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(item: dict, ev_limit: int,
                    no_think: bool = False,
                    calculator_hint: str = "",
                    formula_text: str = "",
                    calc_type: str = "") -> list:
    """
    Build the chat message list for one sample.

    calculator_hint : when non-empty, prepended to the user message so the LLM
                      knows which calculator to apply in STEP 2 of its reasoning.
                      Also selects SYSTEM_PROMPT_WITH_CLF over SYSTEM_PROMPT_NO_CLF.
    formula_text    : canonical formula looked up from formula_cleaned.json for
                      the predicted calculator.  Injected as a hard constraint
                      directly after the calculator hint so the LLM cannot
                      substitute an alternative formula variant.  A unit-
                      conversion reminder is appended.  Only injected when
                      calculator_hint is also set.
    calc_type       : "Rule-Based" or "Equation-Based" from FormulaLoader.get_type().
                      When "Rule-Based", the mandatory boolean-pattern hint is
                      appended to the formula section.  Falls back to text
                      heuristics on formula_text when calc_type is empty.
    no_think        : prepends '/no_think' for Qwen3 thinking models to skip the
                      <think> block and output the JSON directly.
    """
    hint_line = (
        f"Calculator identified by classifier: {calculator_hint}\n\n"
        if calculator_hint else ""
    )

    if calculator_hint and formula_text:
        is_rb = _is_rule_based_formula(formula_text, calc_type)
        rb_hint = ("\n" + _RULE_BASED_SCORING_HINT) if is_rb else ""
        formula_section = (
            f"REQUIRED FORMULA (use this exactly — do not substitute an "
            f"alternative variant):\n{formula_text}{rb_hint}\n"
            f"UNIT REQUIREMENT: Convert all input values to the units expected "
            f"by the formula before substituting (e.g. cm → inches for IBW, "
            f"cm → m for BMI). Show each conversion step explicitly in your "
            f"substitution.\n\n"
        )
    else:
        formula_section = ""

    user_msg = (
        f"{hint_line}"
        f"{formula_section}"
        f"Evidence:\n{item['evidence'][:ev_limit]}\n\n"
        f"Claim:\n{item['claim']}\n\n"
        "Output ONLY the JSON object. No prose, no markdown fences."
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
    formula_loader: "FormulaLoader" = None,
) -> list:
    """
    Run inference on a list of items one sample at a time (batch_size=1).
    Greedy decoding is pinned unconditionally (do_sample=False).

    When both *classifier* and *formula_loader* are provided and the classifier
    returns a non-null prediction, the canonical formula for the identified
    calculator is looked up and injected into the user message as a hard
    constraint, preventing the LLM from inventing alternative formula variants.

    Returns a list of dicts:
      {"reasoning": str, "label": str, "calculator_selected": str}
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

        # Step 0b: look up the canonical formula and type for the predicted calculator
        formula_text = ""
        calc_type    = ""
        if calc_name and formula_loader is not None:
            calc_id      = _resolve_calculator_id(calc_name)
            formula_text = formula_loader.get_formula(calc_id)
            calc_type    = formula_loader.get_type(calc_id)
            if formula_text:
                print(f"  [formula] {calc_name} (ID {calc_id}, {calc_type}): "
                      f"{formula_text[:80]}{'...' if len(formula_text) > 80 else ''}")
            else:
                print(f"  [formula] No entry for '{calc_name}' (ID {calc_id})")

        messages = _build_messages(item, ev_limit,
                                   no_think=no_think,
                                   calculator_hint=calc_name,
                                   formula_text=formula_text,
                                   calc_type=calc_type)

        out = pipe(
            messages,
            generation_config=gen_cfg,
            return_full_text=False,
        )
        raw  = out[0]["generated_text"].strip()
        resp = _parse_response(raw, item.get("claim", ""),
                               formula_text=formula_text,
                               calc_type=calc_type)
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
    print(f"\n=== Summary ===")
    print(f"Total         : {n}")
    print(f"Correct       : {correct}")
    print(f"Wrong         : {n - correct}")
    print(f"Accuracy      : {correct / n * 100:.1f}%")

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
            "Chain of Thought (CoT) + Logical Reasoner.  The FormulaClassifier "
            "identifies the calculator; the LLM then reasons step-by-step through "
            "STEP 1→2→3 and outputs a structured JSON object directly.  The host "
            "parses the JSON and derives the final label.  Compatible with "
            "DeepSeek-R1, Llama-3, and Qwen.  When --classifier is omitted or "
            "loading fails, the LLM self-identifies the calculator exactly as in "
            "zero_shot_CoT.py."
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
                            "CoT reasoning chains are typically longer than bare "
                            "JSON; increase to 8192 for Qwen thinking models "
                            "without --no-think."
                        ))
    parser.add_argument("--start", type=int, default=1,
                        help="1-based index to start from (default: 1)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between samples (default: 0.0)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: auto-generated with timestamp)")
    parser.add_argument("--formula-data", default="formula_cleaned.json",
                        help=(
                            "Path to formula_cleaned.json (default: formula_cleaned.json). "
                            "When provided AND the classifier returns a non-null prediction, "
                            "the canonical formula for that calculator is injected into the "
                            "user message as a hard constraint. Pass an empty string to run "
                            "without formula injection."
                        ))
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

    # -- Load formula lookup -----------------------------------------------
    formula_loader = None
    if args.formula_data:
        formula_loader = FormulaLoader(args.formula_data)
        if formula_loader.is_empty():
            print("Warning: formula loader is empty — no formulas will be injected.\n")
        elif classifier is None:
            print("Note: formula data loaded but classifier is unavailable. "
                  "Formulas will only be injected when the classifier provides "
                  "a non-null prediction.\n")
    else:
        print("No formula data path provided — running without formula injection.\n")

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
    out_path = args.output or f"results_clf_cot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "index", "evidence", "claim", "true_label", "predicted_label",
        "match", "calculator_selected", "model_reasoning", "dataset_explanation",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    print(f"Writing results live to: {out_path}\n")

    # -- Evaluation loop -----------------------------------------------------
    results = []
    total   = len(subset)
    print(f"Starting CoT evaluation ({total} samples, greedy decoding, fp16)...\n")

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
                formula_loader=formula_loader,
            )
        except Exception as e:
            print(f"  Error ({e}), retrying up to 3 times...")
            resps = []
            for item in batch:
                calc_name = _classify_claim(classifier, item["claim"])
                print(f"  Classifier selected: {calc_name or '(none)'}")

                # Look up formula + type once; reuse across all retry attempts
                retry_formula = ""
                retry_type    = ""
                if calc_name and formula_loader is not None:
                    retry_calc_id = _resolve_calculator_id(calc_name)
                    retry_formula = formula_loader.get_formula(retry_calc_id)
                    retry_type    = formula_loader.get_type(retry_calc_id)

                for attempt in range(3):
                    try:
                        msgs = _build_messages(item, args.ev_limit,
                                               no_think=args.no_think,
                                               calculator_hint=calc_name,
                                               formula_text=retry_formula,
                                               calc_type=retry_type)
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
                        resp = _parse_response(raw, item.get("claim", ""),
                                               formula_text=retry_formula,
                                               calc_type=retry_type)
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
                    "calc":        calc_selected,
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
                    "calculator_selected": row["calc"],
                    "model_reasoning":     row["reasoning"].replace("\n", " "),
                    "dataset_explanation": row["explanation"].replace("\n", " "),
                })

        if args.delay > 0 and batch_start + 1 < total:
            time.sleep(args.delay)

    print_summary(results)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()