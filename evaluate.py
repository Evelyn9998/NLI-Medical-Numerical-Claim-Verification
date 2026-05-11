"""
evaluate.py – Preprocess (if needed) + Evaluate in one step.

Usage
-----
  python evaluate.py <result_csv> [--gt mrt_claim_cleaned.csv] [--name display_name] [--save output.csv]

What it does
------------
  1. Loads the result CSV.
  2. Detects which of the three columns are missing:
       • calculator_selected
       • step1_extracted_params
       • step2_computed_value
     Fills any missing ones from model_reasoning; skips if all present.
  3. Runs full evaluation (TF / CS / PE / QP + classification report).
"""

import argparse
import ast
import json
import re
import warnings
from datetime import datetime
from dateutil import parser as dateparser

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── PART 1  Preprocessing  (3columns logic inlined) ──────────────────────────

def extract_calculator_selected(formula: str) -> str:
    if not formula or formula.strip().lower() in ("not applicable", "not found", "", "none"):
        return "not applicable"
    formula = formula.strip()
    formula = re.sub(
        r"\s*(calculation|formula)\s+not\s+provided.*", "", formula, flags=re.IGNORECASE
    ).strip()
    m = re.match(r"^(.+?)\s*(?:=|:)\s", formula)
    if m:
        return m.group(1).strip()
    return formula.split("(")[0].strip()[:80]


def parse_evidence_value(ev):
    if ev is None:
        return "not found"
    if isinstance(ev, (int, float)):
        return [float(ev)]
    ev_str = str(ev).strip()
    if not ev_str:
        return "not found"
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", ev_str):
        return ev_str
    bp = re.match(r"^(\d{2,3})/(\d{2,3})\s*(.*)$", ev_str)
    if bp:
        systolic, diastolic, unit = float(bp.group(1)), float(bp.group(2)), bp.group(3).strip()
        if 40 <= diastolic <= 130:
            return [systolic, unit if unit else "mmHg"]
    rate = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)\s*(/\s*\w+)\s*$", ev_str)
    if rate:
        return [float(rate.group(1)), rate.group(2).strip()]
    num_unit = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)\s+(.+)$", ev_str)
    if num_unit:
        return [float(num_unit.group(1)), num_unit.group(2).strip()]
    if re.match(r"^-?[0-9]+(?:\.[0-9]+)?$", ev_str):
        return [float(ev_str)]
    return ev_str


def extract_step1_params(entities: list) -> dict:
    params = {}
    for e in entities:
        name = e.get("name", "")
        if name:
            params[name] = parse_evidence_value(e.get("evidence_value", "not found"))
    return params


def _days_to_gest_tuple(total_days) -> str:
    weeks, days = divmod(round(total_days), 7)
    return f"('{weeks} weeks', '{days} days')"


def _weeks_to_gest_tuple(total_weeks) -> str:
    whole_weeks    = int(total_weeks)
    remaining_days = round((total_weeks - whole_weeks) * 7)
    if remaining_days == 7:
        whole_weeks += 1
        remaining_days = 0
    return f"('{whole_weeks} weeks', '{remaining_days} days')"


def _parse_gestational_age(s: str):
    SEP = r"\s*(?:and|,|&)?\s*"
    DPY, DPM, DPW = 365, 30, 7

    m = re.match(r"^\(\s*(\d+)\s+weeks?\s*,\s*(\d+)\s+days?\s*\)$", s, re.IGNORECASE)
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"
    m = re.match(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$", s)
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"
    m = re.match(r"^(\d+)\s+days?,\s*\d+:\d+:\d+$", s, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1)))

    core = re.sub(r"^approximately\s+", "", s, flags=re.IGNORECASE).strip()

    m = re.match(r"^(\d+(?:\.\d+)?)\s+weeks?$", core, re.IGNORECASE)
    if m:
        return _weeks_to_gest_tuple(float(m.group(1)))
    m = re.match(r"^(\d+(?:\.\d+)?)\s+days?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(float(m.group(1)))
    m = re.match(r"^(\d+)\s+weeks?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE)
    if m:
        return f"('{m.group(1)} weeks', '{m.group(2)} days')"
    m = re.match(r"^(\d+)\s+years?" + SEP + r"(\d+)\s+months?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPY + int(m.group(2))*DPM + int(m.group(3)))
    m = re.match(r"^(\d+)\s+years?" + SEP + r"(\d+)\s+months?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPY + int(m.group(2))*DPM)
    m = re.match(r"^(\d+)\s+years?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPY)
    m = re.match(r"^(\d+)\s+months?" + SEP + r"(\d+)\s+weeks?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPM + int(m.group(2))*DPW + int(m.group(3)))
    m = re.match(r"^(\d+)\s+months?" + SEP + r"(\d+)\s+weeks?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPM + int(m.group(2))*DPW)
    m = re.match(r"^(\d+)\s+months?" + SEP + r"(\d+)\s+days?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPM + int(m.group(2)))
    m = re.match(r"^(\d+)\s+months?$", core, re.IGNORECASE)
    if m:
        return _days_to_gest_tuple(int(m.group(1))*DPM)
    return None


def parse_computed_value(raw):
    if raw is None:
        return "not calculated"
    s = str(raw).strip()
    if not s or s.lower() in ("none", ""):
        return "not calculated"
    if s.lower() in ("not applicable", "n/a"):
        return "not applicable"
    if s.lower() in ("not calculated", "not found",
                     "no correct result as the evidence does not match the claim"):
        return s
    dt = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:\s+\d{2}:\d{2}:\d{2})?$", s)
    if dt:
        return f"{dt.group(2)}/{dt.group(3)}/{dt.group(1)}"
    gest = _parse_gestational_age(s)
    if gest is not None:
        return gest
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", s):
        return s
    m = re.match(r"^(-?[0-9]+(?:\.[0-9]+)?)", s)
    if m:
        v = m.group(1)
        return int(v) if "." not in v else float(v)
    return s


def _regex_entities(text: str) -> list:
    entities = []
    for chunk in re.findall(r"\{[^{}]*?\"name\"[^{}]*?\}", text, re.DOTALL):
        try:
            entities.append(json.loads(chunk))
        except json.JSONDecodeError:
            name = re.search(r'"name":\s*"([^"]+)"', chunk)
            ev   = re.search(r'"evidence_value":\s*"([^"]*)"', chunk)
            if name and ev:
                entities.append({"name": name.group(1), "evidence_value": ev.group(1)})
    return entities


def _regex_formula(text: str):
    m = re.search(r'"formula":\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _regex_correct_result(text: str):
    m = re.search(r'"correct_result":\s*(?:"([^"]*)"|(-?[0-9]+(?:\.[0-9]*)?))', text)
    if m:
        return m.group(1) if m.group(1) is not None else m.group(2)
    return None


def extract_from_reasoning(reasoning) -> tuple:
    if pd.isna(reasoning) or str(reasoning).strip() == "":
        return "not applicable", {}, "not calculated"
    text = str(reasoning)
    try:
        d          = json.loads(text)
        entities   = d.get("step1_entities", d.get("entities", []))
        calc_block = d.get("step2_calculation", d.get("calculation", {}))
        formula    = calc_block.get("formula", "")
        raw_result = calc_block.get("correct_result", None)
    except json.JSONDecodeError:
        entities   = _regex_entities(text)
        formula    = _regex_formula(text)
        raw_result = _regex_correct_result(text)
    return (
        extract_calculator_selected(formula),
        extract_step1_params(entities),
        parse_computed_value(raw_result),
    )


def _fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    need_calc = "calculator_selected"    not in df.columns
    need_pe   = "step1_extracted_params" not in df.columns
    need_qp   = "step2_computed_value"   not in df.columns

    if not (need_calc or need_pe or need_qp):
        print("[evaluate] All three columns present — skipping preprocessing.")
        return df

    missing = [c for c, flag in [
        ("calculator_selected",    need_calc),
        ("step1_extracted_params", need_pe),
        ("step2_computed_value",   need_qp),
    ] if flag]
    print(f"[evaluate] Adding missing column(s): {missing}")

    calculators, params_list, values = [], [], []
    for _, row in df.iterrows():
        calc, params, value = extract_from_reasoning(row["model_reasoning"])
        calculators.append(calc)
        params_list.append(json.dumps(params))
        values.append(value)

    df = df.copy()
    if need_calc: df["calculator_selected"]    = calculators
    if need_pe:   df["step1_extracted_params"] = params_list
    if need_qp:   df["step2_computed_value"]   = values
    return df


# ── PART 2  Evaluation ────────────────────────────────────────────────────────

def parse_relevant_entities(s):
    try:    return ast.literal_eval(str(s))
    except: return {}


def parse_step1(s):
    try:    return json.loads(str(s))
    except:
        try:    return ast.literal_eval(str(s))
        except: return {}


def is_negatively_framed(entity_name: str) -> bool:
    nl = entity_name.lower()
    return any(re.search(p, nl) for p in
               ["absent", "absence", "alternative diagnosis", "without", r"\bno "])


def is_found(v) -> bool:
    if v is None: return False
    if isinstance(v, str) and v.strip().lower() in (
        "not found", "not applicable", "not calculated", "n/a", ""
    ): return False
    return True


def to_bool(val):
    if isinstance(val, bool):         return val
    if isinstance(val, (int, float)): return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true",  "yes", "1"): return True
        if v in ("false", "no",  "0"): return False
    return None


def to_numeric(val):
    try:
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, list) and val: return float(val[0])
        if isinstance(val, str):
            m = re.search(r"[-+]?\d*\.?\d+", val)
            if m: return float(m.group())
    except: pass
    return None


def _normalize_entity(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'\b(the|a|an)\b', '', s.lower())).strip()


def find_key_in_step1(entity_name: str, step1_dict: dict):
    en_lower = entity_name.lower()
    for k, v in step1_dict.items():
        if k.lower() == en_lower: return k, v
    en_norm = _normalize_entity(entity_name)
    for k, v in step1_dict.items():
        k_norm = _normalize_entity(k)
        if k_norm == en_norm or k_norm in en_norm or en_norm in k_norm: return k, v
    return None, None


def entity_is_correct(entity_name, gt_val, step1_dict) -> bool:
    _, step1_val = find_key_in_step1(entity_name, step1_dict)
    negative = is_negatively_framed(entity_name)
    if not is_found(step1_val):
        if isinstance(gt_val, bool):
            if not negative and gt_val is False: return True
            if negative     and gt_val is True:  return True
        elif isinstance(gt_val, str):
            gb = to_bool(gt_val)
            if gb is not None:
                if not negative and gb is False: return True
                if negative     and gb is True:  return True
        return False
    if isinstance(gt_val, bool):
        sb = to_bool(step1_val)
        return sb is not None and gt_val == sb
    if isinstance(gt_val, list) and gt_val and isinstance(gt_val[0], (int, float)):
        sn = to_numeric(step1_val)
        return sn is not None and abs(float(gt_val[0]) - sn) <= 1e-6 * max(1, abs(float(gt_val[0])))
    if isinstance(gt_val, str):
        gb, sb = to_bool(gt_val), to_bool(step1_val)
        if gb is not None and sb is not None: return gb == sb
        gn, sn = to_numeric(gt_val), to_numeric(step1_val)
        if gn is not None and sn is not None:
            return abs(gn - sn) <= 1e-6 * max(1, abs(gn))
        return str(gt_val).strip().lower() == str(step1_val).strip().lower()
    return False


# ── Calculator ID resolver ────────────────────────────────────────────────────

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
    "ibw": 10, "ideal body weight": 10,
    "bmi": 6, "body mass index": 6,
    "gfr": 9, "mdrd gfr": 9, "mdrd gfr equation": 9,
    "ckd-epi": 3, "ckd epi": 3,
    "cockcroft-gault": 2, "cockcroft gault": 2, "creatinine clearance": 2,
    "corrected calcium": 7, "calcium correction": 7, "corrected calcium concentration": 7,
    "homa-ir": 31, "homa ir": 31,
    "corrected qt interval": 11, "qtc": 11, "qtc bazett": 11,
    "qtc fridericia": 56, "fridericia": 56,
    "qtc framingham": 57, "framingham qtc": 57,
    "qtc hodges": 58, "hodges": 58,
    "qtc rautaharju": 59, "rautaharju": 59,
    "ldl cholesterol": 44, "ldl calculated": 44, "ldl": 44,
    "map": 5, "mean arterial pressure": 5,
    "serum osmolality": 30, "calculated serum osmolality": 30,
    "well's dvt score": 16, "well's score for dvt": 16,
    "well's score for pulmonary embolism": 8, "well's pe score": 8,
    "cardiac risk index": 17, "cardiac risk index score": 17, "revised cardiac risk index": 17,
    "has-bled score": 25, "hasbled": 25,
    "curb-65 score": 45, "curb65": 45,
    "cha2ds2-vasc score": 4, "cha2ds2 vasc": 4,
    "10-year risk percentage of mi or death": 46,
}

_MATCH_STOP = frozenset({
    'the', 'a', 'an', 'for', 'of', 'and', 'or', 'score', 'index',
    'calculation', 'calculations', 'calculator', 'criteria', 'modified',
    'based', 's', 'by', 'rule',
})

_CALC_NAME_TO_ID: dict = {}
_CS_MATCH_STATS: dict  = {"exact": 0, "alias": 0, "substring": 0, "word_overlap": 0, "no_match": 0}


def _build_name_to_id() -> dict:
    mapping: dict = {}
    for line in _CALCULATOR_LIST.strip().splitlines():
        m = re.match(r"ID\s+(\d+)\s+[—–-]+\s+(.+?)(?::|$)", line.strip())
        if m:
            mapping[m.group(2).strip().lower()] = int(m.group(1))
    return mapping


def _resolve_calculator_id(calc_name: str) -> tuple:
    global _CALC_NAME_TO_ID
    if not _CALC_NAME_TO_ID:
        _CALC_NAME_TO_ID = _build_name_to_id()
    key = calc_name.lower().strip()
    if key in _CALC_NAME_TO_ID:
        _CS_MATCH_STATS["exact"] += 1;        return _CALC_NAME_TO_ID[key], "exact"
    if key in _CALC_ALIASES:
        _CS_MATCH_STATS["alias"] += 1;        return _CALC_ALIASES[key], "alias"
    for alias, cid in _CALC_ALIASES.items():
        if alias in key or key in alias:
            _CS_MATCH_STATS["alias"] += 1;    return cid, "alias"
    for name, cid in sorted(_CALC_NAME_TO_ID.items(), key=lambda x: len(x[0]), reverse=True):
        if name in key or key in name:
            _CS_MATCH_STATS["substring"] += 1; return cid, "substring"
    key_words = set(re.findall(r'[a-z0-9]+', key)) - _MATCH_STOP
    best_cid, best_overlap = 0, 0
    for name, cid in _CALC_NAME_TO_ID.items():
        overlap = len(key_words & (set(re.findall(r'[a-z0-9]+', name)) - _MATCH_STOP))
        if overlap > best_overlap:
            best_overlap, best_cid = overlap, cid
    if best_overlap >= 2:
        _CS_MATCH_STATS["word_overlap"] += 1; return best_cid, "word_overlap"
    _CS_MATCH_STATS["no_match"] += 1;         return 0, "no_match"


# ── Metric computers ──────────────────────────────────────────────────────────

def compute_tf(row) -> bool:
    return str(row["predicted_label"]).strip().lower() != "unknown"


def compute_cs(mrt_row, llm_row):
    pred_id, tier = _resolve_calculator_id(str(llm_row["calculator_selected"]))
    return int(mrt_row["Calculator ID"]) == pred_id, tier


def compute_pe(mrt_row, llm_row):
    gt = parse_relevant_entities(mrt_row["Relevant Entities"])
    s1 = parse_step1(llm_row["step1_extracted_params"])
    if not gt: return np.nan
    return all(entity_is_correct(n, v, s1) for n, v in gt.items())


def _compare_dates(pred_str: str, gt_str: str) -> bool:
    """Compare date strings; tries both MM/DD/YYYY and DD/MM/YYYY for ambiguous pred dates."""
    try:
        gt_date = dateparser.parse(gt_str).date()
    except Exception:
        return False
    try:
        if dateparser.parse(pred_str).date() == gt_date:
            return True
    except Exception:
        pass
    try:
        if datetime.strptime(pred_str, "%d/%m/%Y").date() == gt_date:
            return True
    except Exception:
        pass
    return False


def _compare_gestational_age(pred_str: str, gt_str: str) -> bool:
    try:
        def to_days(s):
            t = ast.literal_eval(s)
            return int(re.search(r'\d+', t[0]).group()) * 7 + int(re.search(r'\d+', t[1]).group())
        return to_days(pred_str) == to_days(gt_str)
    except: return False


def compute_qp(mrt_row, llm_row) -> bool:
    s2 = str(llm_row["step2_computed_value"]).strip()
    lo = str(mrt_row["Lower Limit"]).strip()
    hi = str(mrt_row["Upper Limit"]).strip()
    try:
        n = to_numeric(s2)
        if n is None: return False
        return float(lo) <= n <= float(hi)
    except (ValueError, TypeError): pass
    if lo.startswith("("): return _compare_gestational_age(s2, lo)
    return _compare_dates(s2, lo)


# ── Classification metrics ────────────────────────────────────────────────────

def compute_metrics(true_labels, pred_labels):
    classes = sorted(set(true_labels))
    n       = len(true_labels)
    per_class = {}
    for cls in classes:
        tp      = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp      = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn      = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support = sum(1 for t in true_labels if t == cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    macro_p  = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / len(classes)
    total    = sum(v["support"]   for v in per_class.values())
    w_p  = sum(v["precision"]*v["support"] for v in per_class.values()) / total
    w_r  = sum(v["recall"]   *v["support"] for v in per_class.values()) / total
    w_f1 = sum(v["f1"]       *v["support"] for v in per_class.values()) / total
    acc  = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / n if n else 0.0
    return {
        "per_class": per_class,
        "macro":    {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": w_p,     "recall": w_r,     "f1": w_f1},
        "accuracy": acc,
        "n_total":   n,
        "n_unknown": sum(1 for p in pred_labels if p == "unknown"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _qp_is_missing(llm_row) -> bool:
    s = str(llm_row["step2_computed_value"]).strip().lower()
    return s in ("not calculated", "not found", "nan", "", "none", "not applicable", "n/a")


def _pe_is_unparseable(llm_row) -> bool:
    return parse_step1(llm_row["step1_extracted_params"]) == {}


def _derive_name(path: str) -> str:
    stem = re.split(r'[/\\]', path)[-1]
    return re.sub(r'\.csv$', '', stem, flags=re.IGNORECASE)


# ── Combined report ───────────────────────────────────────────────────────────

def run_evaluation(llm: pd.DataFrame, mrt: pd.DataFrame, label: str, name: str = None):
    for k in _CS_MATCH_STATS: _CS_MATCH_STATS[k] = 0
    assert len(mrt) == len(llm), (
        f"Row count mismatch: ground truth {len(mrt)} rows, result {len(llm)} rows."
    )

    name   = name or _derive_name(label)
    n      = len(mrt)
    has_cs = "calculator_selected"    in llm.columns
    has_pe = "step1_extracted_params" in llm.columns
    has_qp = "step2_computed_value"   in llm.columns

    tf_list, cs_list, pe_list, qp_list = [], [], [], []
    pe_unparseable = 0
    qp_missing     = 0

    for i in range(n):
        mrt_row, llm_row = mrt.iloc[i], llm.iloc[i]
        tf_list.append(compute_tf(llm_row))
        if has_cs:
            cs_list.append(compute_cs(mrt_row, llm_row))
        if has_pe:
            if _pe_is_unparseable(llm_row): pe_unparseable += 1
            pe_list.append(compute_pe(mrt_row, llm_row))
        if has_qp:
            if _qp_is_missing(llm_row): qp_missing += 1
            qp_list.append(compute_qp(mrt_row, llm_row))

    true_labels = llm["true_label"].str.strip().str.lower().tolist()
    pred_labels = llm["predicted_label"].str.strip().str.lower().tolist()
    clf         = compute_metrics(true_labels, pred_labels)
    pc          = clf["per_class"]
    classes     = sorted(pc.keys())
    col_w       = max(len(c) for c in classes) + 2
    W           = 60

    print("=" * W)
    print(f"  {name}")
    print("=" * W)
    print(f"  Total samples : {n}")
    print(f"  Unknown preds : {clf['n_unknown']} (treated as wrong)")
    print(f"  Accuracy      : {clf['accuracy']:.4f}")
    print()

    hdr = f"  {'Class':<{col_w}}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cls in classes:
        v = pc[cls]
        print(f"  {cls:<{col_w}}  {v['precision']:>10.4f}  {v['recall']:>10.4f}"
              f"  {v['f1']:>10.4f}  {v['support']:>8}")
    print()
    print(f"  {'Macro avg':<{col_w}}  {clf['macro']['precision']:>10.4f}"
          f"  {clf['macro']['recall']:>10.4f}  {clf['macro']['f1']:>10.4f}")
    print(f"  {'Weighted avg':<{col_w}}  {clf['weighted']['precision']:>10.4f}"
          f"  {clf['weighted']['recall']:>10.4f}  {clf['weighted']['f1']:>10.4f}")
    print()

    print(f"  --- Paper Metrics ({name}) ---")
    tf_ok = sum(tf_list)
    print(f"  TF (Task Fulfillment)       : {tf_ok/n*100:.2f}%  ({tf_ok}/{n} tasks with valid output)")
    if has_cs:
        cs_ok = sum(c for c, _ in cs_list)
        print(f"  CS (Calculator Selection)   : {cs_ok/n*100:.2f}%  ({cs_ok}/{n} correct)")
    if has_pe:
        pe_ok = int(pd.Series(pe_list).dropna().sum())
        print(f"  PE (Parameter Extraction)   : {pe_ok/n*100:.2f}%  "
              f"({pe_ok}/{n} rows fully correct, {pe_unparseable} unparseable)")
    if has_qp:
        qp_ok = int(pd.Series(qp_list).dropna().sum())
        print(f"  QP (Quantitative Precision) : {qp_ok/n*100:.2f}%  "
              f"({qp_ok}/{n} within tolerance, {qp_missing} missing/unparseable)")
    print("=" * W)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Preprocess (add missing columns) then evaluate a model result CSV."
    )
    ap.add_argument("result_file",
                    help="Model result CSV.")
    ap.add_argument("--gt", default="mrt_claim_cleaned.csv",
                    help="Ground-truth CSV (default: mrt_claim_cleaned.csv).")
    ap.add_argument("--name", default=None,
                    help="Display name for the model (default: derived from filename).")
    ap.add_argument("--save", metavar="OUTPUT_CSV",
                    help="Save the preprocessed CSV to this path before evaluating.")
    args = ap.parse_args()

    llm = pd.read_csv(args.result_file)
    llm = _fill_missing_columns(llm)

    if args.save:
        llm.to_csv(args.save, index=False)
        print(f"[evaluate] Saved preprocessed CSV → {args.save}")

    mrt = pd.read_csv(args.gt)
    run_evaluation(llm, mrt, label=args.result_file, name=args.name)
