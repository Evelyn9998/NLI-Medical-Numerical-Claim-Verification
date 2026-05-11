"""
Evaluation script for three indicators:
  CS  – Calculator Selection
  PE  – Parameter Extraction
  QP  – Quantitative Precision

Inputs:
  mrt_claim_cleaned.csv              (ground truth)
  01_zs_cot_llama_processed_filled.csv  (model output)

Both files are row-aligned (same order, 2814 rows each).
"""

import ast
import json
import re
from dateutil import parser as dateparser
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── 0. Load data ──────────────────────────────────────────────────────────────
mrt = pd.read_csv("mrt_claim_cleaned.csv")
llm = pd.read_csv("01_zs_cot_llama_processed_filled.csv")

assert len(mrt) == len(llm), "Row count mismatch between the two files!"
n = len(mrt)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_relevant_entities(s):
    """Parse the ground-truth Relevant Entities string (Python-literal dict)."""
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def parse_step1(s):
    """Parse the model step1_extracted_params string (JSON dict)."""
    try:
        return json.loads(str(s))
    except Exception:
        try:
            return ast.literal_eval(str(s))
        except Exception:
            return {}


def is_negatively_framed(entity_name: str) -> bool:
    """
    A negatively-framed entity describes absence/negation in its name.
    Examples: 'Cough Absent', 'Absence of cough or coryza',
              'Alternative diagnosis to DVT as likely or more likely'
    """
    nl = entity_name.lower()
    patterns = ["absent", "absence", "alternative diagnosis", "without", r"\bno "]
    return any(re.search(p, nl) for p in patterns)


def is_found(step1_val) -> bool:
    """Return True if the model actually extracted a value (not 'not found')."""
    if step1_val is None:
        return False
    if isinstance(step1_val, str) and step1_val.strip().lower() in (
        "not found", "not applicable", "not calculated", "n/a", ""
    ):
        return False
    return True


def to_bool(val) -> bool | None:
    """Coerce various yes/no/True/False representations to Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def to_numeric(val):
    """
    Extract a float from:
      - a number            → 4.2
      - a list [4.2, 'unit'] → 4.2
      - a list [4.2]         → 4.2
      - a string '4.2'       → 4.2
    Returns None on failure.
    """
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, list) and len(val) >= 1:
            return float(val[0])
        if isinstance(val, str):
            m = re.search(r"[-+]?\d*\.?\d+", val)
            if m:
                return float(m.group())
    except Exception:
        pass
    return None


def _normalize_entity(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(the|a|an)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def find_key_in_step1(entity_name: str, step1_dict: dict):
    """
    Key lookup with two-tier matching:
      1. Exact case-insensitive match
      2. Fuzzy match: strip articles (the/a/an), then check substring containment
    Returns (matched_key, value) or (None, None).
    """
    en_lower = entity_name.lower()
    # 1. exact
    for k, v in step1_dict.items():
        if k.lower() == en_lower:
            return k, v
    # 2. fuzzy
    en_norm = _normalize_entity(entity_name)
    for k, v in step1_dict.items():
        k_norm = _normalize_entity(k)
        if k_norm == en_norm or k_norm in en_norm or en_norm in k_norm:
            return k, v
    return None, None


def entity_is_correct(entity_name: str, gt_val, step1_dict: dict) -> bool:
    """
    Assess whether one entity was correctly extracted.

    Rules
    ─────
    1. Look up entity_name in step1_dict (case-insensitive).
    2. If NOT found (missing key or value == 'not found'):
         • Positively-framed  + gt_val is False/No  → correct
         • Negatively-framed  + gt_val is True/Yes  → correct
         • Otherwise                                 → incorrect
    3. If found:
         a. Boolean gt_val  →  compare to bool(step1_val)
         b. Numeric gt_val  →  compare numeric values
         c. String  gt_val  →  case-insensitive equality
    """
    _, step1_val = find_key_in_step1(entity_name, step1_dict)
    negative = is_negatively_framed(entity_name)

    # ── entity NOT found in step1 ──────────────────────────────────────────
    if not is_found(step1_val):
        if isinstance(gt_val, bool):
            if not negative and gt_val is False:   # positively-framed, absent
                return True
            if negative and gt_val is True:        # negatively-framed, truly absent
                return True
        elif isinstance(gt_val, str):
            gt_bool = to_bool(gt_val)
            if gt_bool is not None:
                if not negative and gt_bool is False:
                    return True
                if negative and gt_bool is True:
                    return True
        return False   # all other not-found cases are incorrect

    # ── entity IS found in step1 ──────────────────────────────────────────

    # Case A: ground truth is boolean
    if isinstance(gt_val, bool):
        step1_bool = to_bool(step1_val)
        if step1_bool is None:
            return False
        return gt_val == step1_bool

    # Case B: ground truth is a numeric list  [value, unit]
    if isinstance(gt_val, list) and len(gt_val) >= 1 and isinstance(gt_val[0], (int, float)):
        gt_num = float(gt_val[0])
        s1_num = to_numeric(step1_val)
        if s1_num is None:
            return False
        # allow a tiny floating-point tolerance
        return abs(gt_num - s1_num) <= 1e-6 * max(1, abs(gt_num))

    # Case C: ground truth is a plain string (e.g. '0.0')
    if isinstance(gt_val, str):
        gt_bool = to_bool(gt_val)
        step1_bool = to_bool(step1_val)
        if gt_bool is not None and step1_bool is not None:
            return gt_bool == step1_bool
        gt_num = to_numeric(gt_val)
        s1_num = to_numeric(step1_val)
        if gt_num is not None and s1_num is not None:
            return abs(gt_num - s1_num) <= 1e-6 * max(1, abs(gt_num))
        # fall back to string equality
        return str(gt_val).strip().lower() == str(step1_val).strip().lower()

    return False


# ── Calculator name → ID resolver ────────────────────────────────────────────

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
    "estimated date of conception": 68, "estimated of conception": 68,
}

_MATCH_STOP = frozenset({
    'the', 'a', 'an', 'for', 'of', 'and', 'or', 'score', 'index',
    'calculation', 'calculations', 'calculator', 'criteria', 'modified',
    'based', 's', 'by', 'rule',
})

_CALC_NAME_TO_ID: dict = {}


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
    if best_overlap >= 2:
        return best_cid
    return 0


# ── 0. TF – Task Fulfillment ──────────────────────────────────────────────────

def compute_tf(llm_row) -> bool:
    return str(llm_row["predicted_label"]).strip().lower() != "unknown"


# ── 1. CS – Calculator Selection ──────────────────────────────────────────────

def compute_cs(mrt_row, llm_row) -> bool:
    gt_id   = int(mrt_row["Calculator ID"])
    pred_id = _resolve_calculator_id(str(llm_row["calculator_selected"]))
    return gt_id == pred_id


# ── 2. PE – Parameter Extraction ─────────────────────────────────────────────

def compute_pe(mrt_row, llm_row) -> bool:
    gt_entities = parse_relevant_entities(mrt_row["Relevant Entities"])
    step1       = parse_step1(llm_row["step1_extracted_params"])

    if not gt_entities:          # nothing to evaluate → skip
        return np.nan

    all_correct = True
    for entity_name, gt_val in gt_entities.items():
        if not entity_is_correct(entity_name, gt_val, step1):
            all_correct = False
            break
    return all_correct


# ── 3. QP – Quantitative Precision ───────────────────────────────────────────

def _compare_dates(pred_str: str, gt_str: str) -> bool:
    """Compare date strings after normalizing to date objects."""
    try:
        return dateparser.parse(pred_str).date() == dateparser.parse(gt_str).date()
    except Exception:
        return False


def _compare_gestational_age(pred_str: str, gt_str: str) -> bool:
    """Compare gestational age strings like \"('28 weeks', '4 days')\" by total days."""
    try:
        def to_days(s):
            t = ast.literal_eval(s)
            weeks = int(re.search(r'\d+', t[0]).group())
            days  = int(re.search(r'\d+', t[1]).group())
            return weeks * 7 + days
        return to_days(pred_str) == to_days(gt_str)
    except Exception:
        return False


def compute_qp(mrt_row, llm_row) -> bool:
    step2_raw = str(llm_row["step2_computed_value"]).strip()
    gt_lo     = str(mrt_row["Lower Limit"]).strip()
    gt_hi     = str(mrt_row["Upper Limit"]).strip()

    # numeric path
    try:
        lo = float(gt_lo)
        hi = float(gt_hi)
        step2_num = to_numeric(step2_raw)
        if step2_num is None:
            return False
        return lo <= step2_num <= hi
    except (ValueError, TypeError):
        pass

    # gestational age path: "('28 weeks', '4 days')"
    if gt_lo.startswith("("):
        return _compare_gestational_age(step2_raw, gt_lo)

    # date path
    return _compare_dates(step2_raw, gt_lo)


# ── Main loop ─────────────────────────────────────────────────────────────────

results = []
for i in range(n):
    mrt_row = mrt.iloc[i]
    llm_row = llm.iloc[i]

    tf = compute_tf(llm_row)
    cs = compute_cs(mrt_row, llm_row)
    pe = compute_pe(mrt_row, llm_row)
    qp = compute_qp(mrt_row, llm_row)

    results.append({
        "row_idx":           i,
        "mrt_row_number":    mrt_row["Row Number"],
        "calculator_gt":     mrt_row["Calculator Name"],
        "calculator_pred":   llm_row["calculator_selected"],
        "TF":                tf,
        "CS":                cs,
        "PE":                pe,
        "QP":                qp,
    })

results_df = pd.DataFrame(results)

# ── Summary ───────────────────────────────────────────────────────────────────

tf_acc = results_df["TF"].mean()
cs_acc = results_df["CS"].mean()
pe_valid = results_df["PE"].dropna()
pe_acc = pe_valid.mean()
qp_acc = results_df["QP"].dropna().mean()

print("=" * 52)
print("  Indicator Summary")
print("=" * 52)
print(f"  Total rows evaluated : {n}")
print()
print(f"  TF (Task Fulfillment)")
print(f"    Success  : {results_df['TF'].sum():.0f} / {n}")
print(f"    Accuracy : {tf_acc:.4f}  ({tf_acc*100:.2f}%)")
print()
print(f"  CS (Calculator Selection)")
print(f"    Correct  : {results_df['CS'].sum():.0f} / {n}")
print(f"    Accuracy : {cs_acc:.4f}  ({cs_acc*100:.2f}%)")
print()
print(f"  PE (Parameter Extraction)")
print(f"    Correct  : {pe_valid.sum():.0f} / {n}")
print(f"    Accuracy : {pe_valid.sum()/n:.4f}  ({pe_valid.sum()/n*100:.2f}%)")
print()
print(f"  QP (Quantitative Precision)")
qp_series = results_df["QP"].dropna()
print(f"    Correct  : {qp_series.sum():.0f} / {n}")
print(f"    Accuracy : {qp_series.sum()/n:.4f}  ({qp_series.sum()/n*100:.2f}%)")
print("=" * 52)
